import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "./resource/pretrained"  # 事前学習モデルの保存先指定

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from sentence_transformers import CrossEncoder

from modules.bm25_search import BM25DocumentSearch
from modules.rag_prompt import RAG_PROMPT_TEMPLATE

# 環境変数の読み込み
load_dotenv()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    persist_directory = Path(os.getenv("DATASET_DIR", "./resource/chroma"))

    # 出力先フォルダの作成
    output_path = Path("./results")
    output_path.mkdir(exist_ok=True)

    # 質問データの読み込み
    question_df = pd.read_csv("./share/質問回答/questions_test.csv", encoding="utf-8")

    # モデルのセットアップ
    ## Ollama-Cloudを使用
    generate_mode = os.getenv("GENERATE_MODE", "ollama")
    if generate_mode == "ollama":
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL_NAME", "gpt-oss:120b"),
            base_url="https://ollama.com",
            api_key=os.getenv("OLLAMA_API_KEY", None),
            temperature=0.0,
        )
        ## ローカルLLMを使用
        # llm = ChatOllama(
        #     model=os.getenv("OLLAMA_MODEL_NAME", "gpt-oss:120b"),
        #     temperature=0.0,
        # )
    elif generate_mode == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL_NAME", "models/gemini-3.1-flash-lite"),
            temperature=0.0,
            thinking_budget=4096,
        )

    # Embeddingモデルの読み込み
    embedding_mode = os.getenv("EMBEDDING_MODE", "huggingface")
    if embedding_mode == "huggingface":
        embedding = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL_NAME", None),
            model_kwargs={"device": "cuda", "trust_remote_code": True},
        )
    elif embedding_mode == "ollama":
        embedding = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", None))
    else:
        raise ValueError(f"Invalid EMBEDDING_MODE: {embedding_mode}")

    # ベクトルDBの読み込み
    vectorstore_all = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory,
        collection_name="shared_folder_all_documents",
    )
    vectorstore_csv = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory,
        collection_name="shared_folder_csv_documents",
    )
    vectorstore_excel = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory,
        collection_name="shared_folder_excel_documents",
    )
    vectorstore_file_info = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory,
        collection_name="shared_folder_file_info_list",
    )
    # BM25Retrieverの読み込み
    bm25 = BM25DocumentSearch(dir_path=persist_directory, k=30)
    # Rerankerモデルの読み込み
    reranker = CrossEncoder(
        os.getenv("RERANKER_MODEL_NAME", None),
        device="cuda",
    )

    # 途中再開か新規作成かで分岐
    if args.resume:
        answer_df = pd.read_csv(output_path / "result_generated.csv", encoding="utf-8-sig")
        answers = answer_df.to_dict(orient="list")
    else:
        answers = {"index": [], "question": [], "answer": [], "reason": [], "files": []}

    # 質問に対する回答の生成
    for idx, row in question_df.iterrows():
        start_time = time.perf_counter()
        input_question = row["question"]
        print(idx, input_question)

        if row["index"] in answers["index"]:
            print("  回答済みのためスキップ")
            continue

        # ファイル情報のベクトルDBから検索対象ファイルを取得
        file_info_docs = vectorstore_file_info.similarity_search(query=input_question, k=5)
        # 検索結果からファイルパスを拡張子ごとに取得
        files, csv_files, excel_files = [], [], []
        for doc in file_info_docs:
            source = doc.metadata["source"]
            ext = Path(source).suffix
            if ext == ".csv":
                csv_files.append(source)
            elif ext == ".xlsx":
                excel_files.append(source)
            else:
                files.append(source)

        # ベクトルDBから検索
        docs = vectorstore_all.max_marginal_relevance_search(
            query=input_question,
            k=50,
            fetch_k=100,
            lambda_mult=0.5,
        )
        if len(files) > 0:
            docs += vectorstore_all.max_marginal_relevance_search(
                query=input_question,
                k=20,
                fetch_k=100,
                lambda_mult=0.5,
                filter={"source": {"$in": files}},
            )
        if len(csv_files) > 0:
            docs += vectorstore_csv.max_marginal_relevance_search(
                query=input_question,
                k=20,
                fetch_k=50,
                lambda_mult=0.5,
                filter={"source": {"$in": csv_files}},
            )
        if len(excel_files) > 0:
            docs += vectorstore_excel.max_marginal_relevance_search(
                query=input_question,
                k=20,
                fetch_k=50,
                lambda_mult=0.5,
                filter={"source": {"$in": excel_files}},
            )

        # BM25Retrieverから検索
        docs += bm25(input_question)

        # page_contentが重複するドキュメントを削除
        seen_contents = set()
        unique_docs = []
        for doc in docs:
            content = doc.page_content
            if content not in seen_contents:
                seen_contents.add(content)
                unique_docs.append(doc)
        docs = unique_docs

        # 検索した類似文書をリランキング
        question_answer_list = [
            (input_question, f"{doc.metadata}\n{doc.page_content}") for doc in docs
        ]
        scores = reranker.predict(question_answer_list)
        reranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in reranked_docs][:10]

        # RAGプロンプトに埋め込むために成形
        context = "\n".join(
            [f"ContextNo.{i + 1}:\n{doc.page_content}" for i, doc in enumerate(top_docs)]
        )
        # 入力内容を作成
        input_messages = [
            HumanMessage(
                content=RAG_PROMPT_TEMPLATE.format(question=input_question, context=context)
            )
        ]

        # 返答の生成
        response = llm.invoke(input_messages)

        if generate_mode == "gemini":
            response_text = response.content[0]["text"]
        else:
            response_text = response.content

        # 返答の変換
        try:
            response_dict = json.loads(response_text)
        except Exception as e:
            response_dict = {
                "answer": response.content.replace("\n", ""),
                "reason": f"jsonのパース失敗({e})",
            }
        print(f"回答: {response_dict}")

        # 生成された返答内容を格納
        answers["index"].append(row["index"])
        answers["question"].append(input_question)
        answers["answer"].append(response_dict["answer"].replace("\n", ""))
        answers["reason"].append(response_dict["reason"])
        answers["files"].append(",".join([doc.metadata["source"] for doc in top_docs]))

        generate_time = time.perf_counter() - start_time
        print(f"返答の生成時間: {generate_time:.2f}s")

        answer_df = pd.DataFrame(answers)
        # 確認用ファイルを作成
        answer_df.to_csv(
            output_path / "result_generated.csv",
            encoding="utf-8-sig",
            index=None,
        )

        # 提出用ファイルを作成
        answer_df[["index", "answer"]].to_csv(
            output_path / "predictions.csv", encoding="utf-8-sig", index=None, header=None
        )


if __name__ == "__main__":
    main()
