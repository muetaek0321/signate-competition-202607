import json
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "./resource/pretrained"  # 事前学習モデルの保存先指定

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

from modules.rag_prompt import RAG_PROMPT_TEMPLATE

# 環境変数の読み込み
load_dotenv()


def main() -> None:
    # 出力先フォルダの作成
    output_path = Path("./results")
    output_path.mkdir(exist_ok=True)

    # 質問データの読み込み
    # question_df = pd.read_csv("./share/質問回答/questions_valid.csv", encoding="utf-8")
    question_df = pd.read_csv("./share/質問回答/questions_test.csv", encoding="utf-8")

    # モデルのセットアップ
    ## Ollama-Cloudを使用
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

    # Embeddingモデルの読み込み
    embedding = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL_NAME", None),
        model_kwargs={"device": "cuda", "trust_remote_code": True},
    )
    # ベクトルDBの読み込み
    vectorstore_all = Chroma(
        embedding_function=embedding,
        persist_directory=os.getenv("DATASET_DIR", "./resource/chroma"),
        collection_name="shared_folder_all_documents",
    )
    vectorstore_csv = Chroma(
        embedding_function=embedding,
        persist_directory=os.getenv("DATASET_DIR", "./resource/chroma"),
        collection_name="shared_folder_csv_documents",
    )
    vectorstore_excel = Chroma(
        embedding_function=embedding,
        persist_directory=os.getenv("DATASET_DIR", "./resource/chroma"),
        collection_name="shared_folder_excel_documents",
    )
    # # BM25Retrieverの読み込み
    # saved_docs = joblib.load("./resource/all_documents.joblib")
    # bm25_retriever = BM25Retriever.from_texts(
    #     saved_docs,
    #     preprocess_func=,
    #     k=50,
    # )
    # Rerankerモデルの読み込み
    reranker = CrossEncoder(
        os.getenv("RERANKER_MODEL_NAME", None),
        device="cuda",
    )

    # 質問に対する回答の生成
    answers = {"index": [], "question": [], "answer": [], "reason": [], "files": []}
    for idx, row in question_df.iterrows():
        start_time = time.perf_counter()
        input_question = row["question"]
        print(idx, input_question)

        # ベクトルDBから検索
        # docs = vectorstore.similarity_search(query=input_question, k=50)
        docs = vectorstore_all.max_marginal_relevance_search(
            query=input_question,
            k=50,
            fetch_k=100,
            lambda_mult=0.5,
        )
        if "csv" in input_question.lower():
            docs += vectorstore_csv.max_marginal_relevance_search(
                query=input_question,
                k=20,
                fetch_k=50,
                lambda_mult=0.5,
            )
        if "excel" in input_question.lower() or "xlsx" in input_question.lower():
            docs += vectorstore_excel.max_marginal_relevance_search(
                query=input_question,
                k=20,
                fetch_k=50,
                lambda_mult=0.5,
            )

        # ベクトル検索した類似文書をリランキング
        question_answer_list = [
            (input_question, f"{doc.metadata}\n{doc.page_content}") for doc in docs
        ]
        scores = reranker.predict(question_answer_list)
        reranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        source_list = [doc.metadata["source"] for doc, _ in reranked_docs]
        source_list = sorted(set(source_list), key=source_list.index)[:5]

        top_docs = []
        for source in source_list:
            ext = Path(source).suffix
            if ext in [".csv", ".tsv"]:
                # "source"が一致するもののうち上位10件を取得
                top_docs += [doc for doc, _ in reranked_docs if doc.metadata["source"] == source][
                    :5
                ]
            else:
                # "source"が一致するもののうち上位3件を取得
                top_docs += [doc for doc, _ in reranked_docs if doc.metadata["source"] == source][
                    :3
                ]

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

        # 返答の変換
        try:
            response_dict = json.loads(response.content)
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
        answers["files"].append(",".join(source_list))

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
