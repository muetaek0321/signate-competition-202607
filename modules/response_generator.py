import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "./resource/pretrained"  # 事前学習モデルの保存先指定

import joblib
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from modules.bm25_search import BM25DocumentSearch
from modules.generate_query import QueryGenerator
from modules.rag_prompt import RAG_PROMPT_TEMPLATE


class Response(BaseModel):
    reason: str = Field(description="回答を導き出すための思考プロセスや、コンテキスト中の根拠")
    answer: str = Field(description="質問に対する直接的な回答")
    search_queries: list[str] = Field(description="再検索用の検索クエリのリスト")


class ResponseGenerator:
    def __init__(self, persist_directory):
        self.num_top_docs = 15

        # モデルのセットアップ
        ## Ollama-Cloudを使用
        self.generate_mode = os.getenv("GENERATE_MODE", "ollama")
        if self.generate_mode == "ollama":
            self.llm = ChatOllama(
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
        elif self.generate_mode == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL_NAME", "models/gemini-3.1-flash-lite"),
                temperature=0.0,
                thinking_budget=4096,
            )
            self.llm = self.llm.with_structured_output(Response)

        # Embeddingモデルの読み込み
        embedding_mode = os.getenv("EMBEDDING_MODE", "huggingface")
        if embedding_mode == "huggingface":
            self.embedding = HuggingFaceEmbeddings(
                model_name=os.getenv("EMBEDDING_MODEL_NAME", None),
                model_kwargs={"device": "cuda", "trust_remote_code": True},
            )
        elif embedding_mode == "ollama":
            self.embedding = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", None))
        else:
            raise ValueError(f"Invalid EMBEDDING_MODE: {embedding_mode}")

        # ベクトルDBの読み込み
        self.vectorstore_all = Chroma(
            embedding_function=self.embedding,
            persist_directory=persist_directory,
            collection_name="shared_folder_all_documents",
        )
        self.vectorstore_csv = Chroma(
            embedding_function=self.embedding,
            persist_directory=persist_directory,
            collection_name="shared_folder_csv_documents",
        )
        self.vectorstore_excel = Chroma(
            embedding_function=self.embedding,
            persist_directory=persist_directory,
            collection_name="shared_folder_excel_documents",
        )
        self.vectorstore_file_info = Chroma(
            embedding_function=self.embedding,
            persist_directory=persist_directory,
            collection_name="shared_folder_file_info_list",
        )
        # BM25Retrieverの読み込み
        self.bm25 = BM25DocumentSearch(dir_path=persist_directory, k=30)
        # Rerankerモデルの読み込み
        self.reranker = CrossEncoder(
            os.getenv("RERANKER_MODEL_NAME", None),
            device="cuda",
        )

        # 検索クエリ作成モデルの読み込み
        self.query_gen = QueryGenerator()

        # 画像データのキャッシュの読み込み
        self.image_store = joblib.load(persist_directory / "image_store.joblib")

    def search_context(self, input_question: str) -> list[str]:
        # 検索クエリを作成
        query = self.query_gen(input_question)

        # ファイル情報のベクトルDBから検索対象ファイルを取得
        file_info_docs = self.vectorstore_file_info.similarity_search(query=query, k=10)
        # 検索結果からファイルパスを拡張子ごとに取得
        files, csv_files, excel_files = [], [], []
        for doc in file_info_docs:
            source = doc.metadata["source"]
            ext = Path(source).suffix
            if ext == ".csv":
                csv_files.append(source)
            elif ext == ".xlsx":
                excel_files.append(source)
            files.extend(
                [str(p) for p in Path(source).parent.iterdir() if p.suffix not in [".csv", ".xlsx"]]
            )

        # ベクトルDBから検索
        docs = []
        if len(files) > 0:
            files = list(set(files))
            docs += self.vectorstore_all.max_marginal_relevance_search(
                query=query,
                k=30,
                fetch_k=100,
                lambda_mult=0.5,
                filter={"source": {"$in": files}},
            )
        if len(csv_files) > 0:
            docs += self.vectorstore_csv.max_marginal_relevance_search(
                query=query,
                k=30,
                fetch_k=100,
                lambda_mult=0.5,
                filter={"source": {"$in": csv_files}},
            )
        if len(excel_files) > 0:
            docs += self.vectorstore_excel.max_marginal_relevance_search(
                query=query,
                k=30,
                fetch_k=100,
                lambda_mult=0.5,
                filter={"source": {"$in": excel_files}},
            )

        if len(docs) == 0:
            docs = self.vectorstore_all.max_marginal_relevance_search(
                query=query,
                k=100,
                fetch_k=200,
                lambda_mult=0.5,
            )
        # BM25Retrieverから検索
        docs += self.bm25(query)

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
        question_answer_list = [(input_question, f"{doc.page_content}") for doc in docs]
        scores = self.reranker.predict(question_answer_list)
        self.reranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        search_files = files + csv_files + excel_files
        return search_files

    def genrate_answer(self, input_question: str) -> tuple[dict[str, str], list[str]]:
        top_docs = [doc for doc, _ in self.reranked_docs][: self.num_top_docs]

        # RAGプロンプトに埋め込むために成形
        contexts = []
        for i, doc in enumerate(top_docs):
            # テキストの追加
            contexts.append({"type": "text", "text": f"ContextNo.{i + 1}:\n{doc.page_content}"})
            # 画像を含む場合は画像も追加
            if doc.metadata.get("image_store_id"):
                image_store_ids = [doc.metadata["image_store_id"]]
            elif doc.metadata.get("image_store_ids"):
                image_store_ids = doc.metadata["image_store_ids"]
            else:
                image_store_ids = []
            for image_store_id in image_store_ids:
                contexts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.image_store[image_store_id]}"
                        },
                    }
                )

        # 入力内容を作成
        input_messages = [
            SystemMessage(content=RAG_PROMPT_TEMPLATE),
            HumanMessage(
                content=contexts
                + [
                    {
                        "type": "text",
                        "text": f"## 質問\n{input_question}",
                    }
                ]
            ),
        ]

        # 返答の生成
        response = self.llm.invoke(input_messages)

        # if self.generate_mode == "gemini":
        #     response_text = response.content[0]["text"]
        # else:
        #     response_text = response.content

        # # 返答の変換
        # try:
        #     response_dict = json.loads(response_text)
        # except Exception as e:
        #     response_dict = {
        #         "answer": response_text.replace("\n", ""),
        #         "reason": f"jsonのパース失敗({e})",
        #     }
        response_dict = response.model_dump()

        doc_files = list(set([doc.metadata["source"] for doc in top_docs]))

        return response_dict, doc_files
