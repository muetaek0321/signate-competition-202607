import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "./resource/pretrained"  # 事前学習モデルの保存先指定

import joblib
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from modules.bm25_search import BM25DocumentSearch
from modules.embedding_models import get_embedding
from modules.filepath_filter import FilePathFilter
from modules.generate_query import QueryGenerator

RAG_PROMPT_TEMPLATE = """
あなたは正確で信頼できるドキュメントQAアシスタントです。
与えられたコンテキスト（共有フォルダ内のドキュメント情報や画像データなど）のみを根拠として、ユーザの質問に日本語で回答してください。

# 🚨 【厳守】絶対的な制約事項（以下のルールに違反した出力はシステムエラーとなります）🚨

1. 【コンテキスト依存の絶対ルール】
   - 回答は提供されたコンテキストの情報のみに基づいてください。
   - あなた自身の事前知識、一般的な事実、推論で情報を補うことは絶対に禁止します。
   - コンテキストに記載がないことは絶対に回答に含めないでください。
   - 前置きなどは不要でシンプルに求められている情報のみを回答として提示してください。

2. 【情報不足時の絶対ルール】
   - コンテキストから確実な回答が導き出せない場合、または情報が部分的にしか存在せず完全に答えられない場合は、決して推測してはいけません。
   - そのような場合、`answer` の値は必ず「わかりません」という文字列にしてください。

3. 【出力フォーマットの絶対ルール】
   - 出力は必ず下記のJSONフォーマットのみとしてください。
   - Markdown記法（```json や ``` など）、挨拶、前置き、後書きなどの余分なテキストは一切出力しないでください。最初から最後までJSONのみを出力してください。

# 出力JSONフォーマット
{
  "reason": "回答を導き出すための思考プロセスや、コンテキスト中の根拠の有無とその評価",
  "answer": "質問に対するシンプルかつ直接的な回答（情報不足の場合は「わかりません」と回答）"
}
"""

AI_RESPONSE_FORMAT = """
回答: {answer}
根拠: {reason}
"""


class Response(BaseModel):
    reason: str = Field(description="回答を導き出すための思考プロセスやコンテキスト中の根拠")
    answer: str = Field(
        description="質問に対するシンプルかつ直接的な回答（情報不足の場合は「わかりません」と回答）"
    )


class ResponseGenerator:
    def __init__(
        self, persist_directory: str | Path, num_top_docs: int = 20, lambda_mult: float = 0.3
    ) -> None:
        """初期化

        Args:
            persist_directory(str | Path): ベクトルDBのパス
            num_top_docs(int): 採用するドキュメント数
            lambda_mult(float): MMR検索のlambda_multの値
        """

        self.num_top_docs = num_top_docs
        self.lambda_mult = lambda_mult
        self.input_messages = []

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
        elif self.generate_mode == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL_NAME", "models/gemini-3.5-flash-lite"),
                temperature=0.0,
                thinking_budget=4096,
            )
            self.llm = self.llm.with_structured_output(Response)

        # Embeddingモデルの読み込み
        self.embedding = get_embedding()

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
        self.bm25 = BM25DocumentSearch(dir_path=persist_directory, k=50)
        # Rerankerモデルの読み込み
        self.reranker = CrossEncoder(
            os.getenv("RERANKER_MODEL_NAME", None),
            device="cuda",
        )

        # 検索クエリ作成モデルの読み込み
        self.query_gen = QueryGenerator()

        # ファイルパス抽出モデルの読み込み
        self.path_filter = FilePathFilter()

        # 画像データのキャッシュの読み込み
        self.image_store = joblib.load(persist_directory / "image_store.joblib")

    def search_context(self, input_question: str) -> list[str]:
        # 検索クエリを作成
        query = self.query_gen(input_question)

        # ファイル情報のベクトルDBから検索対象ファイルを取得
        file_info_docs = self.vectorstore_file_info.similarity_search(query=query, k=20)
        # 検索したファイル情報をリランキング
        question_file_info_list = [
            (input_question, f"{doc.page_content}") for doc in file_info_docs
        ]
        scores = self.reranker.predict(question_file_info_list)
        reranked_file_info_docs = sorted(
            zip(file_info_docs, scores), key=lambda x: x[1], reverse=True
        )
        file_info_docs = [doc for doc, score in reranked_file_info_docs]

        # LLMによるファイルパスの絞り込みを適用
        sources = self.path_filter(input_question, query, file_info_docs)
        if len(sources) == 0:
            sources = [doc.metadata["source"] for doc in file_info_docs][:5]

        # 検索結果からファイルパスを拡張子ごとに取得
        files, csv_files, excel_files = [], [], []
        for source in sources:
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
                k=50,
                fetch_k=150,
                lambda_mult=self.lambda_mult,
                filter={"source": {"$in": files}},
            )
        if len(csv_files) > 0:
            docs += self.vectorstore_csv.max_marginal_relevance_search(
                query=query,
                k=30,
                fetch_k=100,
                lambda_mult=self.lambda_mult,
                filter={"source": {"$in": csv_files}},
            )
        if len(excel_files) > 0:
            docs += self.vectorstore_excel.max_marginal_relevance_search(
                query=query,
                k=30,
                fetch_k=100,
                lambda_mult=self.lambda_mult,
                filter={"source": {"$in": excel_files}},
            )

        if len(docs) == 0:
            docs = self.vectorstore_all.max_marginal_relevance_search(
                query=query,
                k=100,
                fetch_k=200,
                lambda_mult=self.lambda_mult,
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

    def genrate_answer(
        self, input_question: str, retry: int = 0
    ) -> tuple[dict[str, str], list[str]]:
        # 初回実行時にシステムプロンプトのみに初期化
        if retry == 0:
            self.input_messages = [SystemMessage(content=RAG_PROMPT_TEMPLATE)]

        # リトライ回数に応じてドキュメントを取得
        top_docs = [doc for doc, _ in self.reranked_docs][
            self.num_top_docs * retry : self.num_top_docs * (retry + 1)
        ]

        # RAGプロンプトに埋め込むために成形
        contexts = []
        for i, doc in enumerate(top_docs):
            # テキストの追加
            context_num = self.num_top_docs * retry + i + 1
            contexts.append(
                {"type": "text", "text": f"ContextNo.{context_num}:\n{doc.page_content}"}
            )
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
        # 最後に質問文を追加
        contexts.append({"type": "text", "text": f"## 質問\n{input_question}"})

        # 入力内容を作成
        self.input_messages.append(HumanMessage(content=contexts))

        # 返答の生成
        response = self.llm.invoke(self.input_messages)
        response_dict = response.model_dump()

        # リトライ時のために今回のAIの返答を履歴に追加（Gemini等のAPIエラー回避）
        self.input_messages.append(
            AIMessage(
                content=AI_RESPONSE_FORMAT.format(answer=response.answer, reason=response.reason)
            )
        )

        doc_files = list(
            {
                doc.metadata["source"]
                for doc, _ in self.reranked_docs[: self.num_top_docs * (retry + 1)]
            }
        )

        return response_dict, doc_files
