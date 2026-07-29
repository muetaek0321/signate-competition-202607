from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# 環境変数の読み込み
load_dotenv()


SYSTEM_PROMPT = """
あなたは、ユーザーの質問および検索クエリに基づき、提示されたファイル一覧の中から回答に関連するファイルパスを正しく選択・抽出する専門のアシスタントです。
対象となる自社の会社名は「株式会社データアステル」（「データアステル」）です。

以下の手順とルールに従って、想定されるファイルパスのリストを作成してください。

### 指示・ルール
1. 入力された「質問」の内容と「検索クエリ」の意図を詳しく分析してください。
2. 質問文や検索クエリに「自社」「社内」「弊社」「当社」「我が社」などの言葉が含まれる場合、あるいは自社の会社名「データアステル」「株式会社データアステル」が含まれる場合は、社内管理フォルダや社内規定・決裁基準・社内用語集等の社内情報に関連するファイルパスを優先的かつ適切に考慮して抽出してください。
3. 提示されたファイルパス一覧（FilePathNo. ...）の中から、質問の回答に直接関係する、または必要な情報が含まれていると推測されるファイルパスのみを抽出してください。
4. 必ず提示されたファイルパス一覧の中に存在するパスのみを選択し、存在しないパスや推測による創作パスは含めないでください。
5. 該当するファイルが複数ある場合はそれらを全てリストに含めてください。該当するファイルがない場合は空のリストを返してください。
"""


class Response(BaseModel):
    path_list: list[str] = Field(
        description="質問文と検索クエリから想定されるファイルのパスを格納したリスト"
    )


class FilePathFilter:
    """LLMを使って検索したファイル情報から質問文に合うファイルパスを抽出する"""

    def __init__(
        self,
        model: str = "models/gemma-4-31b-it",
    ) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
        )
        self.llm = self.llm.with_structured_output(Response)

    def __call__(self, input_question: str, query: str, file_docs: list[Document]) -> list[str]:
        # コンテキストを作成
        contexts = []
        for i, doc in enumerate(file_docs, start=1):
            # テキストの追加
            contexts.append(
                {"type": "text", "text": f'FilePathNo.{i} = "{doc.metadata["source"]}"'}
            )
        # 最後に質問文と検索クエリを追加
        contexts.append({"type": "text", "text": f"## 質問\n{input_question}"})
        contexts.append({"type": "text", "text": f"## 検索クエリ\n{query}"})

        input_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=contexts),
        ]

        # 返答の生成
        response: Response = self.llm.invoke(input_messages)

        # パスの存在チェック
        sources = [p for p in response.path_list if Path(p).exists()]

        return sources


if __name__ == "__main__":
    question = ""
