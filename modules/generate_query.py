from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# 環境変数の読み込み
load_dotenv()

PROMPT_TEMPLATE = """
ユーザーは社内情報ファイルサーバに対する情報検索のための質問文を入力します。
ユーザーの質問内容と社内用語集をもとにベクトルデータベースやハイブリッド検索に最適な検索クエリ（キーワード群）を生成してください。
以下のルールに従って、質問文から検索に必要な重要キーワードを抽出し、スペース区切りで出力してください。

【ルール】
1. 質問の意図を汲み取り、対象となる文書（ファイルサーバ内の資料など）に直接含まれていそうな単語を予測して抽出すること。
2. 不要な助詞、接続詞、敬語、および検索ノイズになる一般的な単語（「教えて」「何ですか」「ついて」など）は完全に除外すること。
3. 固有名詞（プロジェクト名、企業名、製品名、ファイル名、拡張子など）と専門用語・技術用語は最優先で抽出・保持すること。
4. 固有名詞は必要に応じて社内用語集の別名もクエリに含めるようにすること。
5. 検索ヒット率を高めるため、重要な単語には一般的な同義語、英語表記、略称などを適宜補完すること。
6. 出力は単語（名詞や短い複合語）のみとし、文章にしないこと。
7. 単語と単語の間は半角スペースで区切ること。
8. 理由の説明、挨拶、または「キーワード:」などの前置き文は一切出力せず、キーワード群のみを出力すること。
9. キーワードは最大20語程度に収めること。

【質問文】
{question}

【社内用語集】
{glossary}

"""


class QueryGenerator:
    """Ollamaを使用した検索クエリ生成クラス"""

    def __init__(self) -> None:
        """初期化"""
        # モデルの準備
        self.llm = ChatGoogleGenerativeAI(model="models/gemma-4-31b-it", temperature=0.0)

        # 社内用語集を読み込み
        glossary_path = "./resource/社内用語集.docx"
        word_doc = UnstructuredWordDocumentLoader(glossary_path).load()
        self.glossary = word_doc[0].page_content

    def __call__(self, input_question: str) -> str:
        """質問文からベクトル検索用のクエリを作成

        Args:
            input_question (str): 質問文

        Returns:
            str: 検索クエリ
        """
        input_messages = [
            HumanMessage(
                content=PROMPT_TEMPLATE.format(question=input_question, glossary=self.glossary)
            ),
        ]

        # 返答の生成
        response = self.llm.invoke(input_messages)

        # 生成された返答の取得
        response_date = response.content
        query = response_date[1]["text"]

        return query


if __name__ == "__main__":
    # test
    query_gen = QueryGenerator()

    input_question = "白峰信用リスク評価の提案書old.pptxから提案書.pptxへの更新内容のうち、案件遂行に関連する実質的な変更を挙げてください。"
    res = query_gen(input_question)
    print(res)

    input_question = "TOTOのFR書にて記載のある選択特徴量のうち、ENG-FTはいくつありますか。"
    res = query_gen(input_question)
    print(res)
