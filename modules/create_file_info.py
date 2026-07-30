from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# 環境変数の読み込み
load_dotenv()

# ファイル情報をLLMで成形するためのプロンプトテンプレート
FILE_INFO_PROMPT = """
あなたは社内ファイル管理システムのアシスタントです。
以下のファイルパスと社内用語集をもとに、検索しやすい形式のファイル情報サマリーを日本語で生成してください。

## 入力情報
- ファイルパス: {filepath}
- 社内用語集: {glossary}

## 制約事項
- **事実に基づくデータ作成**: ファイルパスの構造（フォルダ名・ファイル名・拡張子など）から読み取れる事実および社内用語集に記載された明確な情報のみをもとにデータを作成してください。根拠のない推測や架空の情報は含めないでください。
- **社内用語集の活用（読み替え・関連情報）**: ファイルパスに含まれる用語や名称が社内用語集に掲載されている場合、用語集を参照して読み替え（正式名称、略称、同義語など）や関連情報を各項目（会社名、カテゴリ、検索キーワード等）に含めて補完してください。
- **出力形式の厳格な遵守**: 指定された「出力形式」のフォーマットを必ず守ってください。
- **余計な文章の排除**: 挨拶、説明、前置き、後置き、解説、コードブロック表記（```）などの余計な文章は一切含めず、出力形式の形のデータのみで返してください。

## 出力形式
ファイルパス: （入力のファイルパスをそのまま記載）
ファイル名: （パスから読み取ったファイル名）
ファイル拡張子: （パスから読み取った拡張子）
ディレクトリ種別: （フォルダ構成から読み取れるディレクトリの種別・用途。不明な場合は「不明」）
会社名: （フォルダ名やファイル名から明確に読み取れる関連会社・顧客・組織名。社内用語集に掲載がある場合は読み替え・正式名称も反映。不明な場合は「不明」）
カテゴリ: （ファイル名やパスから明確に読み取れるカテゴリ。社内用語集に掲載がある場合は分類や関連情報を反映。不明な場合は「不明」）
検索キーワード: （ファイルパスから読み取れるキーワードに加え、社内用語集に基づく読み替え（正式名称・略称・同義語）や関連情報をカンマ区切りで列挙）
"""


class FileInfoCreator:
    """LLMを使ってファイル情報をまとめる"""

    def __init__(
        self,
        model: str = "models/gemma-4-31b-it",
    ) -> None:
        # モデルの準備
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
        )

        # 社内用語集を読み込み
        glossary_path = "./resource/社内用語集.docx"
        word_doc = UnstructuredWordDocumentLoader(glossary_path).load()
        self.glossary = word_doc[0].page_content

    def __call__(self, filepath: Path) -> str:
        input_messages = [
            HumanMessage(
                content=FILE_INFO_PROMPT.format(filepath=str(filepath), glossary=self.glossary)
            ),
        ]

        # 返答の生成
        response = self.llm.invoke(input_messages)

        # 生成された返答の取得
        response_date = response.content
        file_info = response_date[1]["text"]

        return file_info


if __name__ == "__main__":
    # test
    file_info_creator = FileInfoCreator()

    filepath = Path(
        r"share\共有ドライブ\プロジェクト\医療法人社団 恒一会 かえで総合病院\06.報告書\医療法人社団 恒一会 かえで総合病院_最終報告_old.pptx"
    )
    file_info = file_info_creator(filepath)
    print(file_info)

    filepath = Path(r"share\共有ドライブ\社内管理\データアステル社内管理_決裁基準.md")
    file_info = file_info_creator(filepath)
    print(file_info)
