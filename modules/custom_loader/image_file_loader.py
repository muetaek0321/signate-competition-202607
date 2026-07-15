import base64
import os
from io import BytesIO
from pathlib import Path

import joblib
from dotenv import load_dotenv
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image

# 環境変数の読み込み
load_dotenv()

PROMPT = """
ファイル検索用のテキスト情報を収集します。指定した画像について、以下の形式に従って詳細に説明してください。
出力は番号付き項目として、項目が存在しない場合は「なし」または「該当なし」と記載してください。

1. 画像全体の概要:
2. 写っている物体:
3. 人物:
4. グラフ・図:
5. 表:
6. OCR結果:
7. 色やハイライト:
8. レイアウト:
9. 検索に有効そうなキーワード: ["キーワード1", "キーワード2", ...]

- 各項目は短く簡潔にまとめる
- 全体の文字数は1500文字を超えないようにする
- 不確かな場合は「不明」または「判別できない」と記載
"""


class ImageDocumentLoader(BaseLoader):
    """
    VisionModelを使用して画像の内容をDocument化するLoader
    """

    def __init__(
        self,
        file_path: str | Path,
        model: str = "models/gemma-4-31b-it",
    ):
        self.file_path = Path(file_path)
        self.prompt = PROMPT

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
        )

        self.image_store_path = (
            Path(os.getenv("DATASET_DIR", "./resource/chroma")) / "image_store.joblib"
        )
        if self.image_store_path.exists():
            self.image_store = joblib.load(self.image_store_path)
        else:
            self.image_store = {}

    def load(self) -> list[Document]:
        image = Image.open(self.file_path)

        response_date, image_store_id = self._describe_image(image)
        description = response_date[1]["text"]

        metadata = {
            "source": str(self.file_path),
            "file_name": self.file_path.name,
            "extension": self.file_path.suffix.lower(),
            "width": image.width,
            "height": image.height,
            "image_store_id": image_store_id,
        }

        return [
            Document(
                page_content=description,
                metadata=metadata,
            )
        ]

    def _describe_image(self, image: Image.Image | str) -> tuple[str, str]:
        if isinstance(image, Image.Image):
            image_base64 = self._image_to_bytes(image)
        else:
            image_base64 = image

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": self.prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ]
        )

        response = self.llm.invoke([message])

        image_store_id = f"{len(self.image_store.keys()):05}"
        self.image_store[image_store_id] = image_base64
        joblib.dump(self.image_store, self.image_store_path)

        return response.content, image_store_id

    @staticmethod
    def _image_to_bytes(image: Image.Image) -> str:
        buffer = BytesIO()

        if image.mode != "RGB":
            image = image.convert("RGB")

        image.save(buffer, format="PNG")

        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        return image_base64
