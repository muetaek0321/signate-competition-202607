from io import BytesIO
from pathlib import Path

import fitz
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.document_loaders import BaseLoader
from PIL import Image

from .image_file_loader import ImageDocumentLoader


class PDFDocumentImageLoader(BaseLoader):
    """PDFの1ページごとのテキストデータと画像化データ格納するDocumentLoader"""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def load(self):
        # テキスト部分の読み込み
        docs = PyPDFLoader(self.file_path).load()

        # PDF全体のデータ読み込み
        pdf_datas = fitz.open(self.file_path)
        for page_doc, page_data in zip(docs, pdf_datas):
            # PDFページを画像化
            pix = page_data.get_pixmap()
            img_data = pix.tobytes("png")
            image = Image.open(BytesIO(img_data))

            # 画像化したPDFの説明文を作成
            response_date, image_store_id = ImageDocumentLoader(self.file_path)._describe_image(
                image
            )
            description = response_date[1]["text"]

            # 説明文を追記
            page_doc.page_content += "\n" + description
            # メタデータに画像情報を追加
            page_doc.metadata["image_store_id"] = image_store_id

        return docs


if __name__ == "__main__":
    path = r"..\..\share\共有ドライブ\プロジェクト\株式会社青潮モビリティサービス\06.報告書\株式会社青潮モビリティサービス_最終報告.pdf"
    loader = PDFDocumentImageLoader(path)
    documents = loader.load()

    for document in documents:
        print(document)
