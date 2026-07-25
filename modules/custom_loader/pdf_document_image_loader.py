from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import fitz
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.document_loaders import BaseLoader
from PIL import Image

from .image_file_loader import ImageDocumentLoader

MAX_WORKERS = 10


class PDFDocumentImageLoader(BaseLoader):
    """PDFの1ページごとのテキストデータと画像化データ格納するDocumentLoader"""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.img_loader = ImageDocumentLoader(self.file_path)

    def _process_page_image(self, image: Image.Image):
        """1ページ分の画像説明を生成する（並列実行用）"""
        response_date, image_store_id = self.img_loader._describe_image(image)
        description = response_date[1]["text"]
        return description, image_store_id

    def load(self):
        # テキスト部分の読み込み
        docs = PyPDFLoader(self.file_path).load()

        # PDF全体のデータ読み込み - 各ページを画像化
        pdf_datas = fitz.open(self.file_path)
        images = []
        for page_data in pdf_datas:
            pix = page_data.get_pixmap()
            img_data = pix.tobytes("png")
            images.append(Image.open(BytesIO(img_data)))

        # ThreadPoolExecutorで画像説明を並列生成
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(self._process_page_image, images))

        # 結果をdocsに反映
        for page_doc, (description, image_store_id) in zip(docs, results):
            page_doc.page_content += "\n" + description
            page_doc.metadata["image_store_id"] = image_store_id

        return docs


if __name__ == "__main__":
    path = r"..\..\share\共有ドライブ\プロジェクト\株式会社青潮モビリティサービス\06.報告書\株式会社青潮モビリティサービス_最終報告.pdf"
    loader = PDFDocumentImageLoader(path)
    documents = loader.load()

    for document in documents:
        print(document)
