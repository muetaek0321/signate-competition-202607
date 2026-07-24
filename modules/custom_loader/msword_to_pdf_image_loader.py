import os
from pathlib import Path

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from modules.check_win32com import check_win32com

from .pdf_document_image_loader import PDFDocumentImageLoader

# win32comでWordが使用可能かチェック
IS_WIN32COM = check_win32com(target="Word")
if IS_WIN32COM:
    import win32com.client


class WordToPdfImageLoader(BaseLoader):
    """Word(.docx)をPDF化->画像化してDocumentを保存するDocumentLoader"""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.temp_dir_path = Path(os.getenv("TEMP_DIR", "../../resource/temp"))

    def load(self) -> list[Document]:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0

        pdf_docs = []
        try:
            doc = word.Documents.Open(str(self.file_path.resolve()))

            pdf_path = self.temp_dir_path / f"{self.file_path.stem}.pdf"

            # PDF化してtempファイルで保存
            doc.SaveAs(
                str(pdf_path.absolute()),
                FileFormat=17,  # wdFormatPDF
            )

            pdf_docs = PDFDocumentImageLoader(pdf_path).load()

            # メタデータを更新
            for pdf_doc in pdf_docs:
                pdf_doc.metadata["source"] = str(self.file_path)

            doc.Close()

        finally:
            word.Quit()

        return pdf_docs


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    path = r"..\..\share\共有ドライブ\プロジェクト\白峰信用リスク評価株式会社\01.契約\契約書.docx"
    loader = WordToPdfImageLoader(path)
    documents = loader.load()
    print(f"実行時間: {time.perf_counter() - start:.3f}s")

    for document in documents:
        print(document)
