import os
from pathlib import Path

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from modules.check_win32com import check_win32com

from .pdf_document_image_loader import PDFDocumentImageLoader

# win32comでPowerPointが使用可能かチェック
IS_WIN32COM = check_win32com(target="PowerPoint")
if IS_WIN32COM:
    import win32com.client


class PowerPointToPdfImageLoader(BaseLoader):
    """PowerPoint(.pptx)をPDF化->画像化してDocumentを保存するDocumentLoader"""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.temp_dir_path = Path(os.getenv("TEMP_DIR", "../../resource/temp"))

    def load(self) -> list[Document]:
        powerpoint = win32com.client.DispatchEx("PowerPoint.Application")

        pdf_docs = []
        try:
            presentation = powerpoint.Presentations.Open(
                self.file_path.resolve(), WithWindow=False, ReadOnly=True
            )

            pdf_path = self.temp_dir_path / f"{self.file_path.stem}.pdf"

            # PDF化してtempファイルで保存
            presentation.SaveAs(pdf_path.absolute(), 32)

            pdf_docs = PDFDocumentImageLoader(pdf_path).load()

            # メタデータを更新
            for pdf_doc in pdf_docs:
                pdf_doc.metadata["source"] = str(self.file_path)

            presentation.Close()

        finally:
            powerpoint.Quit()

        return pdf_docs


if __name__ == "__main__":
    path = r"..\..\share\共有ドライブ\プロジェクト\白峰信用リスク評価株式会社\06.報告書\白峰信用リスク評価株式会社_最終報告.pptx"
    loader = PowerPointToPdfImageLoader(path)
    documents = loader.load()

    for document in documents:
        print(document)
