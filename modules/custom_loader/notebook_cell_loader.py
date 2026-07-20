import base64
import re
from io import BytesIO
from pathlib import Path

import nbformat
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from PIL import Image

from .image_file_loader import ImageDocumentLoader


class NotebookCellLoader(BaseLoader):
    """
    Jupyter Notebook(.ipynb)用カスタムLoader

    1セル = 1Document

    取得情報
    --------
    ・markdownセルのテキスト（画像データを除く）
    ・codeセルのソースコード
    ・codeセルのテキスト出力結果
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def load(self) -> list[Document]:
        # ipynbファイルを読み込み
        with open(self.file_path, mode="r", encoding="utf-8-sig") as f:
            nb = nbformat.read(f, as_version=4)

        # セルごとに処理
        docs = []
        for cell in nb["cells"]:
            cell_type = cell["cell_type"]
            if cell_type == "markdown":
                source = cell["source"]
                extracted_images = []
                
                # Attachmentsから画像の抽出
                if "attachments" in cell and cell["attachments"]:
                    for attachment_name, attachment_data in cell["attachments"].items():
                        for mime_type, b64_data in attachment_data.items():
                            if mime_type.startswith("image/"):
                                img_bytes = base64.b64decode(b64_data)
                                img = Image.open(BytesIO(img_bytes))
                                extracted_images.append(img)
                                
                # HTMLのBase64直接埋め込み画像の抽出（簡易的）
                if "data:image/" in source:
                    # 正規表現で data:image/xxx;base64, 以降のBase64文字列を抽出
                    pattern = r"data:image/[^;]+;base64,([^\"\'\)\s>]+)"
                    matches = re.finditer(pattern, source)
                    for match in matches:
                        try:
                            b64_data = match.group(1)
                            img_bytes = base64.b64decode(b64_data)
                            img = Image.open(BytesIO(img_bytes))
                            extracted_images.append(img)
                            # sourceからBase64画像部分を除去
                            source = source.replace(match.group(0), "")
                        except Exception:
                            # 画像として開けなかった場合などはスキップ
                            pass

                metadata = {"source": str(self.file_path), "cell_type": "markdown"}
                if extracted_images:
                    image_store_ids = []
                    for img in extracted_images:
                        response_date, image_store_id = ImageDocumentLoader(self.file_path)._describe_image(img)
                        source += f"\n画像の情報：\n{response_date[1]['text']}"
                        image_store_ids.append(image_store_id)
                    metadata["image_store_ids"] = image_store_ids

                docs.append(
                    Document(
                        page_content=f"\n```markdown\n{source}\n```",
                        metadata=metadata,
                    )
                )
            elif cell_type == "code":
                source = cell["source"]
                code_content = f"\n```python\n{source}\n```"
                
                # コード部分を格納
                metadata = {"source": str(self.file_path), "cell_type": "code"}
                docs.append(
                    Document(
                        page_content=code_content,
                        metadata=metadata,
                    )
                )
                
                extracted_images = []
                if len(cell["outputs"]) > 0:
                    output_content = ""
                    for output in cell["outputs"]:
                        if "text" in output:
                            output_content += f"\n{output['text']}"
                        
                        if "data" in output:
                            for mime_type, data_content in output["data"].items():
                                if mime_type.startswith("image/"):
                                    img_bytes = base64.b64decode(data_content)
                                    img = Image.open(BytesIO(img_bytes))
                                    extracted_images.append(img)

                
                    if extracted_images:
                        image_store_ids = []
                        for img in extracted_images:
                            response_date, image_store_id = ImageDocumentLoader(self.file_path)._describe_image(img)
                            output_content += f"\n画像の情報：\n{response_date[1]['text']}"
                            image_store_ids.append(image_store_id)
                        metadata["image_store_ids"] = image_store_ids

                    # 出力部分を格納
                    metadata = {"source": str(self.file_path), "cell_type": "output"}
                    docs.append(
                        Document(
                            page_content=output_content,
                            metadata=metadata,
                        )
                    )

        return docs
    
    
if __name__ == "__main__":
    path = r"..\..\share\共有ドライブ\プロジェクト\医療法人社団 恒一会 かえで総合病院\04.分析\analysis_project\notebooks\01_eda.ipynb"
    loader = NotebookCellLoader(path)
    docs = loader.load()
    
    for doc in docs:
        print(doc.metadata)
        print(doc.page_content)
        print("-----------------")
        
