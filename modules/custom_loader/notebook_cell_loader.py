from pathlib import Path

import nbformat
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document


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

        # TODO: base64の画像を読み込めるか検討する

        # セルごとに処理
        docs = []
        for cell in nb["cells"]:
            cell_type = cell["cell_type"]
            if cell_type == "markdown":
                source = cell["source"]
                if "data:image/png" in source:
                    continue  # 画像データは一旦スキップ
                docs.append(
                    Document(
                        page_content=f"\n```markdown\n{source}\n```",
                        metadata={"source": str(self.file_path)},
                    )
                )
            elif cell_type == "code":
                source = cell["source"]
                code_content = f"\n```python\n{source}\n```"
                if (len(cell["outputs"]) > 0) and ("text" in cell["outputs"][0].keys()):
                    outputs = cell["outputs"][0]["text"]  # 一旦テキストのみ
                    code_content += f"\n出力結果：\n{outputs}"
                docs.append(
                    Document(
                        page_content=code_content,
                        metadata={"source": str(self.file_path)},
                    )
                )

        # with open("./check_result/ipynb_conv.md", mode="w", encoding="utf-8-sig") as f:
        #     f.write(doc.page_content)

        return docs
