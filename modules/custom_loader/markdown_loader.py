from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
)


class MarkdownLoader(BaseLoader):
    def __init__(
        self,
        file_path: str | Path,
        encoding: str = "utf-8",
        headers_to_split_on: list[tuple[str, str]] | None = None,
    ):
        self.file_path = Path(file_path)
        self.encoding = encoding

        self.headers_to_split_on = (
            headers_to_split_on
            if headers_to_split_on is not None
            else [
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ]
        )

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )

    def load(self) -> list[Document]:
        with open(self.file_path, "r", encoding=self.encoding) as f:
            markdown = f.read()

        # 見出し単位に分割
        docs = self.header_splitter.split_text(markdown)

        for i, doc in enumerate(docs):
            doc.metadata.update(
                {
                    "source": str(self.file_path),
                    "file_name": self.file_path.name,
                    "extension": self.file_path.suffix,
                    "chunk": i,
                    "title": doc.metadata.get("h4")
                    or doc.metadata.get("h3")
                    or doc.metadata.get("h2")
                    or doc.metadata.get("h1"),
                    "header_path": " > ".join(
                        filter(
                            None,
                            [
                                doc.metadata.get("h1"),
                                doc.metadata.get("h2"),
                                doc.metadata.get("h3"),
                                doc.metadata.get("h4"),
                            ],
                        )
                    ),
                }
            )

        return docs


if __name__ == "__main__":
    # test
    path = "../../share/共有ドライブ/プロジェクト/医療法人社団 蒼樹会 みなみ野女性医療センター/04.分析/analysis_project/README.md"
    docs = MarkdownLoader(path).load()
    for doc in docs:
        print(doc.metadata)
        print(doc.page_content)
        print("--------------------------")
