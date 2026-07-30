from pathlib import Path

import pandas as pd
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class CsvChunkLoader(BaseLoader):
    """
    CSVをチャンク単位で分割するDocumentLoader

    Parameters
    ----------
    file_path : str | Path
        CSVファイルパス

    chunk_rows : int
        1 Documentあたりの行数

    delimiter : str
        区切り文字
    """

    def __init__(
        self,
        file_path: str | Path,
        chunk_rows: int = 300,
        delimiter: str = ",",
    ) -> None:
        self.file_path = Path(file_path)
        self.chunk_rows = chunk_rows
        self.delimiter = delimiter

    def load(self) -> list[Document]:
        # チャンクでcsvを読み込み
        df_chunk = pd.read_csv(
            self.file_path, encoding="utf-8-sig", chunksize=self.chunk_rows, sep=self.delimiter
        )

        # チャンクごとにDocument形式に格納
        docs: list[Document] = []
        for chunk in df_chunk:
            start_row = chunk.index[0] + 1
            end_row = chunk.index[-1] + 1
            add_info = f"# （{start_row}行目～{end_row}行目）\n\n"
            row_content = ",\n".join(
                "  " + row.to_json(force_ascii=False) for _, row in chunk.iterrows()
            )
            page_content = add_info + "```json\n[\n" + row_content + "\n]\n```"

            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source": str(self.file_path),
                        "start_row": start_row,
                        "end_row": end_row,
                    },
                )
            )

        return docs


if __name__ == "__main__":
    path = r"..\..\share\共有ドライブ\プロジェクト\医療法人社団 恒一会 かえで総合病院\03.データ\train.csv"
    loader = CsvChunkLoader(path, chunk_rows=5)
    docs = loader.load()
    print(docs[0].metadata)
    print(docs[0].page_content)

    path = (
        r"..\..\share\共有ドライブ\プロジェクト\株式会社青潮モビリティサービス\03.データ\train.tsv"
    )
    loader = CsvChunkLoader(path, chunk_rows=5, delimiter="\t")
    docs = loader.load()
    print(docs[0].metadata)
    print(docs[0].page_content)
