from pathlib import Path

import nbformat
import pandas as pd
from langchain_core.documents import Document


def csv_loader(path: Path) -> list[Document]:
    # チャンクでcsvを読み込み
    df_chunk = pd.read_csv(path, encoding="utf-8-sig", chunksize=50)

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
                    "source": str(path),
                    "start_row": start_row,
                    "end_row": end_row,
                },
            )
        )

    # with open("./check_result/csv_conv.md", mode="w", encoding="utf-8-sig") as f:
    #     f.write(docs[-1].page_content)

    return docs


def excel_loader(path: Path) -> list[Document]:
    # チャンクでExcelファイルを読み込み
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print("読み込みに失敗しました:", path.name)
        print("Error:", e)
        return []

    # チャンクごとにDocument形式に格納
    chunksize = 50
    docs: list[Document] = []
    for start_row in range(0, len(df), chunksize):
        end_row = start_row + chunksize
        chunk = df.iloc[start_row:end_row]
        add_info = f"# （{start_row}行目～{end_row}行目）\n\n"
        row_content = []
        for _, row in chunk.iterrows():
            try:
                row_content.append("  " + row.to_json(force_ascii=False))
            except Exception as e:
                print(row.to_dict())
                print("Error:", e)

        row_content_str = ",\n".join(row_content)
        page_content = add_info + "```json\n[\n" + row_content_str + "\n]\n```"

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": str(path),
                    "start_row": start_row,
                    "end_row": end_row,
                },
            )
        )

    # with open("./check_result/excel_conv.md", mode="w", encoding="utf-8-sig") as f:
    #     f.write(docs[-1].page_content)

    return docs


def notebook_loader(path: Path) -> Document:
    # ipynbファイルを読み込み
    with open(path, mode="r", encoding="utf-8-sig") as f:
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
                    metadata={"source": str(path)},
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
                    metadata={"source": str(path)},
                )
            )

    # with open("./check_result/ipynb_conv.md", mode="w", encoding="utf-8-sig") as f:
    #     f.write(doc.page_content)

    return docs


if __name__ == "__main__":
    # 単体テスト

    csv_loader(
        path=Path(
            "./share/共有ドライブ/プロジェクト/株式会社青嶺不動産アセットマネジメント/03.データ/train.csv"
        )
    )

    excel_loader(
        path=Path(
            "./share/共有ドライブ/プロジェクト/株式会社青嶺不動産アセットマネジメント/03.データ/train.xlsx"
        )
    )

    notebook_loader(
        path=Path(
            "./share/共有ドライブ/プロジェクト/白峰信用リスク評価株式会社/04.分析/analysis_project/notebooks/01_eda_old.ipynb"
        )
    )
