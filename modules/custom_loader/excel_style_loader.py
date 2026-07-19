from __future__ import annotations

from io import BytesIO
from pathlib import Path

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from openpyxl import load_workbook
from PIL import Image

from .image_file_loader import ImageDocumentLoader


class ExcelStyleLoader(BaseLoader):
    """
    Excel(.xlsx)用カスタムLoader

    1シート = 1Document

    取得情報
    --------
    ・セル値
    ・数式
    ・フォント
    ・背景色
    ・文字色
    ・太字
    ・斜体
    ・下線
    ・配置
    ・コメント
    ・ハイパーリンク
    ・結合セル
    ・画像
    ・グラフ（種類, タイトル, 軸, 系列情報）
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def load(self) -> list[Document]:

        wb = load_workbook(
            self.file_path,
            data_only=False,
        )

        style_docs = []
        for ws in wb.worksheets:
            sheet_parts = []

            ###########################################################
            # セル
            ###########################################################

            for row in ws.iter_rows():
                for cell in row:
                    if cell.value is None:
                        continue

                    font_color = None
                    if cell.font.color and cell.font.color.type == "rgb":
                        font_color = cell.font.color.rgb

                    fill_color = None
                    if cell.fill.fill_type == "solid" and cell.fill.fgColor.type == "rgb":
                        fill_color = cell.fill.fgColor.rgb

                    cell_info = {
                        "coordinate": cell.coordinate,
                        "row": cell.row,
                        "column": cell.column,
                        "value": str(cell.value),
                        "style": cell.style_id,
                        "font_name": cell.font.name,
                        "font_size": (cell.font.sz if cell.font.sz else None),
                        "bold": cell.font.bold,
                        "italic": cell.font.italic,
                        "underline": cell.font.underline,
                        "font_color": font_color,
                        "fill_color": fill_color,
                        "alignment": {
                            "horizontal": cell.alignment.horizontal,
                            "vertical": cell.alignment.vertical,
                            "wrap_text": cell.alignment.wrap_text,
                        },
                        "comment": (cell.comment.text if cell.comment else None),
                        "hyperlink": (cell.hyperlink.target if cell.hyperlink else None),
                    }

                    sheet_parts.append(
                        {
                            "type": "cell",
                            "data": cell_info,
                        }
                    )

            ###########################################################
            # 結合セル
            ###########################################################

            merged_cells = [str(rng) for rng in ws.merged_cells.ranges]

            ###########################################################
            # 画像
            ###########################################################

            image_idxes, image_store_ids = [], []
            image_descriptions = []

            for index, img in enumerate(ws._images):
                image_bytes = img._data()

                image = Image.open(BytesIO(image_bytes))

                response_date, image_store_id = ImageDocumentLoader(self.file_path)._describe_image(
                    image
                )
                description = response_date[1]["text"]

                image_idxes.append(index)
                image_store_ids.append(image_store_id)
                image_descriptions.append({"image_index": index, "description": description})

            sheet_parts.append({"type": "image", "data": image_descriptions})

            ###########################################################
            # グラフ
            ###########################################################

            chart_descriptions = []
            for chart_index, chart in enumerate(ws._charts):
                chart_info = self._extract_chart_info(chart, chart_index)
                chart_descriptions.append(chart_info)

            sheet_parts.append({"type": "chart", "data": chart_descriptions})

            ###########################################################
            # metadata
            ###########################################################

            metadata = {
                "source": str(self.file_path),
                "sheet_name": ws.title,
                "sheet_state": ws.sheet_state,
                "max_row": ws.max_row,
                "max_column": ws.max_column,
                "merged_cells": merged_cells if len(merged_cells) > 0 else None,
                "image_count": len(image_idxes),
                "image_idxes": image_idxes if len(image_idxes) > 0 else None,
                "image_store_ids": image_store_ids if len(image_store_ids) > 0 else None,
                "chart_count": len(chart_descriptions),
            }

            style_docs.append(
                Document(
                    page_content=self._build_text(
                        ws.title,
                        sheet_parts,
                        merged_cells,
                    ),
                    metadata=metadata,
                )
            )

        return style_docs

    ##################################################################

    def _build_text(
        self,
        sheet_name,
        sheet_parts,
        merged_cells,
    ):

        texts = []

        texts.append(f"# Sheet: {sheet_name}")
        texts.append("")

        if merged_cells:
            texts.append("[Merged Cells]")

            for m in merged_cells:
                texts.append(m)

            texts.append("")

        for part in sheet_parts:
            if part["type"] == "cell":
                cell = part["data"]

                texts.append(f"Cell={cell['coordinate']}")

                texts.append(f"Value={cell['value']}")

                if cell["bold"]:
                    texts.append("[BOLD]")

                if cell["italic"]:
                    texts.append("[ITALIC]")

                if cell["underline"]:
                    texts.append("[UNDERLINE]")

                if cell["fill_color"]:
                    texts.append(f"[FILL:{cell['fill_color']}]")

                if cell["font_color"]:
                    texts.append(f"[FONT_COLOR:{cell['font_color']}]")

                if cell["font_name"]:
                    texts.append(f"[FONT:{cell['font_name']}]")

                if cell["font_size"]:
                    texts.append(f"[SIZE:{cell['font_size']}]")

                if cell["comment"]:
                    texts.append(f"[COMMENT]{cell['comment']}")

                if cell["hyperlink"]:
                    texts.append(f"[LINK]{cell['hyperlink']}")

            elif part["type"] == "image":
                texts.append("[IMAGE]")

                for desc in part["data"]:
                    texts.append(f"Index {desc['image_index']}")
                    texts.append(desc["description"])

            elif part["type"] == "chart":
                for chart_desc in part["data"]:
                    texts.append(f"[CHART:{chart_desc['chart_type']}]")

                    if chart_desc["title"]:
                        texts.append(f"Title={chart_desc['title']}")

                    if chart_desc["x_axis_title"]:
                        texts.append(f"X_Axis={chart_desc['x_axis_title']}")

                    if chart_desc["y_axis_title"]:
                        texts.append(f"Y_Axis={chart_desc['y_axis_title']}")

                    if chart_desc["style"] is not None:
                        texts.append(f"Style={chart_desc['style']}")

                    for s in chart_desc["series"]:
                        texts.append(f"  Series[{s['index']}]")

                        if s.get("title"):
                            texts.append(f"    Name={s['title']}")

                        if s.get("values_ref"):
                            texts.append(f"    Values={s['values_ref']}")

                        if s.get("categories_ref"):
                            texts.append(f"    Categories={s['categories_ref']}")

            texts.append("")

        return "\n".join(texts)

    def _extract_chart_info(self, chart, chart_index: int) -> dict:
        """グラフオブジェクトから情報を抽出する"""

        chart_type = type(chart).__name__

        # タイトル
        title = None
        if chart.title is not None:
            title = str(chart.title)

        # 軸タイトル
        x_axis_title = None
        y_axis_title = None
        if hasattr(chart, "x_axis") and chart.x_axis is not None:
            if chart.x_axis.title is not None:
                x_axis_title = str(chart.x_axis.title)
        if hasattr(chart, "y_axis") and chart.y_axis is not None:
            if chart.y_axis.title is not None:
                y_axis_title = str(chart.y_axis.title)

        # 系列情報
        series_list = []
        for s_idx, series in enumerate(chart.series):
            series_info = {
                "index": s_idx,
                "title": str(series.title) if series.title else None,
            }

            # データ参照範囲
            if hasattr(series, "val") and series.val is not None:
                num_ref = getattr(series.val, "numRef", None)
                if num_ref is not None and hasattr(num_ref, "f"):
                    series_info["values_ref"] = str(num_ref.f)

            # カテゴリ参照範囲
            if hasattr(series, "cat") and series.cat is not None:
                cat_ref = (
                    getattr(series.cat, "strRef", None)
                    or getattr(series.cat, "numRef", None)
                )
                if cat_ref is not None and hasattr(cat_ref, "f"):
                    series_info["categories_ref"] = str(cat_ref.f)

            series_list.append(series_info)

        # スタイル
        style = getattr(chart, "style", None)

        return {
            "chart_index": chart_index,
            "chart_type": chart_type,
            "title": title,
            "x_axis_title": x_axis_title,
            "y_axis_title": y_axis_title,
            "series": series_list,
            "style": style,
        }


if __name__ == "__main__":
    path = r"..\..\share\共有ドライブ\プロジェクト\株式会社青潮モビリティサービス\03.データ\train.xlsx"
    loader = ExcelStyleLoader(path)
    docs = loader.load()
    print(docs[0].metadata)
    print(docs[0].page_content)
    