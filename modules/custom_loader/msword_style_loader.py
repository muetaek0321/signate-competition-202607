from io import BytesIO
from pathlib import Path

from docx import Document as WordDocument
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from PIL import Image

from .image_file_loader import ImageDocumentLoader


class WordDocumentStyleLoader(BaseLoader):
    """
    Word(.docx)用カスタムLoader

    取得する情報
    ----------
    ・本文
    ・段落スタイル
    ・見出しレベル
    ・Run単位の書式
        - 太字
        - 斜体
        - 下線
        - 色
        - ハイライト
        - フォント
        - サイズ
    ・表
    ・画像
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def load(self) -> list[Document]:

        doc = WordDocument(self.file_path)

        parts = []

        style_docs = []

        ############################################
        # Paragraph
        ############################################

        for idx, paragraph in enumerate(doc.paragraphs):
            paragraph_info = {
                "index": idx,
                "style": paragraph.style.name,
                "text": paragraph.text,
                "runs": [],
            }

            if paragraph.style.name.startswith("Heading"):
                paragraph_info["heading_level"] = paragraph.style.name

            for run in paragraph.runs:
                font = run.font

                color = None
                if font.color.rgb:
                    color = str(font.color.rgb)

                highlight = None
                if font.highlight_color:
                    highlight = str(font.highlight_color)

                run_info = {
                    "text": run.text,
                    "bold": font.bold,
                    "italic": font.italic,
                    "underline": font.underline,
                    "font_name": font.name,
                    "font_size": font.size.pt if font.size else None,
                    "font_color": color,
                    "highlight": highlight,
                }

                paragraph_info["runs"].append(run_info)

            parts.append(
                {
                    "type": "paragraph",
                    "data": paragraph_info,
                }
            )

        ############################################
        # Tables
        ############################################

        for t_idx, table in enumerate(doc.tables):
            rows = []

            for row in table.rows:
                rows.append([cell.text for cell in row.cells])

            parts.append(
                {
                    "type": "table",
                    "data": {
                        "table_index": t_idx,
                        "rows": rows,
                    },
                }
            )

        ############################################
        # Images
        ############################################

        image_idxes, image_store_ids = [], []
        rels = doc.part.rels
        image_index = 0
        image_descriptions = []

        for rel in rels.values():
            if "image" not in rel.target_ref:
                continue

            image_bytes = rel.target_part.blob

            image = Image.open(BytesIO(image_bytes))

            response_date, image_store_id = ImageDocumentLoader(self.file_path)._describe_image(
                image
            )
            description = response_date[1]["text"]

            image_idxes.append(image_index)
            image_store_ids.append(image_store_id)
            image_descriptions.append({"image_index": image_index, "description": description})

            image_index += 1

        parts.append({"type": "image", "data": image_descriptions})

        ############################################

        metadata = {
            "source": str(self.file_path),
            "paragraph_count": len(doc.paragraphs),
            "table_count": len(doc.tables),
            "image_count": len(image_idxes),
            "image_idxes": image_idxes if len(image_idxes) > 0 else None,
            "image_store_ids": image_store_ids if len(image_store_ids) > 0 else None,
        }

        style_docs.append(
            Document(
                page_content=self._build_text(parts),
                metadata=metadata,
            )
        )

        return style_docs

    def _build_text(self, parts):

        texts = []

        for part in parts:
            if part["type"] == "paragraph":
                p = part["data"]

                texts.append(f"[{p['style']}]")

                texts.append(p["text"])

                texts.append("")

            elif part["type"] == "table":
                texts.append("[TABLE]")

                for row in part["data"]["rows"]:
                    texts.append(" | ".join(row))

                texts.append("")

            elif part["type"] == "image":
                texts.append("[IMAGE]")

                for desc in part["data"]:
                    texts.append(f"Index {desc['image_index']}")
                    texts.append(desc["description"])

                texts.append("")

        return "\n".join(texts)
