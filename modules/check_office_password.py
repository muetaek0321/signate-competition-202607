from pathlib import Path

import msoffcrypto
from langchain_core.documents import Document
from pypdf import PdfReader


def is_password_protected(file_path: str | Path) -> tuple[bool, list[Document]]:
    """Officeファイルが暗号化されているか判定する"""
    path = Path(file_path)

    with path.open("rb") as f:
        office_file = msoffcrypto.OfficeFile(f)
        is_encrypted = office_file.is_encrypted()

    docs = []
    if is_encrypted:
        docs.append(
            Document(
                metadata={
                    "source": str(file_path),
                    "file_name": path.name,
                    "extension": path.suffix,
                    "encrypted": True,
                    "password_required": True,
                    "index_status": "metadata_only",
                },
                page_content=f"{path.name}: パスワード保護されているため開けません",
            )
        )

    return is_encrypted, docs


def is_pdf_password_protected(file_path: str | Path) -> tuple[bool, list[Document]]:
    """PDFファイルがパスワード保護されているか判定する"""
    path = Path(file_path)

    reader = PdfReader(path)
    is_encrypted = reader.is_encrypted

    docs = []
    if is_encrypted:
        docs.append(
            Document(
                metadata={
                    "source": str(file_path),
                    "file_name": path.name,
                    "extension": path.suffix,
                    "encrypted": True,
                    "password_required": True,
                    "index_status": "metadata_only",
                },
                page_content=f"{path.name}: パスワード保護されているため開けません",
            )
        )

    return is_encrypted, docs


if __name__ == "__main__":
    path = "../share/共有ドライブ/プロジェクト/医療法人社団 蒼樹会 みなみ野女性医療センター/01.契約/契約書.docx"
    print(is_password_protected(path))

    path = "../share/共有ドライブ/プロジェクト/医療法人社団 恒一会 かえで総合病院/01.契約/契約書_pw-kaede20250902.docx"
    print(is_password_protected(path))

    path = "../share/共有ドライブ/プロジェクト/医療法人社団 蒼樹会 みなみ野女性医療センター/02.計画/スケジュール.xlsx"
    print(is_password_protected(path))

    path = "../share/共有ドライブ/プロジェクト/医療法人社団 恒一会 かえで総合病院/02.計画/スケジュール.xlsx"
    print(is_password_protected(path))

    path = "../share/共有ドライブ/プロジェクト/医療法人社団 蒼樹会 みなみ野女性医療センター/00.提案/提案書.pptx"
    print(is_password_protected(path))

    path = "../share/共有ドライブ/プロジェクト/医療法人社団 蒼樹会 みなみ野女性医療センター/06.報告書/医療法人社団 蒼樹会 みなみ野女性医療センター_最終報告.pdf"
    print(is_pdf_password_protected(path))
