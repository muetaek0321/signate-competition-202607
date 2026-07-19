import json
import os
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "./resource/pretrained"  # 事前学習モデルの保存先指定

import joblib
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    PythonLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from modules.check_office_password import is_password_protected, is_pdf_password_protected
from modules.custom_loader import (
    ExcelChunkLoader,
    ExcelStyleLoader,
    ImageDocumentLoader,
    MarkdownLoader,
    NotebookCellLoader,
    PDFDocumentImageLoader,
    PowerPointStyleLoader,
    WordDocumentStyleLoader,
    CsvChunkLoader
)
from modules.file_info_format import FILE_INFO_FORMAT_INTERNAL, FILE_INFO_FORMAT_PROJECT

# 環境変数の読み込み
load_dotenv()


def docs_ids(docs: list[Document], base_path: Path) -> tuple[list[Document], list[str]]:
    """Documentにファイル名を追加+DocumentのIDを作成"""
    updated_docs, ids = [], []
    for i, doc in enumerate(docs):
        doc.page_content = f"# ファイル名:{str(base_path)}  \n{doc.page_content}"
        updated_docs.append(doc)
        ids.append(f"{str(base_path)}_{i:03d}")

    return updated_docs, ids


def main() -> None:
    persist_directory = Path(os.getenv("DATASET_DIR", "./resource/chroma"))

    # ファイル情報のリストを読み込み
    file_info_list_path = persist_directory / "file_info_list.csv"
    if file_info_list_path.exists():
        file_info_df = pd.read_csv(file_info_list_path, encoding="utf-8-sig")
        file_info_dict = file_info_df.to_dict(orient="list")
    else:
        file_info_dict = {"path": [], "name": [], "extension": []}

    # 保存済みDocumentを読み込み
    all_docs_cache_path = persist_directory / "all_documents.joblib"
    if all_docs_cache_path.exists():
        all_docs: list[Document] = joblib.load(all_docs_cache_path)
    else:
        all_docs: list[Document] = []

    # スプリッタの用意
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        separators=["\n\n", "\n", "。", "、", ".", ",", "．", "，", " ", ""],
    )
    py_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=1500,
        chunk_overlap=250,
    )

    # ベクトル化する準備
    embedding_mode = os.getenv("EMBEDDING_MODE", "huggingface")
    if embedding_mode == "huggingface":
        embedding = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL_NAME", None),
            model_kwargs={"device": "cuda", "trust_remote_code": True},
        )
    elif embedding_mode == "ollama":
        embedding = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", None))
    else:
        raise ValueError(f"Invalid EMBEDDING_MODE: {embedding_mode}")

    # 全ファイルを読み込みリスト化
    vectorstores = {
        "shared_folder_all_documents": Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory,
            collection_name="shared_folder_all_documents",
        ),
        "shared_folder_csv_documents": Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory,
            collection_name="shared_folder_csv_documents",
        ),
        "shared_folder_excel_documents": Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory,
            collection_name="shared_folder_excel_documents",
        ),
        "shared_folder_file_info_list": Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory,
            collection_name="shared_folder_file_info_list",
        ),
    }

    error_logs = []
    batchsize = 500
    all_file_list = [
        path for path in Path(os.getenv("TARGET_DIR", "./")).glob("**/*") if path.is_file()
    ]
    num_files = len(all_file_list)
    for i, path in enumerate(all_file_list, start=1):
        ext = path.suffix.lower()
        print(f"[{i}/{num_files}][{ext}] {str(path)}")

        if str(path) in file_info_dict["path"]:
            print("  既に読み込み済みのためスキップ")
            continue

        # パスワードロックされているか
        is_encrypted = False

        try:
            if ext == ".csv":
                docs = CsvChunkLoader(path).load()
                docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    for i in range(0, len(docs), batchsize):
                        batch_docs = docs[i : i + batchsize]
                        batch_ids = ids[i : i + batchsize]
                        vectorstores["shared_folder_csv_documents"].add_documents(
                            batch_docs, ids=batch_ids
                        )
            elif ext == ".xlsx":
                is_encrypted, docs = is_password_protected(path)
                if not is_encrypted:
                    docs = ExcelChunkLoader(path).load()
                    docs += ExcelStyleLoader(path).load()
                    docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    for i in range(0, len(docs), batchsize):
                        batch_docs = docs[i : i + batchsize]
                        batch_ids = ids[i : i + batchsize]
                        vectorstores["shared_folder_excel_documents"].add_documents(
                            batch_docs, ids=batch_ids
                        )
            elif ext == ".tsv":
                docs = CsvChunkLoader(path, delimiter="\t").load()
                docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    for i in range(0, len(docs), batchsize):
                        batch_docs = docs[i : i + batchsize]
                        batch_ids = ids[i : i + batchsize]
                        vectorstores["shared_folder_all_documents"].add_documents(
                            batch_docs, ids=batch_ids
                        )
            elif ext == ".docx":
                is_encrypted, docs = is_password_protected(path)
                if not is_encrypted:
                    docs = UnstructuredWordDocumentLoader(path).load()
                    docs += WordDocumentStyleLoader(path).load()
                    docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    vectorstores["shared_folder_all_documents"].add_documents(docs, ids=ids)
            elif ext == ".pptx":
                is_encrypted, docs = is_password_protected(path)
                if not is_encrypted:
                    docs = UnstructuredPowerPointLoader(path).load()
                    docs += PowerPointStyleLoader(path).load()
                    docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    vectorstores["shared_folder_all_documents"].add_documents(docs, ids=ids)
            elif ext == ".py":
                docs = PythonLoader(path).load()
                docs = py_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    vectorstores["shared_folder_all_documents"].add_documents(docs, ids=ids)
            elif ext == ".ipynb":
                docs = NotebookCellLoader(file_path=path).load()
                docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    vectorstores["shared_folder_all_documents"].add_documents(docs, ids=ids)
            elif ext == ".pdf":
                is_encrypted, docs = is_pdf_password_protected(path)
                if not is_encrypted:
                    docs = PDFDocumentImageLoader(path).load()
                    docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    vectorstores["shared_folder_all_documents"].add_documents(docs, ids=ids)
            elif ext == ".md":
                docs = MarkdownLoader(path).load()
                docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    vectorstores["shared_folder_all_documents"].add_documents(docs, ids=ids)
            elif ext in [".png", ".jpg", "jpeg", ".bmp", ".tiff"]:
                docs = ImageDocumentLoader(path).load()
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    vectorstores["shared_folder_all_documents"].add_documents(docs, ids=ids)
            elif ext in [".json", ".txt", ".toml"]:
                docs = TextLoader(path, encoding="utf-8").load()
                docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    vectorstores["shared_folder_all_documents"].add_documents(docs, ids=ids)
            else:
                print("  未対応拡張子のためスキップ")
                continue
        except Exception as e:
            print(f"  読み込みエラーのためスキップ: {e}")
            error_logs.append({"file": str(path), "traceback": traceback.format_exc()})
            with open(persist_directory / "error_log.json", "w", encoding="utf-8") as f:
                json.dump(error_logs, f, ensure_ascii=False, indent=2)
            continue

        # ファイルの情報を記録してファイル情報をCSVで保存
        file_info_dict["path"].append(str(path))
        file_info_dict["name"].append(path.name)
        file_info_dict["extension"].append(ext)
        file_info_df = pd.DataFrame(file_info_dict)
        file_info_df.to_csv(
            file_info_list_path,
            encoding="utf-8-sig",
            index=None,
        )

        # ファイル情報のベクトル化
        path_parts = path.parts
        if path_parts[2] == "プロジェクト":
            file_info_str = FILE_INFO_FORMAT_PROJECT.format(
                filepath=str(path),
                filename=path.name,
                extension=path.suffix,
                directory_type=path_parts[2],
                company_name=path_parts[3],
                category=path_parts[4],
            )
        else:
            file_info_str = FILE_INFO_FORMAT_INTERNAL.format(
                filepath=str(path),
                filename=path.name,
                extension=path.suffix,
                directory_type=path_parts[2],
            )
        file_info_doc = Document(
            page_content=file_info_str,
            metadata={
                "source": str(path),
                "directory": str(path.parent),
                "extension": path.suffix,
                "is_encrypted": is_encrypted,
            },
        )
        vectorstores["shared_folder_file_info_list"].add_documents([file_info_doc], ids=[str(path)])

        # Documentをjoblibで保存
        joblib.dump(all_docs, all_docs_cache_path)


if __name__ == "__main__":
    main()

    # test
    query = "恒一会 かえで総合病院の提案書内で、重視するとされている評価指標を答えてください。"

    embedding_mode = os.getenv("EMBEDDING_MODE", "huggingface")
    if embedding_mode == "huggingface":
        embedding = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL_NAME", None),
            model_kwargs={"device": "cuda", "trust_remote_code": True},
        )
    elif embedding_mode == "ollama":
        embedding = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", None))
    else:
        raise ValueError(f"Invalid EMBEDDING_MODE: {embedding_mode}")

    # ベクトルDBから検索
    vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory=os.getenv("DATASET_DIR", "./resource/chroma"),
        collection_name="shared_folder_all_documents",
    )
    docs = vectorstore.similarity_search(query=query, k=10)

    for index, doc in enumerate(docs):
        print(f"{index + 1}: \n{doc.page_content}\n")
