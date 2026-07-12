import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "./resource/pretrained"  # 事前学習モデルの保存先指定

import joblib
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    PythonLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from modules.custom_document_loader import csv_loader, notebook_loader

# from langchain_huggingface.embeddings import HuggingFaceEmbeddings

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
    # embedding = HuggingFaceEmbeddings(
    #     model_name=os.getenv("EMBEDDING_MODEL_NAME", None),
    #     model_kwargs={"device": "cuda", "trust_remote_code": True},
    # )
    embedding = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", None))

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
    }

    batchsize = 500
    all_file_list = list(Path(os.getenv("TARGET_DIR", "./")).glob("**/*"))
    num_files = len(all_file_list)
    for i, path in enumerate(all_file_list, start=1):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        print(f"[{i}/{num_files}][{ext}] {str(path)}")

        if str(path) in file_info_dict["path"]:
            print("  既に読み込み済みのためスキップ")
            continue

        try:
            if ext == ".csv":
                docs = csv_loader(path)
                docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    for i in range(0, len(docs), batchsize):
                        batch_docs = docs[i : i + batchsize]
                        vectorstores["shared_folder_csv_documents"].add_documents(
                            batch_docs, ids=ids[i : i + batchsize]
                        )
            elif ext == ".xlsx":
                docs = UnstructuredExcelLoader(path).load()
                docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    for i in range(0, len(docs), batchsize):
                        batch_docs = docs[i : i + batchsize]
                        vectorstores["shared_folder_excel_documents"].add_documents(
                            batch_docs, ids=ids[i : i + batchsize]
                        )
            elif ext == ".tsv":
                docs = CSVLoader(path, csv_args={"delimiter": "\t"}).load()
                docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    vectorstores["shared_folder_all_documents"].add_documents(docs, ids=ids)
            elif ext == ".docx":
                docs = UnstructuredWordDocumentLoader(path).load()
                docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    vectorstores["shared_folder_all_documents"].add_documents(docs, ids=ids)
            elif ext == ".pptx":
                docs = UnstructuredPowerPointLoader(path).load()
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
                docs = notebook_loader(path)
                docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    vectorstores["shared_folder_all_documents"].add_documents(docs, ids=ids)
            elif ext == ".pdf":
                docs = PyPDFLoader(path).load()
                docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    vectorstores["shared_folder_all_documents"].add_documents(docs, ids=ids)
            elif ext in [".json", ".txt", ".md", ".toml", ".lock"]:
                docs = TextLoader(path).load()
                docs = text_splitter.split_documents(docs)
                docs, ids = docs_ids(docs, path)
                all_docs.extend(docs)
                if docs:
                    vectorstores["shared_folder_all_documents"].add_documents(docs, ids=ids)
            else:
                print("  未対応拡張子のためスキップ")
                continue
        except Exception as e:
            print(f"  読み込みに失敗したためスキップ: {e}")
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
        # Documentをjoblibで保存
        joblib.dump(all_docs, all_docs_cache_path)


if __name__ == "__main__":
    main()

    # test
    query = "恒一会 かえで総合病院の提案書内で、重視するとされている評価指標を答えてください。"

    # embedding = HuggingFaceEmbeddings(
    #     model_name=os.getenv("EMBEDDING_MODEL_NAME", None),
    #     model_kwargs={"device": "cuda", "trust_remote_code": True},
    # )
    embedding = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", None))

    # ベクトルDBから検索
    vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory=os.getenv("DATASET_DIR", "./resource/chroma"),
        collection_name="shared_folder_all_documents",
    )
    docs = vectorstore.similarity_search(query=query, k=10)

    for index, doc in enumerate(docs):
        print(f"{index + 1}: \n{doc.page_content}\n")
