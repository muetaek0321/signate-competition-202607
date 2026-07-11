import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "./resource/pretrained"  # 事前学習モデルの保存先指定

import joblib
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    PyPDFLoader,
    PythonLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from modules.custom_document_loader import csv_loader, notebook_loader

# from langchain_ollama.embeddings import OllamaEmbeddings

# 環境変数の読み込み
load_dotenv()


def main() -> None:
    # ファイル種別に対応するDocumentLoaderを設定
    loader_mapping = {
        ".csv": None,  # CSVLoader,
        ".tsv": lambda path: CSVLoader(path, csv_args={"delimiter": "\t"}),
        ".docx": UnstructuredWordDocumentLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".json": TextLoader,
        ".py": PythonLoader,
        ".ipynb": None,  # NotebookLoader,
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".toml": TextLoader,
        ".lock": TextLoader,
    }

    # スプリッタの用意
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        separators=["\n\n", "\n", "。", "、", ".", ",", "．", "，", " ", ""],
    )
    code_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=1000,
        chunk_overlap=200,
    )

    # 全ファイルを読み込みリスト化
    target_dir = Path(os.getenv("TARGET_DIR", "./"))
    all_docs: list[Document] = []
    all_csv_docs: list[Document] = []
    all_excel_docs: list[Document] = []
    for ext, file_loader in loader_mapping.items():
        print(f"extension: {ext}")

        docs = []
        if ext == ".csv":
            for csv_path in target_dir.glob("**/*.csv"):
                docs.extend(csv_loader(csv_path))
        # elif ext == ".xlsx":
        #     for excel_path in target_dir.glob("**/*.xlsx"):
        #         docs.extend(excel_loader(excel_path))
        elif ext == ".ipynb":
            for nb_path in target_dir.glob("**/*.ipynb"):
                docs.extend(notebook_loader(nb_path))
        else:
            # 共有フォルダ内のファイルを再帰的に読み込み
            loader = DirectoryLoader(
                path=target_dir,
                glob=f"**/*{ext}",
                loader_cls=file_loader,
                show_progress=True,
                silent_errors=True,
            )
            docs = loader.load()

        # 読込した内容を分割する
        if ext == ".py":
            docs = code_splitter.split_documents(docs)
        else:
            docs = text_splitter.split_documents(docs)

        # リストに追加
        if ext == ".csv":
            all_csv_docs.extend(docs)
        elif ext == ".xlsx":
            all_excel_docs.extend(docs)
        else:
            all_docs.extend(docs)

    # ファイル名をpage_contentに追加
    for doc in all_docs:
        doc.page_content = f"# ファイル名:{doc.metadata['source']}  \n{doc.page_content}"

    # Documentをjoblibで保存
    joblib.dump(all_docs + all_csv_docs + all_excel_docs, "./resource/all_documents.joblib")

    # ベクトル化する準備
    embedding = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL_NAME", None),
        model_kwargs={"device": "cuda", "trust_remote_code": True},
    )
    # embedding = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", None))

    start = time.perf_counter()

    # 読込した内容を保存
    Chroma.from_documents(
        documents=all_docs,
        embedding=embedding,
        persist_directory=os.getenv("DATASET_DIR", "./resource/chroma"),
        collection_name="shared_folder_all_documents",
    )

    Chroma.from_documents(
        documents=all_csv_docs,
        embedding=embedding,
        persist_directory=os.getenv("DATASET_DIR", "./resource/chroma"),
        collection_name="shared_folder_csv_documents",
    )

    Chroma.from_documents(
        documents=all_excel_docs,
        embedding=embedding,
        persist_directory=os.getenv("DATASET_DIR", "./resource/chroma"),
        collection_name="shared_folder_excel_documents",
    )

    print(f"DB作成時間: {time.perf_counter() - start}秒")


if __name__ == "__main__":
    main()

    # test
    query = "恒一会 かえで総合病院の提案書内で、重視するとされている評価指標を答えてください。"

    embedding = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL_NAME", None),
        model_kwargs={"device": "cuda", "trust_remote_code": True},
    )
    # embedding = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", None))

    # ベクトルDBから検索
    vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory=os.getenv("DATASET_DIR", "./chroma"),
        collection_name="shared_folder_all_documents",
    )
    docs = vectorstore.similarity_search(query=query, k=10)

    for index, doc in enumerate(docs):
        print(f"{index + 1}: \n{doc.page_content}\n")
