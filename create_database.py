import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "./pretrained"  # 事前学習モデルの保存先指定

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    NotebookLoader,
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

# from langchain_ollama.embeddings import OllamaEmbeddings

# 環境変数の読み込み
load_dotenv()


def main() -> None:
    # ファイル種別に対応するDocumentLoaderを設定
    loader_mapping = {
        ".csv": CSVLoader,
        ".tsv": lambda path: CSVLoader(path, csv_args={"delimiter": "\t"}),
        ".docx": UnstructuredWordDocumentLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".json": TextLoader,
        ".py": PythonLoader,
        ".ipynb": NotebookLoader,
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".toml": TextLoader,
        ".lock": TextLoader,
    }

    # スプリッタの用意
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
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
    for ext, file_loader in loader_mapping.items():
        print(f"extension: {ext}")

        docs = []
        # if ext == ".csv":
        #     for csv_path in target_dir.glob("**/*.csv"):
        #         docs.extend(csv_loader(csv_path))
        # elif ext == ".xlsx":
        #     for excel_path in target_dir.glob("**/*.xlsx"):
        #         docs.extend(excel_loader(excel_path))
        # elif ext == ".ipynb":
        #     for nb_path in target_dir.glob("**/*.ipynb"):
        #         docs.append(notebook_loader(nb_path))
        # else:
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
        all_docs.extend(docs)

    # ベクトル化する準備
    embedding = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL_NAME", None),
        model_kwargs={"device": "cuda", "trust_remote_code": True},
    )
    # embedding = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", None))

    start = time.perf_counter()

    with open("./check_result/data_length.csv", mode="w", encoding="utf-8-sig") as f:
        for doc in all_docs:
            f.write(f"{doc.metadata['source']},{len(doc.page_content)}\n")

    # 読込した内容を保存
    Chroma.from_documents(
        documents=all_docs,
        embedding=embedding,
        persist_directory=os.getenv("DATASET_DIR", "./chroma"),
        collection_name="shared_folder_documents",
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
        collection_name="shared_folder_documents",
    )
    docs = vectorstore.similarity_search(query=query, k=10)

    for index, doc in enumerate(docs):
        print(f"{index + 1}: \n{doc.page_content}\n")
