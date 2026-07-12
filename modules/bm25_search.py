from pathlib import Path

import joblib
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sudachipy import Dictionary

DOCUMENT_CACHE_NAME = "all_documents.joblib"


class BM25DocumentSearch:
    def __init__(self, dir_path: Path, k: int = 50) -> None:
        self.all_docs = joblib.load(dir_path / DOCUMENT_CACHE_NAME)
        self.tokenizer = Dictionary(dict="full").create()

        self.bm25_retriever = BM25Retriever.from_documents(
            self.all_docs,
            preprocess_func=self._preprocess,
            k=k,
        )

    def __call__(self, query: str) -> list[Document]:
        """BM25Retrieverを用いて、クエリに対する類似文書を取得する"""
        return self.bm25_retriever.invoke(query)

    def _preprocess(self, text: str) -> list[str]:
        """日本語の文章を形態素解析してトークン化する"""
        return [m.surface() for m in self.tokenizer.tokenize(text)]
