import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_ollama.embeddings import OllamaEmbeddings


def get_embedding() -> Embeddings:
    """Embeddingモデルの読み込み"""
    # 使用するEmbeddingモデルの種別を取得
    embedding_mode = os.getenv("EMBEDDING_MODE", "huggingface")

    # Embeddingモデルの読み込み
    if embedding_mode == "huggingface":
        embedding = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL_NAME", None),
            model_kwargs={"device": "cuda", "trust_remote_code": True},
        )
    elif embedding_mode == "ollama":
        embedding = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME", None))
    else:
        raise ValueError(f"Invalid EMBEDDING_MODE: {embedding_mode}")

    return embedding
