"""Kho vector Qdrant (chạy dạng tệp, không cần máy chủ) và model embedding FastEmbed (chạy trên máy)."""

from __future__ import annotations

from functools import lru_cache

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from . import config


@lru_cache(maxsize=1)
def embed_model() -> FastEmbedEmbedding:
    return FastEmbedEmbedding(model_name=config.EMBED_MODEL, cache_dir=str(config.STORAGE_DIR / "models"))


@lru_cache(maxsize=1)
def qdrant() -> QdrantClient:
    # Chế độ tệp chỉ cho một tiến trình mở cùng lúc, nên dùng chung một client.
    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.QDRANT_PATH))


def close() -> None:
    if qdrant.cache_info().currsize:
        qdrant().close()


@lru_cache(maxsize=1)
def vector_store() -> QdrantVectorStore:
    if config.HYBRID:
        return QdrantVectorStore(
            collection_name=config.COLLECTION,
            client=qdrant(),
            enable_hybrid=True,
            fastembed_sparse_model=config.SPARSE_MODEL,
        )
    return QdrantVectorStore(collection_name=config.COLLECTION, client=qdrant())


def index() -> VectorStoreIndex:
    if not qdrant().collection_exists(config.COLLECTION):
        raise RuntimeError("Chưa có dữ liệu. Chạy: python -m rag_erp ingest")
    return VectorStoreIndex.from_vector_store(vector_store(), embed_model=embed_model())
