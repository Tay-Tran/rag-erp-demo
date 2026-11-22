"""Nạp tài liệu: đọc Markdown, chia theo đề mục rồi theo câu, tạo embedding, lưu vào Qdrant."""

from __future__ import annotations

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

from . import config, store


def ingest() -> dict[str, int]:
    documents = SimpleDirectoryReader(str(config.DOCS_DIR), required_exts=[".md"]).load_data()

    client = store.qdrant()
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)

    pipeline = IngestionPipeline(
        transformations=[
            MarkdownNodeParser(),
            SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP),
            store.embed_model(),
        ],
        vector_store=store.vector_store(),
    )
    nodes = pipeline.run(documents=documents, show_progress=False)
    return {"documents": len(documents), "chunks": len(nodes)}
