"""Hỏi đáp: LlamaIndex tìm các đoạn liên quan trong Qdrant, Claude trả lời kèm trích dẫn.

Mỗi đoạn tìm được gửi cho Claude thành một khối `document` bật `citations`, nên câu trả lời
trỏ đúng về đoạn văn nguồn thay vì để model tự ghi tên tài liệu.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import anthropic

from . import config, store, usage
from .prompts import ANSWER


@dataclass
class Source:
    file: str
    section: str
    score: float


@dataclass
class Citation:
    file: str
    section: str
    cited_text: str


@dataclass
class Answer:
    text: str
    sources: list[Source]
    citations: list[Citation] = field(default_factory=list)
    usage: usage.CallUsage | None = None


def _section(metadata: dict) -> str:
    path = (metadata.get("header_path") or "").strip("/")
    return path.replace("/", " › ")


def retrieve(question: str, top_k: int | None = None) -> list:
    k = top_k or config.TOP_K
    if config.HYBRID:
        retriever = store.index().as_retriever(similarity_top_k=k, sparse_top_k=k * 2, vector_store_query_mode="hybrid")
    else:
        retriever = store.index().as_retriever(similarity_top_k=k)
    return retriever.retrieve(question)


def ask(question: str, client: anthropic.Anthropic | None = None, top_k: int | None = None) -> Answer:
    client = client or anthropic.Anthropic()
    hits = retrieve(question, top_k)
    sources = [Source(h.node.metadata.get("file_name", "?"), _section(h.node.metadata), float(h.score or 0)) for h in hits]

    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": h.node.get_content()},
            "title": f"{s.file} › {s.section}" if s.section else s.file,
            "citations": {"enabled": True},
        }
        for h, s in zip(hits, sources)
    ]

    started = time.perf_counter()
    response = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=4000,
        system=ANSWER.system,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": [*documents, {"type": "text", "text": question}]}],
    )
    used = usage.record("answer", ANSWER.version, config.LLM_MODEL, response.usage, started)

    if response.stop_reason == "refusal":
        return Answer("Yêu cầu bị từ chối.", sources, usage=used)

    parts: list[str] = []
    citations: list[Citation] = []
    for block in response.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        for c in block.citations or []:
            s = sources[c.document_index]
            citations.append(Citation(s.file, s.section, c.cited_text.strip()))
    return Answer("".join(parts).strip(), sources, citations, used)
