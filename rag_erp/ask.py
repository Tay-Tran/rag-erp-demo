"""Hỏi đáp: LlamaIndex tìm các đoạn liên quan trong Qdrant, Claude trả lời kèm trích dẫn.

Mỗi đoạn tìm được gửi cho Claude thành một khối `document` bật `citations`, nên câu trả lời
trỏ đúng về đoạn văn nguồn thay vì để model tự ghi tên tài liệu.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import anthropic

from . import config, replay, store, usage
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


def _section(node) -> str:
    """Đường dẫn đề mục: các đề mục cha (header_path của LlamaIndex) cộng đề mục của chính đoạn."""
    # strip(): tệp tài liệu có thể mang \r\n khi git trên Windows đổi kiểu xuống dòng lúc clone.
    parts = [p.strip() for p in (node.metadata.get("header_path") or "").split("/") if p.strip()]
    first = node.get_content().strip().splitlines()[0].strip() if node.get_content().strip() else ""
    if first.startswith("#"):
        parts.append(first.lstrip("#").strip())
    return " › ".join(parts)


def retrieve(question: str, top_k: int | None = None) -> list:
    k = top_k or config.TOP_K
    if config.HYBRID:
        retriever = store.index().as_retriever(similarity_top_k=k, sparse_top_k=k * 2, vector_store_query_mode="hybrid")
    else:
        retriever = store.index().as_retriever(similarity_top_k=k)
    return retriever.retrieve(question)


def ask(question: str, client: anthropic.Anthropic | None = None, top_k: int | None = None) -> Answer:
    """Không có khóa API thì phát lại bản ghi (Answer.usage.replayed_from có ngày ghi),
    hoặc ném replay.NoRecording nếu câu hỏi chưa từng được chạy thật."""
    hits = retrieve(question, top_k)
    sources = [Source(h.node.metadata.get("file_name", "?"), _section(h.node), float(h.score or 0)) for h in hits]

    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": h.node.get_content()},
            "title": f"{s.file} › {s.section}" if s.section else s.file,
            "citations": {"enabled": True},
        }
        for h, s in zip(hits, sources)
    ]

    key = replay.record_key("answer", ANSWER.version, config.LLM_MODEL, {"question": question, "documents": documents})
    if replay.mode() == "replay":
        rec = replay.load("answer", key, question)
        response = rec.message
        used = usage.from_recording("answer", ANSWER.version, rec.model, response.usage, rec.latency_ms, rec.recorded_at)
    else:
        client = client or anthropic.Anthropic()
        started = time.perf_counter()
        response = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=4000,
            system=ANSWER.system,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": [*documents, {"type": "text", "text": question}]}],
        )
        used = usage.record("answer", ANSWER.version, config.LLM_MODEL, response.usage, started)
        replay.save("answer", key, question, config.LLM_MODEL, response, used.latency_ms)

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
