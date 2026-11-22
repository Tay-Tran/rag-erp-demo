"""Cấu hình chung. Mọi giá trị đổi được bằng biến môi trường."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DOCS_DIR = ROOT / "data" / "tai-lieu"
VOUCHERS_DIR = ROOT / "data" / "chung-tu"
EVAL_DIR = ROOT / "data" / "eval"
STORAGE_DIR = ROOT / "storage"
QDRANT_PATH = STORAGE_DIR / "qdrant"
RESULTS_DIR = ROOT / "results"
REPLAY_DIR = ROOT / "data" / "replay"
USAGE_LOG = STORAGE_DIR / "usage.jsonl"

# Tìm kiếm kết hợp: vector ngữ nghĩa (dense) + từ khóa BM25 (sparse). Tắt bằng RAG_HYBRID=0.
HYBRID = os.getenv("RAG_HYBRID", "1") != "0"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION = os.getenv("RAG_COLLECTION", "erp_docs_hybrid" if HYBRID else "erp_docs")
LLM_MODEL = os.getenv("RAG_MODEL", "claude-haiku-5-5")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "384"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "48"))

# Giá theo USD cho 1 triệu token (bảng giá Anthropic, prompt dưới 100K token).
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
