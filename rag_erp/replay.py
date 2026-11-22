"""Ghi lại câu trả lời thật của Claude và phát lại khi máy không có khóa API.

- Có khóa API: gọi Claude như thường, và lưu nguyên câu trả lời vào data/replay/<loại>.json.
- Không có khóa: không gọi Claude; nếu đã có bản ghi cho đúng yêu cầu đó thì trả lại bản ghi,
  kèm ngày ghi và model, để màn hình nói rõ đây là câu trả lời ghi sẵn.

Khóa của một bản ghi gồm phiên bản prompt, model và toàn bộ nội dung gửi đi (câu hỏi cùng các đoạn
tài liệu tìm được, hoặc nội dung chứng từ). Sửa tài liệu, prompt hay model thì bản ghi cũ không còn
khớp, nên không bao giờ phát lại một câu trả lời cho câu hỏi khác.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from anthropic.types import Message

from . import config


def mode() -> str:
    """'live' khi có khóa API (hoặc RAG_MODE=live), 'replay' khi không có hoặc RAG_MODE=replay."""
    forced = os.getenv("RAG_MODE", "auto").lower()
    if forced in ("live", "replay"):
        return forced
    return "live" if os.getenv("ANTHROPIC_API_KEY") else "replay"


def record_key(kind: str, prompt_version: str, model: str, payload: Any) -> str:
    raw = json.dumps([kind, prompt_version, model, payload], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


@dataclass
class Recording:
    message: Message
    recorded_at: str
    model: str
    latency_ms: int
    label: str


class NoRecording(LookupError):
    """Không có khóa API và chưa có bản ghi cho yêu cầu này."""

    def __init__(self, kind: str, label: str):
        super().__init__(label)
        self.kind = kind
        self.label = label


def _path(kind: str):
    return config.REPLAY_DIR / f"{kind}.json"


def _load(kind: str) -> dict[str, dict]:
    p = _path(kind)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save(kind: str, key: str, label: str, model: str, response: Any, latency_ms: int) -> None:
    data = _load(kind)
    data[key] = {
        "label": label,
        "recorded_at": time.strftime("%Y-%m-%d"),
        "model": model,
        "latency_ms": latency_ms,
        "response": response.model_dump(mode="json"),
    }
    config.REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    _path(kind).write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load(kind: str, key: str, label: str) -> Recording:
    item = _load(kind).get(key)
    if item is None:
        raise NoRecording(kind, label)
    return Recording(
        message=Message.model_validate(item["response"]),
        recorded_at=item["recorded_at"],
        model=item["model"],
        latency_ms=item["latency_ms"],
        label=item["label"],
    )


def recorded_labels(kind: str) -> list[str]:
    return sorted({v["label"] for v in _load(kind).values()})


def banner(recorded_at: str, model: str) -> str:
    """Dòng giải thích hiện ở đầu mọi kết quả phát lại, viết cho người không làm kỹ thuật."""
    d = time.strptime(recorded_at, "%Y-%m-%d")
    return (
        f"LƯU Ý: Đây là câu trả lời đã ghi lại từ một lần chạy thật với Claude ({model}) ngày {time.strftime('%d/%m/%Y', d)}. "
        "Lần này máy không gọi Claude vì chưa có khóa API. "
        "Phần tìm tài liệu và phần kiểm tra lỗi báo giá vẫn chạy thật trên máy."
    )


def missing_message(kind: str) -> str:
    what = "câu hỏi này" if kind == "answer" else "chứng từ này"
    lines = [
        f"Chưa có câu trả lời ghi sẵn cho {what}.",
        "Máy chưa có khóa API của Claude, nên chỉ phát lại được những yêu cầu đã chạy thật trước đó.",
        f"Muốn {'hỏi câu mới' if kind == 'answer' else 'đọc chứng từ mới'}: điền ANTHROPIC_API_KEY vào tệp .env (xem README) rồi chạy lại.",
    ]
    labels = recorded_labels(kind)
    if labels:
        lines.append("")
        lines.append("Các câu đã có bản ghi, thử ngay được:" if kind == "answer" else "Các chứng từ đã có bản ghi:")
        lines += [f"  - {label}" for label in labels]
    else:
        lines.append("Hiện chưa có bản ghi nào. Bản ghi được tạo tự động ở lần đầu chạy với khóa API.")
    return "\n".join(lines)
