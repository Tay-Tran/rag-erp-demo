"""Trích xuất báo giá thành dữ liệu có cấu trúc bằng Claude (structured output theo Pydantic),
rồi kiểm tra bằng quy tắc cố định ở checks.py."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import anthropic

from . import config, replay, usage
from .checks import VanDe, kiem_tra
from .prompts import EXTRACT
from .schema import BaoGia


@dataclass
class KetQuaTrichXuat:
    bao_gia: BaoGia
    van_de: list[VanDe]
    usage: usage.CallUsage


class ExtractionError(RuntimeError):
    pass


def trich_xuat(path: Path, client: anthropic.Anthropic | None = None) -> KetQuaTrichXuat:
    """Không có khóa API thì phát lại bản ghi của đúng chứng từ này (cùng nội dung tệp),
    hoặc ném replay.NoRecording. Phần kiểm tra luôn chạy lại thật bằng Python."""
    text = path.read_text(encoding="utf-8")
    key = replay.record_key("extract", EXTRACT.version, config.LLM_MODEL, {"file": text})

    if replay.mode() == "replay":
        rec = replay.load("extract", key, path.name)
        response = rec.message
        used = usage.from_recording("extract", EXTRACT.version, rec.model, response.usage, rec.latency_ms, rec.recorded_at)
        body = next((b.text for b in response.content if b.type == "text"), None)
        bao_gia = BaoGia.model_validate_json(body) if body else None
    else:
        client = client or anthropic.Anthropic()
        started = time.perf_counter()
        response = client.messages.parse(
            model=config.LLM_MODEL,
            max_tokens=8000,
            system=EXTRACT.system,
            messages=[{"role": "user", "content": f"<chung_tu ten_tep=\"{path.name}\">\n{text}\n</chung_tu>"}],
            output_format=BaoGia,
        )
        used = usage.record("extract", EXTRACT.version, config.LLM_MODEL, response.usage, started)
        bao_gia = response.parsed_output
        if bao_gia is not None and response.stop_reason not in ("refusal", "max_tokens"):
            replay.save("extract", key, path.name, config.LLM_MODEL, response, used.latency_ms)

    if response.stop_reason in ("refusal", "max_tokens") or bao_gia is None:
        raise ExtractionError(f"{path.name}: không đọc được chứng từ (stop_reason={response.stop_reason})")
    return KetQuaTrichXuat(bao_gia=bao_gia, van_de=kiem_tra(bao_gia), usage=used)
