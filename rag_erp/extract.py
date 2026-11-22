"""Trích xuất báo giá thành dữ liệu có cấu trúc bằng Claude (structured output theo Pydantic),
rồi kiểm tra bằng quy tắc cố định ở checks.py."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import anthropic

from . import config, usage
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
    client = client or anthropic.Anthropic()
    text = path.read_text(encoding="utf-8")
    started = time.perf_counter()
    response = client.messages.parse(
        model=config.LLM_MODEL,
        max_tokens=8000,
        system=EXTRACT.system,
        messages=[{"role": "user", "content": f"<chung_tu ten_tep=\"{path.name}\">\n{text}\n</chung_tu>"}],
        output_format=BaoGia,
    )
    used = usage.record("extract", EXTRACT.version, config.LLM_MODEL, response.usage, started)
    if response.stop_reason in ("refusal", "max_tokens") or response.parsed_output is None:
        raise ExtractionError(f"{path.name}: không đọc được chứng từ (stop_reason={response.stop_reason})")
    bao_gia = response.parsed_output
    return KetQuaTrichXuat(bao_gia=bao_gia, van_de=kiem_tra(bao_gia), usage=used)
