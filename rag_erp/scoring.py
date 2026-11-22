"""Hàm chấm điểm thuần (không gọi API), dùng cho bộ đo ở evaluate.py."""

from __future__ import annotations

import unicodedata

from .prompts import REFUSAL


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def answer_correct(answer: str, must_include: list[list[str]]) -> bool:
    """Đúng khi mỗi nhóm có ít nhất một cách viết xuất hiện trong câu trả lời."""
    a = normalize(answer)
    return all(any(normalize(alt) in a for alt in group) for group in must_include)


def is_refusal(answer: str) -> bool:
    a = normalize(answer)
    return normalize(REFUSAL) in a or "không tìm thấy thông tin" in a


def retrieval_hit(retrieved: list[str], expected: list[str]) -> bool:
    return all(src in retrieved for src in expected)


def field_matches(actual, expected) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return normalize(expected) in normalize(actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(actual - expected) < 1e-6
    return actual == expected
