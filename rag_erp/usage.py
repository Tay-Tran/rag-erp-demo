"""Ghi chi phí và độ trễ của từng lần gọi Claude vào storage/usage.jsonl."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

from . import config


@dataclass
class CallUsage:
    kind: str
    prompt_version: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = config.PRICES_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def record(kind: str, prompt_version: str, model: str, usage, started: float) -> CallUsage:
    """`usage` là response.usage của Anthropic SDK; `started` là time.perf_counter() trước khi gọi."""
    item = CallUsage(
        kind=kind,
        prompt_version=prompt_version,
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost_usd(model, usage.input_tokens, usage.output_tokens),
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with config.USAGE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **asdict(item)}, ensure_ascii=False) + "\n")
    return item
