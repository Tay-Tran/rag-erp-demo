"""Bộ đo độ chính xác: hỏi đáp (tìm đúng nguồn, trả lời đúng, từ chối đúng, có trích dẫn)
và trích xuất báo giá (đúng từng trường, phát hiện đúng vấn đề)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import anthropic

from . import config, prompts, replay
from .ask import ask, retrieve
from .extract import trich_xuat
from .scoring import answer_correct, field_matches, is_refusal, retrieval_hit


def _load(name: str) -> list[dict]:
    return json.loads((config.EVAL_DIR / name).read_text(encoding="utf-8"))


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "—"


def eval_retrieval(top_k: int | None = None) -> dict:
    """Chỉ đo bước tìm kiếm; không gọi Claude, không tốn tiền."""
    items = [i for i in _load("hoi-dap.json") if i["sources"]]
    misses = []
    for item in items:
        files = [h.node.metadata.get("file_name") for h in retrieve(item["question"], top_k)]
        if not retrieval_hit(files, item["sources"]):
            misses.append({"id": item["id"], "expected": item["sources"], "got": files})
    return {"top_k": top_k or config.TOP_K, "hit": len(items) - len(misses), "total": len(items), "misses": misses}


def eval_qa(client: anthropic.Anthropic | None) -> dict:
    rows, missing = [], []
    for item in _load("hoi-dap.json"):
        try:
            a = ask(item["question"], client)
        except replay.NoRecording:
            missing.append(item["id"])
            continue
        files = [s.file for s in a.sources]
        refusal = is_refusal(a.text)
        if item.get("expect_refusal"):
            correct = refusal
        else:
            correct = not refusal and answer_correct(a.text, item["must_include"])
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "answer": a.text,
            "in_scope": not item.get("expect_refusal", False),
            "retrieval_hit": retrieval_hit(files, item["sources"]),
            "correct": correct,
            "refusal_ok": refusal == bool(item.get("expect_refusal")),
            "cited": bool(a.citations),
            "citations": [asdict(c) for c in a.citations],
            "cost_usd": a.usage.cost_usd if a.usage else 0,
            "latency_ms": a.usage.latency_ms if a.usage else 0,
            "recorded_at": a.usage.replayed_from if a.usage else None,
        })
    return {"rows": rows, "missing": missing}


def eval_extract(client: anthropic.Anthropic | None) -> dict:
    rows, missing = [], []
    for item in _load("trich-xuat.json"):
        try:
            kq = trich_xuat(config.VOUCHERS_DIR / item["file"], client)
        except replay.NoRecording:
            missing.append(item["file"])
            continue
        bg = kq.bao_gia.model_dump()
        bg["so_dong"] = len(kq.bao_gia.dong)
        fields = {k: field_matches(bg.get(k), v) for k, v in item["expected"].items()}
        issues = [v.ma for v in kq.van_de]
        rows.append({
            "file": item["file"],
            "fields_ok": sum(fields.values()),
            "fields_total": len(fields),
            "wrong_fields": [k for k, ok in fields.items() if not ok],
            "issues": issues,
            "issues_ok": sorted(issues) == sorted(item["expected_issues"]),
            "cost_usd": kq.usage.cost_usd,
            "latency_ms": kq.usage.latency_ms,
            "recorded_at": kq.usage.replayed_from,
        })
    return {"rows": rows, "missing": missing}


def summary(retrieval: dict, qa: dict, extract: dict) -> str:
    q = qa["rows"]
    in_scope = [r for r in q if r["in_scope"]]
    out_scope = [r for r in q if not r["in_scope"]]
    e = extract["rows"]
    cost = sum(r["cost_usd"] for r in q + e)
    latency = sorted(r["latency_ms"] for r in q)
    p50 = latency[len(latency) // 2] if latency else 0
    lines: list[str] = []
    dates = sorted({r["recorded_at"] for r in q + e if r.get("recorded_at")})
    if dates:
        span = dates[0] if len(dates) == 1 else f"{dates[0]} đến {dates[-1]}"
        lines += [
            f"> **Kết quả phát lại.** Các câu trả lời của Claude dưới đây đã được ghi từ lần chạy thật ngày {span}; "
            "lượt đo này không gọi Claude. Phần tìm tài liệu và phần kiểm tra báo giá vẫn chạy thật. "
            "Độ trễ và chi phí là số của lần chạy thật lúc ghi.",
            "",
        ]
    n_missing = len(qa.get("missing", [])) + len(extract.get("missing", []))
    if n_missing:
        lines += [f"> {n_missing} mục chưa có bản ghi nên không được tính: {', '.join(qa.get('missing', []) + extract.get('missing', []))}.", ""]
    lines += [
        f"Model: `{config.LLM_MODEL}` · embedding: `{config.EMBED_MODEL}` · top-k: {retrieval['top_k']} · prompt: `{prompts.ANSWER.version}`, `{prompts.EXTRACT.version}`",
        "",
        "| Chỉ số | Kết quả |",
        "| --- | --- |",
        f"| Tìm đúng tài liệu nguồn (top-{retrieval['top_k']}) | {_pct(retrieval['hit'], retrieval['total'])} |",
        f"| Trả lời đúng (câu trong phạm vi) | {_pct(sum(r['correct'] for r in in_scope), len(in_scope))} |",
        f"| Từ chối đúng (câu ngoài tài liệu) | {_pct(sum(r['correct'] for r in out_scope), len(out_scope))} |",
        f"| Câu trả lời có trích dẫn | {_pct(sum(r['cited'] for r in in_scope), len(in_scope))} |",
        f"| Trích xuất báo giá: trường đúng | {_pct(sum(r['fields_ok'] for r in e), sum(r['fields_total'] for r in e))} |",
        f"| Trích xuất báo giá: phát hiện đúng vấn đề | {_pct(sum(r['issues_ok'] for r in e), len(e))} |",
        f"| Độ trễ hỏi đáp (trung vị) | {p50} ms |",
        f"| Tổng chi phí một lượt đo | ${cost:.4f} |",
    ]
    return "\n".join(lines)


def run() -> str:
    client = anthropic.Anthropic() if replay.mode() == "live" else None
    retrieval = eval_retrieval()
    qa = eval_qa(client)
    extract = eval_extract(client)
    if not qa["rows"] and not extract["rows"]:
        return (
            f"Tìm đúng tài liệu nguồn: {retrieval['hit']}/{retrieval['total']} (phần này chạy thật trên máy).\n\n"
            "Chưa đo được phần hỏi đáp và trích xuất: máy chưa có khóa API của Claude và chưa có câu trả lời ghi sẵn nào.\n"
            "Điền ANTHROPIC_API_KEY vào tệp .env rồi chạy lại lệnh này; các câu trả lời thật sẽ được ghi lại để lần sau phát lại được."
        )
    text = summary(retrieval, qa, extract)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {"at": stamp, "mode": replay.mode(), "model": config.LLM_MODEL, "embed_model": config.EMBED_MODEL,
               "prompts": [prompts.ANSWER.version, prompts.EXTRACT.version],
               "retrieval": retrieval, "qa": qa, "extract": extract}
    (config.RESULTS_DIR / f"eval-{stamp}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.RESULTS_DIR / "latest.md").write_text(f"# Kết quả đo gần nhất ({stamp})\n\n{text}\n", encoding="utf-8")
    return text
