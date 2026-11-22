"""Dòng lệnh: python -m rag_erp ingest | ask "câu hỏi" | extract <tệp> | eval [--retrieval-only]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, replay


def _usage_line(u) -> str:
    base = f"{u.model} · {u.prompt_version} · {u.input_tokens}+{u.output_tokens} token · ${u.cost_usd:.5f} · {u.latency_ms} ms"
    return f"phát lại, không tốn tiền; số liệu của lần chạy thật lúc ghi: {base}" if u.replayed_from else base


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="rag_erp", description="Demo RAG trên tài liệu quy trình nội bộ (LlamaIndex + Qdrant + Claude)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="Nạp tài liệu trong data/tai-lieu vào Qdrant")
    p_ask = sub.add_parser("ask", help="Hỏi một câu")
    p_ask.add_argument("question")
    p_ext = sub.add_parser("extract", help="Trích xuất và kiểm tra một báo giá")
    p_ext.add_argument("file", type=Path)
    p_eval = sub.add_parser("eval", help="Chạy bộ đo độ chính xác")
    p_eval.add_argument("--retrieval-only", action="store_true", help="Chỉ đo bước tìm kiếm (không gọi Claude)")
    args = parser.parse_args(argv)
    try:
        return _run(args)
    finally:
        from . import store

        store.close()


def _run(args: argparse.Namespace) -> int:
    if args.cmd == "ingest":
        from .ingest import ingest

        stats = ingest()
        print(f"Đã nạp {stats['documents']} tài liệu thành {stats['chunks']} đoạn vào Qdrant ({config.QDRANT_PATH}).")

    elif args.cmd == "ask":
        from .ask import _section, ask, retrieve

        try:
            a = ask(args.question)
        except replay.NoRecording as e:
            print(replay.missing_message(e.kind))
            print("\nPhần tìm tài liệu vẫn chạy thật trên máy. Các đoạn liên quan nhất tìm được:")
            for h in retrieve(args.question):
                print(f"  - {h.node.metadata.get('file_name')} › {_section(h.node)}")
            return 1
        if a.usage and a.usage.replayed_from:
            print(replay.banner(a.usage.replayed_from, a.usage.model) + "\n")
        print(a.text)
        if a.citations:
            print("\nTrích dẫn:")
            seen = set()
            for c in a.citations:
                key = (c.file, c.cited_text)
                if key not in seen:
                    seen.add(key)
                    print(f"  - {c.file} › {c.section}: \"{c.cited_text}\"")
        if a.usage:
            print(f"\n[{_usage_line(a.usage)}]")

    elif args.cmd == "extract":
        from .extract import trich_xuat

        path = args.file if args.file.exists() else config.VOUCHERS_DIR / args.file
        try:
            kq = trich_xuat(path)
        except replay.NoRecording as e:
            print(replay.missing_message(e.kind))
            return 1
        if kq.usage.replayed_from:
            print(replay.banner(kq.usage.replayed_from, kq.usage.model) + "\n")
        print(kq.bao_gia.model_dump_json(indent=2))
        if kq.van_de:
            print("\nVấn đề phát hiện:")
            for v in kq.van_de:
                print(f"  [{'LỖI' if v.muc == 'loi' else 'CẢNH BÁO'}] {v.mo_ta}")
        else:
            print("\nKhông phát hiện vấn đề.")
        print(f"\n[{_usage_line(kq.usage)}]")

    elif args.cmd == "eval":
        from .evaluate import eval_retrieval, run

        if args.retrieval_only:
            r = eval_retrieval()
            print(f"Tìm đúng nguồn: {r['hit']}/{r['total']} (top-{r['top_k']})")
            for m in r["misses"]:
                print(f"  trượt {m['id']}: cần {m['expected']}, được {m['got']}")
        else:
            print(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
