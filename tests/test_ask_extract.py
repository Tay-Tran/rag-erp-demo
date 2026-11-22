"""Luồng hỏi đáp, trích xuất, ghi và phát lại với client Claude giả lập (không gọi API, không cần Qdrant)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from anthropic.types import Message

from rag_erp import ask as ask_mod
from rag_erp import config, extract, replay, usage
from rag_erp.schema import BaoGia, DongBaoGia


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(response):
    return SimpleNamespace(messages=FakeMessages(response))


class ParsedFake(Message):
    parsed_output: BaoGia | None = None


def message(text, citations=(), cls=Message, **extra):
    return cls.model_validate({
        "id": "msg_test", "type": "message", "role": "assistant", "model": config.LLM_MODEL,
        "content": [{"type": "text", "text": text, "citations": list(citations) or None}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 50}, **extra,
    })


def node(text, file, header):
    return SimpleNamespace(node=SimpleNamespace(metadata={"file_name": file, "header_path": header}, get_content=lambda: text), score=0.8)


HITS = [node("Chiết khấu trên 15% phải được Trưởng phòng Kinh doanh duyệt.", "bao-gia.md", "/Quy trình/Chiết khấu/"),
        node("Đơn hàng trên 200 triệu do Tổng Giám đốc duyệt.", "mua-hang.md", "/Quy trình/")]
CITE = {"type": "char_location", "cited_text": " Chiết khấu trên 15% phải được Trưởng phòng Kinh doanh duyệt. ",
        "document_index": 0, "document_title": "bao-gia.md", "start_char_index": 0, "end_char_index": 60}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    monkeypatch.setattr(config, "REPLAY_DIR", tmp_path / "replay")
    monkeypatch.setattr(ask_mod, "retrieve", lambda q, k=None: HITS)
    monkeypatch.setenv("RAG_MODE", "live")
    return tmp_path


def quote():
    return BaoGia(
        so_bao_gia="BG-2026-0157", ngay_lap="2026-11-10", khach_hang="Công ty CP Thời trang Lam", ma_so_thue="0109876543",
        so_tuan_hop_dong=8,
        dong=[DongBaoGia(stt=1, noi_dung="a", hang_vi_tri="A", spot_giay=30, so_man_hinh=5, so_tuan=8, so_luong=40, don_gia=3_200_000, thanh_tien=128_000_000),
              DongBaoGia(stt=2, noi_dung="b", hang_vi_tri="C", spot_giay=30, so_man_hinh=30, so_tuan=8, so_luong=240, don_gia=1_000_000, thanh_tien=204_000_000)],
        tam_tinh=332_000_000, chiet_khau_phan_tram=20, chiet_khau=66_400_000, sau_chiet_khau=265_600_000,
        vat_phan_tram=10, vat=26_560_000, tong_cong=292_160_000, da_duyet_chiet_khau=False,
    )


def test_ask_sends_documents_with_citations_and_maps_them_back(isolated):
    client = fake_client(message("Cần Trưởng phòng Kinh doanh duyệt.", [CITE]))

    a = ask_mod.ask("Chiết khấu 20% ai duyệt?", client)

    sent = client.messages.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in sent] == ["document", "document", "text"]
    assert sent[0]["citations"] == {"enabled": True}
    assert sent[0]["title"] == "bao-gia.md › Quy trình › Chiết khấu"
    assert sent[1]["title"] == "mua-hang.md › Quy trình"
    assert a.text == "Cần Trưởng phòng Kinh doanh duyệt."
    assert a.citations[0].file == "bao-gia.md"
    assert a.citations[0].cited_text.startswith("Chiết khấu")
    assert a.usage.cost_usd == usage.cost_usd(config.LLM_MODEL, 1000, 50)
    assert a.usage.replayed_from is None
    assert (isolated / "usage.jsonl").read_text(encoding="utf-8").count("answer-v1") == 1


def test_section_adds_the_chunk_own_heading():
    n = SimpleNamespace(metadata={"header_path": "/Quy định nghỉ phép/"}, get_content=lambda: "## Đăng ký nghỉ\n\n- Nghỉ 1 ngày ...")
    assert ask_mod._section(n) == "Quy định nghỉ phép › Đăng ký nghỉ"


def test_live_answer_is_recorded_then_replayed_without_a_client(isolated, monkeypatch):
    ask_mod.ask("Chiết khấu 20% ai duyệt?", fake_client(message("Cần Trưởng phòng Kinh doanh duyệt.", [CITE])))

    monkeypatch.setenv("RAG_MODE", "replay")
    a = ask_mod.ask("Chiết khấu 20% ai duyệt?")

    assert a.text == "Cần Trưởng phòng Kinh doanh duyệt."
    assert a.citations[0].file == "bao-gia.md"
    assert a.usage.replayed_from is not None
    # Phát lại không ghi thêm vào nhật ký chi phí.
    assert (isolated / "usage.jsonl").read_text(encoding="utf-8").count("answer-v1") == 1


def test_replay_never_answers_a_different_question(isolated, monkeypatch):
    ask_mod.ask("Chiết khấu 20% ai duyệt?", fake_client(message("Cần Trưởng phòng Kinh doanh duyệt.")))
    monkeypatch.setenv("RAG_MODE", "replay")

    with pytest.raises(replay.NoRecording):
        ask_mod.ask("Phép năm bao nhiêu ngày?")


def test_replay_breaks_when_the_retrieved_documents_change(isolated, monkeypatch):
    ask_mod.ask("Chiết khấu 20% ai duyệt?", fake_client(message("Cần Trưởng phòng Kinh doanh duyệt.")))
    monkeypatch.setenv("RAG_MODE", "replay")
    monkeypatch.setattr(ask_mod, "retrieve", lambda q, k=None: HITS[:1])

    with pytest.raises(replay.NoRecording):
        ask_mod.ask("Chiết khấu 20% ai duyệt?")


def test_messages_shown_to_people_say_it_is_recorded(isolated):
    ask_mod.ask("Chiết khấu 20% ai duyệt?", fake_client(message("Cần Trưởng phòng Kinh doanh duyệt.")))

    text = replay.banner("2026-11-25", "claude-haiku-5-5")
    assert "đã ghi lại từ một lần chạy thật" in text
    assert "25/11/2026" in text
    assert "không gọi Claude" in text
    missing = replay.missing_message("answer")
    assert "Chưa có câu trả lời ghi sẵn" in missing
    assert "Chiết khấu 20% ai duyệt?" in missing


def test_missing_message_without_any_recording(isolated):
    assert "Hiện chưa có bản ghi nào" in replay.missing_message("answer")


def test_extract_runs_checks_and_replays_from_the_recorded_json(isolated, monkeypatch):
    bg = quote()
    client = fake_client(message(bg.model_dump_json(), cls=ParsedFake, parsed_output=bg.model_dump()))
    path = Path(config.VOUCHERS_DIR / "BG-2026-0157.txt")

    live = extract.trich_xuat(path, client)
    monkeypatch.setenv("RAG_MODE", "replay")
    replayed = extract.trich_xuat(path)

    assert client.messages.calls[0]["output_format"] is BaoGia
    expected = ["thanh_tien_dong_2", "chiet_khau_vuot_bac", "chiet_khau_chua_duyet"]
    assert [v.ma for v in live.van_de] == expected
    assert [v.ma for v in replayed.van_de] == expected
    assert replayed.bao_gia == bg
    assert replayed.usage.replayed_from is not None
