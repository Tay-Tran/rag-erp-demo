"""Kiểm tra luồng hỏi đáp và trích xuất với client Claude giả lập (không gọi API, không cần Qdrant)."""

from pathlib import Path
from types import SimpleNamespace

from rag_erp import ask as ask_mod
from rag_erp import config, extract, usage
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


def node(text, file, header):
    return SimpleNamespace(node=SimpleNamespace(metadata={"file_name": file, "header_path": header}, get_content=lambda: text), score=0.8)


def test_ask_sends_documents_with_citations_and_maps_them_back(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    hits = [node("Chiết khấu trên 15% phải được Trưởng phòng Kinh doanh duyệt.", "bao-gia.md", "/Quy trình/Chiết khấu/"),
            node("Đơn hàng trên 200 triệu do Tổng Giám đốc duyệt.", "mua-hang.md", "/Quy trình/")]
    monkeypatch.setattr(ask_mod, "retrieve", lambda q, k=None: hits)
    response = SimpleNamespace(
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1000, output_tokens=50),
        content=[SimpleNamespace(type="text", text="Cần Trưởng phòng Kinh doanh duyệt.",
                                 citations=[SimpleNamespace(document_index=0, cited_text=" Chiết khấu trên 15% phải được Trưởng phòng Kinh doanh duyệt. ")])],
    )
    client = fake_client(response)

    a = ask_mod.ask("Chiết khấu 20% ai duyệt?", client)

    sent = client.messages.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in sent] == ["document", "document", "text"]
    assert sent[0]["citations"] == {"enabled": True}
    assert sent[0]["title"] == "bao-gia.md › Quy trình › Chiết khấu"
    assert a.text == "Cần Trưởng phòng Kinh doanh duyệt."
    assert a.citations[0].file == "bao-gia.md"
    assert a.citations[0].cited_text.startswith("Chiết khấu")
    assert a.usage.cost_usd == usage.cost_usd(config.LLM_MODEL, 1000, 50)
    assert (tmp_path / "usage.jsonl").read_text(encoding="utf-8").count("answer-v1") == 1


def test_extract_runs_checks_on_parsed_output(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "USAGE_LOG", tmp_path / "usage.jsonl")
    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    bg = BaoGia(
        so_bao_gia="BG-2026-0157", ngay_lap="2026-11-10", khach_hang="Công ty CP Thời trang Lam", ma_so_thue="0109876543",
        so_tuan_hop_dong=8,
        dong=[DongBaoGia(stt=1, noi_dung="a", hang_vi_tri="A", spot_giay=30, so_man_hinh=5, so_tuan=8, so_luong=40, don_gia=3_200_000, thanh_tien=128_000_000),
              DongBaoGia(stt=2, noi_dung="b", hang_vi_tri="C", spot_giay=30, so_man_hinh=30, so_tuan=8, so_luong=240, don_gia=1_000_000, thanh_tien=204_000_000)],
        tam_tinh=332_000_000, chiet_khau_phan_tram=20, chiet_khau=66_400_000, sau_chiet_khau=265_600_000,
        vat_phan_tram=10, vat=26_560_000, tong_cong=292_160_000, da_duyet_chiet_khau=False,
    )
    response = SimpleNamespace(stop_reason="end_turn", parsed_output=bg, usage=SimpleNamespace(input_tokens=900, output_tokens=400))
    client = fake_client(response)

    kq = extract.trich_xuat(Path(config.VOUCHERS_DIR / "BG-2026-0157.txt"), client)

    assert client.messages.calls[0]["output_format"] is BaoGia
    assert [v.ma for v in kq.van_de] == ["thanh_tien_dong_2", "chiet_khau_vuot_bac", "chiet_khau_chua_duyet"]
