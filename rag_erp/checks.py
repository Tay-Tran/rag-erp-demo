"""Kiểm tra báo giá bằng quy tắc cố định, không dùng LLM.

LLM chỉ đọc chứng từ thành dữ liệu có cấu trúc; mọi phép tính và đối chiếu quy định nằm ở đây,
để kết quả lặp lại được và kiểm thử được. Các con số lấy từ
data/tai-lieu/quy-trinh-bao-gia-quang-cao.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .schema import BaoGia

BANG_GIA: dict[tuple[str, int], int] = {
    ("A", 15): 1_800_000,
    ("A", 30): 3_200_000,
    ("B", 15): 1_100_000,
    ("B", 30): 1_900_000,
    ("C", 15): 600_000,
    ("C", 30): 1_000_000,
}
VAT_PHAN_TRAM = 10.0
NGUONG_DUYET_TRUONG_PHONG = 15.0
NGUONG_DUYET_GIAM_DOC = 25.0


def bac_chiet_khau(so_tuan: int | None) -> float:
    """Chiết khấu chuẩn theo thời lượng hợp đồng."""
    if so_tuan is None or so_tuan < 4:
        return 0.0
    if so_tuan < 12:
        return 5.0
    if so_tuan < 26:
        return 10.0
    return 15.0


@dataclass(frozen=True)
class VanDe:
    ma: str
    muc: Literal["loi", "canh_bao"]
    mo_ta: str


def _tien(n: int) -> str:
    return f"{n:,}".replace(",", ".") + " đ"


def kiem_tra(bg: BaoGia) -> list[VanDe]:
    van_de: list[VanDe] = []

    for d in bg.dong:
        if d.so_man_hinh is not None and d.so_tuan is not None and d.so_man_hinh * d.so_tuan != d.so_luong:
            van_de.append(VanDe(f"so_luong_dong_{d.stt}", "loi", f"Dòng {d.stt}: {d.so_man_hinh} màn hình × {d.so_tuan} tuần = {d.so_man_hinh * d.so_tuan}, chứng từ ghi {d.so_luong}."))
        dung = d.so_luong * d.don_gia
        if dung != d.thanh_tien:
            van_de.append(VanDe(f"thanh_tien_dong_{d.stt}", "loi", f"Dòng {d.stt}: {d.so_luong} × {_tien(d.don_gia)} = {_tien(dung)}, chứng từ ghi {_tien(d.thanh_tien)}."))
        if d.hang_vi_tri and d.spot_giay:
            gia = BANG_GIA.get((d.hang_vi_tri, d.spot_giay))
            if gia is not None and gia != d.don_gia:
                van_de.append(VanDe(f"don_gia_dong_{d.stt}", "canh_bao", f"Dòng {d.stt}: bảng giá hạng {d.hang_vi_tri}, spot {d.spot_giay}s là {_tien(gia)}, chứng từ ghi {_tien(d.don_gia)}."))

    tong_dong = sum(d.thanh_tien for d in bg.dong)
    if tong_dong != bg.tam_tinh:
        van_de.append(VanDe("tam_tinh", "loi", f"Cộng thành tiền các dòng được {_tien(tong_dong)}, tạm tính ghi {_tien(bg.tam_tinh)}."))

    ck = round(bg.tam_tinh * bg.chiet_khau_phan_tram / 100)
    if ck != bg.chiet_khau:
        van_de.append(VanDe("chiet_khau_tien", "loi", f"Chiết khấu {bg.chiet_khau_phan_tram:g}% của {_tien(bg.tam_tinh)} là {_tien(ck)}, chứng từ ghi {_tien(bg.chiet_khau)}."))
    if bg.tam_tinh - bg.chiet_khau != bg.sau_chiet_khau:
        van_de.append(VanDe("sau_chiet_khau", "loi", f"Tạm tính trừ chiết khấu là {_tien(bg.tam_tinh - bg.chiet_khau)}, chứng từ ghi {_tien(bg.sau_chiet_khau)}."))

    if bg.vat_phan_tram != VAT_PHAN_TRAM:
        van_de.append(VanDe("thue_suat_vat", "loi", f"Quy định áp dụng VAT {VAT_PHAN_TRAM:g}%, chứng từ ghi {bg.vat_phan_tram:g}%."))
    vat = round(bg.sau_chiet_khau * bg.vat_phan_tram / 100)
    if vat != bg.vat:
        van_de.append(VanDe("tien_vat", "loi", f"VAT {bg.vat_phan_tram:g}% của {_tien(bg.sau_chiet_khau)} là {_tien(vat)}, chứng từ ghi {_tien(bg.vat)}."))
    if bg.sau_chiet_khau + bg.vat != bg.tong_cong:
        van_de.append(VanDe("tong_cong", "loi", f"Sau chiết khấu cộng VAT là {_tien(bg.sau_chiet_khau + bg.vat)}, tổng cộng ghi {_tien(bg.tong_cong)}."))

    chuan = bac_chiet_khau(bg.so_tuan_hop_dong)
    if bg.chiet_khau_phan_tram > chuan:
        van_de.append(VanDe("chiet_khau_vuot_bac", "canh_bao", f"Hợp đồng {bg.so_tuan_hop_dong} tuần có chiết khấu chuẩn {chuan:g}%, báo giá ghi {bg.chiet_khau_phan_tram:g}%."))
    if bg.chiet_khau_phan_tram > NGUONG_DUYET_TRUONG_PHONG and bg.da_duyet_chiet_khau is not True:
        nguoi = "Giám đốc Kinh doanh" if bg.chiet_khau_phan_tram > NGUONG_DUYET_GIAM_DOC else "Trưởng phòng Kinh doanh"
        van_de.append(VanDe("chiet_khau_chua_duyet", "loi", f"Chiết khấu {bg.chiet_khau_phan_tram:g}% phải được {nguoi} duyệt trên ERP trước khi gửi khách hàng."))

    return van_de
