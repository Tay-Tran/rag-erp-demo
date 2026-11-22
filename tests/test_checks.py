from rag_erp.checks import bac_chiet_khau, kiem_tra
from rag_erp.schema import BaoGia, DongBaoGia


def dong(stt, hang, spot, mh, tuan, don_gia, thanh_tien=None):
    sl = mh * tuan
    return DongBaoGia(
        stt=stt, noi_dung="x", hang_vi_tri=hang, spot_giay=spot, so_man_hinh=mh, so_tuan=tuan,
        so_luong=sl, don_gia=don_gia, thanh_tien=sl * don_gia if thanh_tien is None else thanh_tien,
    )


def bao_gia(dongs, pct, tuan, duyet=None, tam_tinh=None):
    tt = sum(d.thanh_tien for d in dongs) if tam_tinh is None else tam_tinh
    ck = round(tt * pct / 100)
    sau = tt - ck
    vat = round(sau * 0.1)
    return BaoGia(
        so_bao_gia="BG-2026-0001", ngay_lap="2026-11-01", khach_hang="K", ma_so_thue=None,
        so_tuan_hop_dong=tuan, dong=dongs, tam_tinh=tt, chiet_khau_phan_tram=pct, chiet_khau=ck,
        sau_chiet_khau=sau, vat_phan_tram=10, vat=vat, tong_cong=sau + vat, da_duyet_chiet_khau=duyet,
    )


def ma(bg):
    return [v.ma for v in kiem_tra(bg)]


def test_bac_chiet_khau():
    assert [bac_chiet_khau(t) for t in (None, 3, 4, 11, 12, 25, 26, 52)] == [0, 0, 5, 5, 10, 10, 15, 15]


def test_bao_gia_dung_khong_co_van_de():
    bg = bao_gia([dong(1, "A", 15, 10, 12, 1_800_000), dong(2, "B", 15, 20, 12, 1_100_000)], 10, 12)
    assert ma(bg) == []


def test_bao_gia_mau_0157():
    # Giống data/chung-tu/BG-2026-0157.txt: dòng 2 ghi sai thành tiền, chiết khấu 20% chưa duyệt.
    bg = bao_gia([dong(1, "A", 30, 5, 8, 3_200_000), dong(2, "C", 30, 30, 8, 1_000_000, thanh_tien=204_000_000)], 20, 8, duyet=False)
    assert ma(bg) == ["thanh_tien_dong_2", "chiet_khau_vuot_bac", "chiet_khau_chua_duyet"]


def test_don_gia_khac_bang_gia():
    bg = bao_gia([dong(1, "B", 30, 1, 4, 2_000_000)], 5, 4)
    assert ma(bg) == ["don_gia_dong_1"]


def test_tong_sai():
    bg = bao_gia([dong(1, "C", 15, 2, 4, 600_000)], 5, 4)
    bg.tong_cong += 1000
    assert ma(bg) == ["tong_cong"]


def test_chiet_khau_tren_25_can_giam_doc():
    bg = bao_gia([dong(1, "C", 15, 2, 26, 600_000)], 30, 26, duyet=False)
    van_de = {v.ma: v for v in kiem_tra(bg)}
    assert "Giám đốc Kinh doanh" in van_de["chiet_khau_chua_duyet"].mo_ta


def test_chiet_khau_da_duyet_khong_bao_loi_duyet():
    bg = bao_gia([dong(1, "C", 15, 2, 26, 600_000)], 20, 26, duyet=True)
    assert ma(bg) == ["chiet_khau_vuot_bac"]
