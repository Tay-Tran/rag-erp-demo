"""Cấu trúc dữ liệu của một báo giá, dùng chung cho bước trích xuất (LLM) và bước kiểm tra (Python)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DongBaoGia(BaseModel):
    stt: int = Field(description="Số thứ tự dòng, bắt đầu từ 1")
    noi_dung: str
    hang_vi_tri: Literal["A", "B", "C"] | None = Field(description="Hạng vị trí màn hình, nếu chứng từ ghi")
    spot_giay: int | None = Field(description="Độ dài spot tính bằng giây, ví dụ 15 hoặc 30")
    so_man_hinh: int | None
    so_tuan: int | None
    so_luong: int = Field(description="Số lượng tính tiền (màn hình × tuần), đúng như chứng từ ghi")
    don_gia: int = Field(description="Đơn giá bằng đồng, số nguyên")
    thanh_tien: int = Field(description="Thành tiền bằng đồng, đúng như chứng từ ghi, kể cả khi tính sai")


class BaoGia(BaseModel):
    so_bao_gia: str
    ngay_lap: str = Field(description="Ngày lập theo dạng YYYY-MM-DD")
    khach_hang: str = Field(description="Tên khách hàng, bỏ các ghi chú trong ngoặc")
    ma_so_thue: str | None
    so_tuan_hop_dong: int | None
    dong: list[DongBaoGia]
    tam_tinh: int
    chiet_khau_phan_tram: float
    chiet_khau: int
    sau_chiet_khau: int
    vat_phan_tram: float
    vat: int
    tong_cong: int
    da_duyet_chiet_khau: bool | None = Field(description="true nếu chứng từ ghi chiết khấu đã duyệt hoặc không cần duyệt, false nếu ghi chưa duyệt, null nếu không ghi")
