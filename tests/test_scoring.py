from rag_erp.prompts import REFUSAL
from rag_erp.scoring import answer_correct, field_matches, is_refusal, retrieval_hit


def test_answer_correct_needs_every_group():
    assert answer_correct("Chiết khấu 10%, đặt cọc 30% giá trị.", [["10%"], ["30%"]])
    assert not answer_correct("Chiết khấu 10%.", [["10%"], ["30%"]])


def test_answer_correct_accepts_alternatives_and_case():
    assert answer_correct("Cần GIÁM ĐỐC TÀI CHÍNH duyệt.", [["giám đốc tài chính"]])
    assert answer_correct("Tối đa 800 nghìn đồng mỗi đêm.", [["800.000", "800 nghìn"]])


def test_is_refusal():
    assert is_refusal(REFUSAL)
    assert not is_refusal("Phép năm là 12 ngày.")


def test_retrieval_hit_requires_all_sources():
    assert retrieval_hit(["a.md", "b.md"], ["a.md"])
    assert not retrieval_hit(["a.md"], ["a.md", "b.md"])
    assert retrieval_hit(["a.md"], [])


def test_field_matches():
    assert field_matches("Công ty TNHH Cà phê Hạt Mộc (khách hàng giả lập)", "Công ty TNHH Cà phê Hạt Mộc")
    assert field_matches(10.0, 10)
    assert not field_matches(475200001, 475200000)
