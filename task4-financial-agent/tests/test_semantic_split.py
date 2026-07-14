import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_split import split_semantic


def test_semantic_split_keeps_related_sentences_together():
    groups = split_semantic(
        "2024年营业收入增长。净利润同比增长。资产负债率下降。监管机构发布新规。",
        threshold=0.2,
    )
    assert groups
    assert any("营业收入" in group and "净利润" in group for group in groups)
    assert any("监管机构" in group for group in groups)


def test_semantic_split_accepts_external_similarity():
    groups = split_semantic(
        "甲段。乙段。丙段。",
        threshold=0.5,
        similarity=lambda left, right: 1.0 if left == "甲段。" else 0.0,
    )
    assert groups == ["甲段。乙段。", "丙段。"]

