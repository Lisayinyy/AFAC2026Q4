import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_split import split_semantic
from answer import _option_context, _option_has_evidence
from parse import chunk_text
from retrieve import normalize_numeric_text, _tokenize


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


def test_option_context_extracts_evidence_sentences_only():
    chunks = [{
        "doc_id": "demo", "chunk_id": 0, "region": 0, "section": "财务",
        "is_table": False,
        "text": "这是无关背景。2024年净利润为20亿元。后续还有无关说明。",
    }]
    result = _option_context(chunks, "2024年净利润为20亿元", max_chars=80)
    assert "净利润" in result
    assert "无关背景" in result
    assert len(result) <= 80


def test_chunks_in_one_section_share_region():
    chunks = chunk_text(
        "第一节 财务情况\n" + "2024年净利润增长。" * 180,
        "demo", "financial_reports",
    )
    assert len(chunks) > 1
    assert len({c["region"] for c in chunks}) == 1


def test_numeric_normalization_handles_financial_variants():
    assert normalize_numeric_text("1,000 万元") == "1000万元"
    assert normalize_numeric_text("百分之十") == "10%"
    assert "1000" in _tokenize("1,000万元")
    assert "10%" in _tokenize("百分之十")


def test_generic_word_does_not_suppress_narrow_retrieval():
    chunks = [{
        "doc_id": "demo", "chunk_id": 0, "region": 0, "section": "",
        "is_table": False, "text": "公司情况和相关信息说明。",
    }]
    assert not _option_has_evidence(chunks, "公司相关情况中的净利润20亿元")


def test_two_indicators_or_indicator_number_count_as_evidence():
    chunks = [{
        "doc_id": "demo", "chunk_id": 0, "region": 0, "section": "",
        "is_table": False, "text": "净利润为20亿元，营业收入同比增长。",
    }]
    assert _option_has_evidence(chunks, "净利润20亿元")
    assert _option_has_evidence(chunks, "净利润和营业收入增长")
