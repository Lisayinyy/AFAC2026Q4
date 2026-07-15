from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import (EvidenceSlot, _best_excerpt, _fallback_plan,
                   _match_option_documents, _merge_review_result,
                   _review_targets, build_evidence_matrix,
                   reconcile_verdicts)
from parse import _chunk_tables, chunk_text
from retrieve import (_financial_metric_anchors, _tokenize, build_context,
                      normalize_numeric_text)
from semantic_split import split_semantic
from segment.insurance import (classify_insurance_fact, expand_parent_groups,
                               segment_insurance_chunks)
from segment.financial_report import (expand_financial_groups,
                                      segment_financial_report_chunks)
from segment.regulatory import (classify_regulatory_role,
                                expand_regulatory_groups,
                                segment_regulatory_chunks)
from compare_scored_runs import ScoredRun, pairwise_implication


def test_semantic_split_keeps_related_financial_sentences_together():
    groups = split_semantic(
        "2024年营业收入增长。营业收入同比增长20%。监管机构发布新规。",
        threshold=0.08,
    )
    assert any("营业收入增长" in group and "同比增长20%" in group for group in groups)
    assert any("监管机构" in group for group in groups)


def test_chunks_share_section_region_and_preserve_metadata():
    chunks = chunk_text(
        "第一节 财务情况\n" + "2024年净利润同比增长20%。" * 180,
        "demo", "financial_reports",
    )
    assert len(chunks) > 1
    assert {chunk["section"] for chunk in chunks} == {"第一节 财务情况"}
    assert len({chunk["region"] for chunk in chunks}) == 1


def test_numeric_normalization_handles_financial_variants():
    assert normalize_numeric_text("1,000 万元") == "1000万元"
    assert normalize_numeric_text("百分之十") == "10%"
    assert "1000" in _tokenize("1,000万元")
    assert "10%" in _tokenize("百分之十")


def test_context_and_excerpt_preserve_complete_table_rows():
    chunks = [{
        "doc_id": "demo", "chunk_id": 1, "section": "利润表", "region": "t1",
        "is_table": True,
        "text": "[表格]\n年份 | 净利润\n2023 | 10亿元\n2024 | 20亿元\n2025 | 30亿元",
    }]
    context = build_context(chunks, max_chars=105)
    assert not context.endswith("2024 | 20亿")
    excerpt = _best_excerpt(chunks[0]["text"], "2024年净利润20亿元", 55)
    assert "年份 | 净利润" in excerpt
    assert "2024 | 20亿元" in excerpt


def test_review_flip_requires_stronger_cited_evidence():
    first = {
        "verdict": "support", "confidence": 0.90,
        "evidence": ["A-E1"], "reason": "直接证据",
    }
    weak = {
        "verdict": "contradict", "confidence": 0.70,
        "evidence": ["A-R1"], "reason": "弱复核",
    }
    assert _merge_review_result(first, weak, []) is first

    strong = {
        "verdict": "contradict", "confidence": 0.96,
        "evidence": ["A-R2"], "reason": "直接相反证据",
    }
    assert _merge_review_result(first, strong, [])["verdict"] == "contradict"


def test_flagged_first_verdict_can_be_repaired_at_lower_threshold():
    first = {"verdict": "uncertain", "confidence": 0.80, "evidence": [], "reason": ""}
    review = {
        "verdict": "support", "confidence": 0.65,
        "evidence": ["B-R1"], "reason": "补齐证据",
    }
    assert _merge_review_result(first, review, ["missing_fact"])["verdict"] == "support"


def test_scored_run_pair_derives_fixed_and_regressed_counts():
    left = ScoredRun(
        "old", pathlib.Path("old.csv"), 3,
        {"q1": "A", "q2": "A", "q3": "A", "q4": "A", "q5": "A"},
    )
    right = ScoredRun(
        "new", pathlib.Path("new.csv"), 5,
        {"q1": "B", "q2": "B", "q3": "A", "q4": "A", "q5": "A"},
    )
    info = pairwise_implication(left, right)
    assert info["distance"] == 2
    assert info["scenarios"] == [{
        "old_only_correct": 0,
        "new_only_correct": 2,
        "both_wrong_disagreement": 0,
        "same_answer_wrong": 0,
    }]


def test_scored_run_pair_allows_different_answers_both_wrong():
    left = ScoredRun(
        "old", pathlib.Path("old.csv"), 2,
        {"q1": "A", "q2": "A", "q3": "A", "q4": "A"},
    )
    right = ScoredRun(
        "new", pathlib.Path("new.csv"), 1,
        {"q1": "B", "q2": "B", "q3": "A", "q4": "A"},
    )
    scenarios = pairwise_implication(left, right)["scenarios"]
    assert any(item["both_wrong_disagreement"] == 1 for item in scenarios)


def test_insurance_product_name_binds_option_to_one_document():
    profiles = {
        "2": "国寿增益宝终身寿险（万能型）条款",
        "6": "太保团体百万医疗保险（2022版）条款",
        "11": "平安家庭财产保险条款",
        "3": "众安在线财产保险股份有限公司 个人急性白血病复发医疗保险条款",
    }
    assert _match_option_documents("太保团体百万医疗", profiles) == ["6"]
    assert _match_option_documents("国寿增益宝", profiles) == ["2"]
    assert _match_option_documents("平安家财险", profiles) == ["11"]
    assert _match_option_documents("众安白血病医疗险", profiles) == ["3"]
    assert _match_option_documents(
        "国寿增益宝与太保团体百万医疗均适用", profiles
    ) == []
    assert _match_option_documents("第二份文档的处理规则", profiles) == []


def test_fixed_benefit_policy_supports_absence_of_drug_expense_coverage():
    q = {
        "domain": "insurance",
        "question": "关于特定药品费用，哪些说法符合条款？",
        "options": {"B": "平安安佑福重疾险不涵盖院外特定药品费用"},
    }
    verdicts = {"options": {"B": {
        "verdict": "contradict", "confidence": 0.8,
        "reason": "未找到直接写明不涵盖的条款",
    }}}
    changes = reconcile_verdicts(
        q, verdicts,
        {"4": "平安安佑福重大疾病保险，仅提供重大疾病及身故保障"},
        {"4": {"特定药品": 0, "院外": 0}},
    )
    assert verdicts["options"]["B"]["verdict"] == "support"
    assert changes[0]["rule"] == "fixed_benefit_policy_lacks_drug_expense_coverage"


def test_product_without_suspension_clause_cannot_borrow_other_policy_evidence():
    q = {
        "domain": "insurance",
        "question": "哪些产品明确宽限期后效力中止且不承担保险责任？",
        "options": {"B": "太保团体百万医疗"},
    }
    verdicts = {"options": {"B": {
        "verdict": "support", "confidence": 0.9,
        "reason": "引用了另一份寿险合同的效力中止条款",
    }}}
    reconcile_verdicts(
        q, verdicts,
        {"2": "国寿增益宝终身寿险", "6": "太保团体百万医疗保险"},
        {"2": {"宽限期": 7, "效力中止": 9},
         "6": {"宽限期": 0, "效力中止": 0}},
    )
    assert verdicts["options"]["B"]["verdict"] == "contradict"


def test_research_exact_fact_is_not_rejected_only_for_omitted_region():
    q = {
        "domain": "research",
        "question": "关于银保渠道发展历程，哪些说法正确？",
        "options": {"B": "2005年至2018年银保渠道复合增速9.9%"},
    }
    verdicts = {"options": {"B": {
        "verdict": "contradict", "confidence": 0.8,
        "reason": "证据明确记载该期间复合增速9.9%，但选项未限定地区。",
    }}}
    reconcile_verdicts(q, verdicts)
    assert verdicts["options"]["B"]["verdict"] == "support"


def test_insurance_segmentation_indexes_atoms_and_expands_parent_clause():
    chunks = [{
        "doc_id": "6", "domain": "insurance", "chunk_id": 97,
        "section": "2.4 保险责任", "region": "2.4", "is_table": False,
        "text": (
            "2.4 保险责任 在本合同有效期内，本公司承担一般医疗保险金责任。"
            "投保人须投保一般医疗保险金作为必选责任；"
            "轻症疾病医疗保险金属于可选责任。"
        ),
    }]
    atoms = segment_insurance_chunks(chunks)
    assert len(atoms) >= 2
    assert all(atom["chunk_type"] == "atomic_fact" for atom in atoms)
    assert all(atom["subject"] for atom in atoms)
    assert any("必选责任" in atom["text"] for atom in atoms)

    matched = next(atom for atom in atoms if "必选责任" in atom["text"])
    expanded = expand_parent_groups([matched])
    assert expanded[0]["atomic_text"] == matched["text"]
    assert "一般医疗保险金" in expanded[0]["text"]
    assert expanded[0]["chunk_type"] == "evidence_group"


def test_insurance_fact_typing_distinguishes_state_and_exclusion():
    assert classify_insurance_fact("宽限期满次日零时起效力中止") == "contract_state"
    assert classify_insurance_fact("本公司对院外购药费用不承担保险责任") == "exclusion"


def test_explicit_missing_insurer_does_not_borrow_other_product(monkeypatch):
    class Retriever:
        def retrieve_for_option(self, *args, **kwargs):
            raise AssertionError("missing insurer should short-circuit retrieval")

    q = {
        "domain": "insurance", "question": "哪些说法符合条款？",
        "options": {"D": "太保团体百万医疗涵盖所有院外药品费用"},
        "doc_ids": ["3", "4", "5"],
    }
    matrix = build_evidence_matrix(
        Retriever(), q, _fallback_plan(q),
        document_profiles={
            "3": "众安白血病医疗保险", "4": "平安安佑福重大疾病保险",
            "5": "平安e生保医疗保险",
        },
    )
    assert matrix["D"].chunks == []
    assert matrix["D"].doc_coverage == []


def test_financial_table_row_inherits_year_unit_scope_and_header():
    chunks = [{
        "doc_id": "annual_demo_2024", "domain": "financial_reports",
        "chunk_id": 30, "is_table": True, "page": 88,
        "section": "主要会计数据", "table_context": "合并口径 单位：万元",
        "text": (
            "[表格]\n项目 | 2024年 | 2023年\n"
            "营业收入 | 12,000 | 10,000\n"
            "归属于上市公司股东的净利润 | 1,200 | 900"
        ),
    }]
    records = segment_financial_report_chunks(chunks)
    revenue = next(r for r in records if "营业收入" in r["text"])
    assert revenue["fact_type"] == "financial_metric"
    assert revenue["qualifiers"]["years"] == ["2024年", "2023年"]
    assert revenue["qualifiers"]["unit"] == "万元"
    assert revenue["qualifiers"]["scope"] == "合并口径"
    assert revenue["qualifiers"]["values"] == {
        "2024年": "12,000", "2023年": "10,000",
    }
    expanded = expand_financial_groups([revenue])
    assert "项目 | 2024年 | 2023年" in expanded[0]["text"]
    assert "营业收入 | 12,000 | 10,000" in expanded[0]["text"]


def test_long_table_chunks_repeat_header_and_keep_page_context():
    rows = ["项目 | 2024年 | 2023年"] + [
        f"指标{i} | {1000 + i} | {900 + i}" for i in range(150)
    ]
    chunks = _chunk_tables([{
        "markdown": "\n".join(rows), "page": 12, "table_no": 3,
        "context": "主要会计数据 单位：百万元",
    }], "annual_demo", "financial_reports", 7)
    assert len(chunks) > 1
    assert all("项目 | 2024年 | 2023年" in c["text"] for c in chunks)
    assert all(c["page"] == 12 for c in chunks)
    assert all(c["table_context"] == "主要会计数据 单位：百万元" for c in chunks)


def test_financial_metric_aliases_route_to_disclosed_row_name():
    assert "营业收入" in _financial_metric_anchors("营业总收入同比增长")
    assert "研发投入占营业收入比例" in _financial_metric_anchors(
        "研发投入占营业收入的比例有所提升"
    )
    assert "归属于上市公司股东的净利润" in _financial_metric_anchors(
        "归母净利润下降"
    )


def test_regulatory_items_expand_to_complete_parent_article():
    chunks = [
        {"doc_id": "strict_demo", "domain": "regulatory", "chunk_id": 0,
         "section": "第一章 总则", "text": "第一章 总则", "is_table": False},
        {"doc_id": "strict_demo", "domain": "regulatory", "chunk_id": 1,
         "section": "第二条", "text": "第二条 金融机构应当履行下列义务：", "is_table": False},
        {"doc_id": "strict_demo", "domain": "regulatory", "chunk_id": 2,
         "section": "（一）", "text": "（一）核实客户身份；", "is_table": False},
        {"doc_id": "strict_demo", "domain": "regulatory", "chunk_id": 3,
         "section": "（二）", "text": "（二）保存交易记录。", "is_table": False},
        {"doc_id": "strict_demo", "domain": "regulatory", "chunk_id": 4,
         "section": "第三条", "text": "第三条 本办法自公布之日起施行。", "is_table": False},
    ]
    records = segment_regulatory_chunks(chunks)
    identity = next(r for r in records if "核实客户身份" in r["text"])
    expanded = expand_regulatory_groups([identity])
    assert identity["qualifiers"]["article"] == "第二条"
    assert identity["fact_type"] == "formal_rule"
    assert "第二条 金融机构应当履行下列义务" in expanded[0]["text"]
    assert "（二）保存交易记录" in expanded[0]["text"]
    assert "第三条" not in expanded[0]["text"]


def test_regulatory_role_separates_argument_fact_and_finding():
    assert classify_regulatory_role("当事人申辩称相关行为不违法") == "party_argument"
    assert classify_regulatory_role("经查明，该公司未按期披露") == "case_fact"
    assert classify_regulatory_role("经复核，我会认为申辩理由不成立") == "authority_finding"


def test_non_article_enforcement_chunks_do_not_form_one_giant_parent():
    chunks = [
        {"doc_id": "case_demo", "domain": "regulatory", "chunk_id": i,
         "section": f"部分{i}", "text": text, "is_table": False}
        for i, text in enumerate([
            "经查明，该公司未按期披露年度报告。",
            "当事人申辩称不存在主观故意。",
            "经复核，我会认为申辩理由不成立。",
        ])
    ]
    records = segment_regulatory_chunks(chunks)
    groups = {record["group_id"] for record in records}
    assert len(groups) == 3
    finding = next(r for r in records if "经复核" in r["text"])
    assert len(expand_regulatory_groups([finding])[0]["text"]) < 100


def test_review_targets_skip_confident_multi_choice_options():
    q = {"answer_format": "multi", "options": {k: k for k in "ABCD"}}
    verdicts = {"options": {
        "A": {"verdict": "support", "confidence": 0.91},
        "B": {"verdict": "support", "confidence": 0.88},
        "C": {"verdict": "contradict", "confidence": 0.90},
        "D": {"verdict": "contradict", "confidence": 0.86},
    }}
    matrix = {
        k: EvidenceSlot(k, k, [], coverage=0.9) for k in "ABCD"
    }
    assert _review_targets(
        q, verdicts, {k: [] for k in "ABCD"}, matrix, {"options": {}}
    ) == []


def test_review_targets_limit_low_selection_appeal_to_one_option():
    q = {"answer_format": "multi", "options": {k: k for k in "ABCD"}}
    verdicts = {"options": {
        "A": {"verdict": "support", "confidence": 0.90},
        "B": {"verdict": "contradict", "confidence": 0.90},
        "C": {"verdict": "contradict", "confidence": 0.90},
        "D": {"verdict": "contradict", "confidence": 0.90},
    }}
    matrix = {
        "A": EvidenceSlot("A", "A", [], coverage=0.8),
        "B": EvidenceSlot("B", "B", [], coverage=0.62),
        "C": EvidenceSlot("C", "C", [], coverage=0.91),
        "D": EvidenceSlot("D", "D", [], coverage=0.70),
    }
    assert _review_targets(
        q, verdicts, {k: [] for k in "ABCD"}, matrix, {"options": {}}
    ) == ["C"]
