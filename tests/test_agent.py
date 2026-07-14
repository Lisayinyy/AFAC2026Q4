from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import (_final_answer, _json_from_text, execute_calculations,
                   reconcile_verdicts)
from run_agent import _use_baseline_tf


class AgentLocalTests(unittest.TestCase):
    def test_json_extraction_from_fence(self):
        value = _json_from_text('```json\n{"options":{"A":{}}}\n```')
        self.assertIn("A", value["options"])

    def test_whitelisted_calculations(self):
        memory = {
            "options": {
                "A": {"calculation": {"op": "growth_rate", "operands": [120, 100]}},
                "B": {"calculation": {"op": "ratio", "operands": [20, 80]}},
                "C": {"calculation": {
                    "op": "compare", "operands": [2, 3], "operator": "<"
                }},
                "D": {"calculation": {"op": "difference", "operands": [10, 4]}},
            }
        }
        execute_calculations(memory)
        self.assertEqual(memory["options"]["A"]["calculation"]["result"], 20.0)
        self.assertEqual(memory["options"]["B"]["calculation"]["result"], 0.25)
        self.assertIs(memory["options"]["C"]["calculation"]["result"], True)
        self.assertEqual(memory["options"]["D"]["calculation"]["result"], 6.0)

    def test_single_fallback_uses_confidence_not_fixed_a(self):
        q = {"answer_format": "mcq", "options": {x: x for x in "ABCD"}}
        verdicts = {
            "options": {
                x: {"verdict": "contradict", "confidence": confidence}
                for x, confidence in zip("ABCD", [0.1, 0.2, 0.3, 0.9])
            }
        }
        self.assertEqual(_final_answer(q, verdicts), "D")

    def test_single_fallback_prefers_uncertain_over_contradiction(self):
        q = {"answer_format": "mcq", "options": {x: x for x in "ABCD"}}
        verdicts = {"options": {
            "A": {"verdict": "uncertain", "confidence": 0.4},
            "B": {"verdict": "contradict", "confidence": 1.0},
            "C": {"verdict": "contradict", "confidence": 1.0},
            "D": {"verdict": "contradict", "confidence": 1.0},
        }}
        self.assertEqual(_final_answer(q, verdicts), "A")

    def test_reconcile_reason_label_conflict(self):
        q = {
            "question": "以下结论成立的是？",
            "options": {"A": "净利润降幅超过15%"},
        }
        verdicts = {"options": {"A": {
            "verdict": "contradict", "confidence": 0.9,
            "reason": "计算降幅18.97%超过15%，则应支持该选项。",
        }}}
        changes = reconcile_verdicts(q, verdicts)
        self.assertEqual(verdicts["options"]["A"]["verdict"], "support")
        self.assertEqual(changes[0]["rule"], "reason_explicitly_supports_claim")

    def test_reconcile_formula_filter_and_per_share_unit(self):
        q = {
            "question": "哪些产品明确给出了具体的计算方法或公式？",
            "options": {
                "A": "甲产品（保单上载明，未给公式）",
                "B": "乙公司每股派发45.53元",
            },
        }
        verdicts = {"options": {
            "A": {"verdict": "support", "confidence": 0.95, "reason": "描述属实"},
            "B": {"verdict": "support", "confidence": 0.95,
                  "reason": "证据为每10股派发45.53元"},
        }}
        changes = reconcile_verdicts(q, verdicts)
        self.assertEqual(verdicts["options"]["A"]["verdict"], "contradict")
        self.assertEqual(verdicts["options"]["B"]["verdict"], "contradict")
        self.assertEqual(len(changes), 2)

    def test_reconcile_rounded_range_and_closed_policy_absence(self):
        q = {
            "domain": "insurance",
            "question": "关于免赔额，哪些说法正确？",
            "options": {
                "A": "平安安佑福重疾险无免赔额",
                "B": "历史数据在63%至66%之间",
            },
        }
        verdicts = {"options": {
            "A": {"verdict": "uncertain", "confidence": 0.6, "reason": "全文未命中"},
            "B": {"verdict": "contradict", "confidence": 0.9,
                  "reason": "数值分别为66.38%、65.54%和63.51%，其中66.38%大于66%"},
        }}
        changes = reconcile_verdicts(
            q, verdicts,
            {"4": "平安安佑福重大疾病保险条款", "5": "平安e生保医疗险"},
            {"4": {"免赔额": 0}, "5": {"免赔额": 29}},
        )
        self.assertEqual(verdicts["options"]["A"]["verdict"], "support")
        self.assertEqual(verdicts["options"]["B"]["verdict"], "support")
        self.assertEqual(len(changes), 2)

    def test_reconcile_payout_filter_and_comparison_do_not_confuse_units(self):
        q = {
            "domain": "insurance",
            "question": "以下哪些保险产品可以赔付？",
            "options": {"A": "甲产品（仅保障身故，不赔医疗费）"},
        }
        verdicts = {"options": {"A": {
            "verdict": "support", "confidence": 0.9, "reason": "括号描述属实",
        }}}
        reconcile_verdicts(q, verdicts)
        self.assertEqual(verdicts["options"]["A"]["verdict"], "contradict")

        compare_q = {
            "domain": "financial_reports", "question": "分红对比",
            "options": {"A": "甲公司每股现金分红高于乙公司"},
        }
        compare_v = {"options": {"A": {
            "verdict": "support", "confidence": 1.0,
            "reason": "甲公司每10股派43元，即每股4.3元，高于乙公司。",
        }}}
        reconcile_verdicts(compare_q, compare_v)
        self.assertEqual(compare_v["options"]["A"]["verdict"], "support")

    def test_reconcile_product_clause_presence_and_research_omitted_subject(self):
        q = {
            "domain": "insurance",
            "question": "哪些产品明确宽限期后效力中止？",
            "options": {"A": "太保团体百万医疗"},
        }
        verdicts = {"options": {"A": {
            "verdict": "support", "confidence": 1.0, "reason": "引用了另一产品条款",
        }}}
        reconcile_verdicts(
            q, verdicts,
            {"6": "太保团体百万医疗保险条款", "2": "国寿增益宝条款"},
            {"6": {"宽限期": 0, "效力中止": 0},
             "2": {"宽限期": 10, "效力中止": 10}},
        )
        self.assertEqual(verdicts["options"]["A"]["verdict"], "contradict")

        rq = {
            "domain": "research", "question": "研究报告中哪些说法正确？",
            "options": {"A": "2005-2018年复合增速为9.9%"},
        }
        rv = {"options": {"A": {
            "verdict": "contradict", "confidence": 1.0,
            "reason": "证据明确指出该事实，但选项未限定主体，属于主体归属错误。",
        }}}
        reconcile_verdicts(rq, rv)
        self.assertEqual(rv["options"]["A"]["verdict"], "support")

    def test_tf_router_keeps_simple_fast_path_but_escalates_numeric_contract(self):
        simple = {
            "answer_format": "tf", "domain": "insurance",
            "doc_ids": ["a", "b"], "question": "两份报告的描述是否正确？",
        }
        composite = {
            "answer_format": "tf", "domain": "financial_contracts",
            "doc_ids": ["a", "b"], "question": "两份文档中还披露了资产负债率43.24%。",
        }
        research = {**simple, "domain": "research"}
        self.assertTrue(_use_baseline_tf(simple, True))
        self.assertFalse(_use_baseline_tf(research, True))
        self.assertFalse(_use_baseline_tf(composite, True))


if __name__ == "__main__":
    unittest.main()
