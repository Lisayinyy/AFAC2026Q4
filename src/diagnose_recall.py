"""检索召回诊断:定位多选题"漏选"的根因是【检索漏】还是【判定漏】。

对每道多选题的每个选项,抽取"特征词"(数字 + 长词元),统计:
  - hit_ctx  : 特征词在【最终召回 context】里的命中率
  - hit_full : 特征词在【完整引用文档全文】里的命中率

据此把"标准答案里的正确选项"分类:
  - RETRIEVED   证据已进 context(hit_ctx 高)         → 若仍漏选 = 判定漏
  - MISSED_DOC  证据在全文但没进 context(hit_full 高、hit_ctx 低) → 检索漏(可扩容/均衡修)
  - NO_LEXICAL  全文里也无字面证据(hit_full 低)        → 改写/推理题,BM25 天然抓不到

用法:
    python src/diagnose_recall.py            # 全部领域
    python src/diagnose_recall.py research   # 指定领域
不消耗 token(检索走纯 BM25,跳过 rerank)。
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from config import QUESTIONS_ROOT, OUTPUT_ROOT
from retrieve import DocRetriever, build_context, _tokenize

GT_PATH = OUTPUT_ROOT / "answer_group_a.csv"
PRED_PATH = OUTPUT_ROOT / "answer.csv"

DOMAIN_FILES = {
    "insurance": "insurance_questions.json",
    "financial_reports": "financial_reports_questions.json",
    "financial_contracts": "financial_contracts_questions.json",
    "regulatory": "regulatory_questions.json",
    "research": "research_questions.json",
}

# 与 answer_question 主入口一致的上下文参数
def _ctx_params(domain: str) -> tuple[int, int]:
    if domain == "financial_reports":
        return 16, 15000
    return 14, 12000


def _load_csv(path: Path) -> dict[str, str]:
    d = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["qid"] == "summary":
                continue
            d[r["qid"]] = (r.get("answer") or "").strip()
    return d


def _option_features(text: str) -> tuple[set[str], set[str]]:
    """抽取选项的特征:数字(含%/小数)与长词元(>=2字),用于在文本里找证据。"""
    nums = set(re.findall(r"\d[\d,\.]*%?", text))
    nums = {n for n in nums if len(n) >= 2}          # 过滤单个数字噪声
    terms = {t for t in _tokenize(text) if len(t) >= 2}
    return nums, terms


def _hit_rate(features: tuple[set[str], set[str]], blob: str) -> float:
    """特征词在文本里的命中率(数字权重更高:数值题的关键)。"""
    nums, terms = features
    total = len(nums) * 2 + len(terms)
    if total == 0:
        return 1.0                                   # 无可判特征,视为可命中
    hit = sum(2 for n in nums if n in blob)
    hit += sum(1 for t in terms if t in blob)
    return hit / total


def diagnose(domains: list[str]) -> None:
    gt = _load_csv(GT_PATH)
    pred = _load_csv(PRED_PATH)

    cats = Counter()                 # 正确选项的证据分类
    miss_cats = Counter()            # 仅统计"被漏掉"的正确选项
    rows = []                        # 明细

    for dom in domains:
        questions = json.loads(
            (QUESTIONS_ROOT / DOMAIN_FILES[dom]).read_text(encoding="utf-8"))
        for q in questions:
            if q["answer_format"] != "multi":
                continue
            qid = q["qid"]
            options = q["options"]
            gold = set(c for c in gt.get(qid, "") if c in options)
            mine = set(c for c in pred.get(qid, "") if c in options)
            missed = gold - mine     # 我漏掉的正确选项

            # 真实检索(纯 BM25,不接 llm=不走 rerank,零 token)
            retriever = DocRetriever(dom, q["doc_ids"], llm=None)
            top_k, max_chars = _ctx_params(dom)
            chunks = retriever.retrieve(
                q["question"], list(options.values()),
                pool=60, top_k=top_k, domain=dom)
            context = build_context(chunks, max_chars=max_chars)
            full_text = "\n".join(c["text"] for c in retriever.chunks)

            for letter in gold:
                feat = _option_features(options[letter])
                h_ctx = _hit_rate(feat, context)
                h_full = _hit_rate(feat, full_text)
                if h_ctx >= 0.5:
                    cat = "RETRIEVED"
                elif h_full >= 0.5:
                    cat = "MISSED_DOC"
                else:
                    cat = "NO_LEXICAL"
                cats[cat] += 1
                if letter in missed:
                    miss_cats[cat] += 1
                    rows.append((qid, dom, letter, cat,
                                 round(h_ctx, 2), round(h_full, 2),
                                 options[letter][:28]))

    # ---- 汇总 ----
    print("=" * 70)
    print("【全部正确选项】证据分布(共 %d 个):" % sum(cats.values()))
    for c in ("RETRIEVED", "MISSED_DOC", "NO_LEXICAL"):
        n = cats[c]
        print(f"  {c:12s}: {n:3d}  ({n/max(1,sum(cats.values())):.0%})")
    print()
    print("【被漏掉的正确选项】证据分布(共 %d 个)——这是要解决的目标:" %
          sum(miss_cats.values()))
    for c in ("RETRIEVED", "MISSED_DOC", "NO_LEXICAL"):
        n = miss_cats[c]
        tot = max(1, sum(miss_cats.values()))
        tag = {"RETRIEVED": "判定漏(证据在context里但没选)",
               "MISSED_DOC": "检索漏(证据在全文但没召回)",
               "NO_LEXICAL": "无字面证据(改写/推理题)"}[c]
        print(f"  {c:12s}: {n:3d}  ({n/tot:.0%})  <- {tag}")
    print("=" * 70)
    print("\n漏选明细: qid | 领域 | 漏掉选项 | 分类 | ctx命中 | 全文命中 | 选项文本")
    for r in sorted(rows, key=lambda x: (x[3], x[1])):
        print("  %-10s | %-18s | %s | %-10s | %.2f | %.2f | %s" %
              (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))


if __name__ == "__main__":
    doms = sys.argv[1:] if len(sys.argv) > 1 else list(DOMAIN_FILES.keys())
    diagnose(doms)
