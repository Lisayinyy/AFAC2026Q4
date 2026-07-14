"""离线评估切块/召回覆盖，不调用 LLM。

用法（需要安装项目依赖）：
    PYTHONPATH=src python src/evaluate_retrieval.py --limit 20

指标：
* option_coverage：选项关键词/数字至少命中一个召回 chunk 的比例；
* doc_coverage：题目引用文档在召回结果中的覆盖率；
* compression：召回上下文字符数 / 相关文档全文字符数。
该工具用于比较切块和召回参数，不等价于官网准确率。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from config import QUESTIONS_ROOT
from retrieve import DocRetriever, _tokenize, build_context


def _evidence_terms(text: str) -> set[str]:
    terms = {t for t in _tokenize(text) if len(t) >= 2}
    terms.update(re.findall(r"\d[\d,.%]*", text))
    return terms


def load_questions(domains: list[str] | None) -> list[dict]:
    names = domains or [
        "insurance", "financial_reports", "financial_contracts",
        "regulatory", "research",
    ]
    out = []
    for domain in names:
        path = QUESTIONS_ROOT / f"{domain}_questions.json"
        out.extend(json.loads(path.read_text(encoding="utf-8")))
    return out


def evaluate(limit: int = 0, domains: list[str] | None = None, top_k: int = 12) -> dict:
    questions = load_questions(domains)
    if limit:
        questions = questions[:limit]
    option_hits = 0
    option_total = 0
    doc_hits = 0
    doc_total = 0
    recalled_chars = 0
    source_chars = 0
    by_domain: dict[str, dict[str, int]] = {}
    for q in questions:
        retriever = DocRetriever(q["domain"], q["doc_ids"])
        chunks = retriever.retrieve(
            q["question"], list(q["options"].values()),
            pool=60, top_k=top_k, domain=q["domain"],
        )
        recalled_chars += len(build_context(chunks, max_chars=10**9))
        source_chars += sum(len(c["text"]) for c in retriever.chunks)
        recalled_docs = {c["doc_id"] for c in chunks}
        doc_hits += len(recalled_docs & set(q["doc_ids"]))
        doc_total += len(set(q["doc_ids"]))
        chunk_terms = [_evidence_terms(c["text"]) for c in chunks]
        for option in q["options"].values():
            option_total += 1
            stats = by_domain.setdefault(q["domain"], {"hits": 0, "total": 0})
            stats["total"] += 1
            terms = _evidence_terms(option)
            if terms and any(terms & cterms for cterms in chunk_terms):
                option_hits += 1
                stats["hits"] += 1
    return {
        "questions": len(questions),
        "option_coverage": round(option_hits / option_total, 4) if option_total else 0,
        "doc_coverage": round(doc_hits / doc_total, 4) if doc_total else 0,
        "compression": round(recalled_chars / source_chars, 4) if source_chars else 0,
        "option_hits": option_hits,
        "option_total": option_total,
        "by_domain": {
            domain: {
                **stats,
                "coverage": round(stats["hits"] / stats["total"], 4) if stats["total"] else 0,
            }
            for domain, stats in sorted(by_domain.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--domains", nargs="*")
    ap.add_argument("--top-k", type=int, default=12)
    args = ap.parse_args()
    print(json.dumps(evaluate(args.limit, args.domains, args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
