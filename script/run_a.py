#!/usr/bin/env python3
# 运行 A 榜全部题目，生成 answer.csv（含 summary token 统计行）+ evidence.json
import csv
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import config
from agent.answer import answer_question
from agent.qwen_client import LEDGER


def load_questions(domains=None):
    qs = []
    for f in sorted(config.QUESTIONS_DIR.glob("*_questions.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("questions", [])
        qs.extend(items)
    if domains:
        qs = [q for q in qs if q["domain"] in domains]
    return qs


def main():
    domains = sys.argv[1:] or None
    questions = load_questions(domains)
    print(f"共 {len(questions)} 题")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results, evidence = [], {}
    for i, q in enumerate(questions):
        try:
            r = answer_question(q)
        except Exception:
            traceback.print_exc()
            r = {"qid": q["qid"], "answer": "", "doc_ids_used": [], "evidence_ids": []}
        results.append(r)
        evidence[q["qid"]] = {"doc_ids": r["doc_ids_used"], "chunks": r["evidence_ids"]}
        print(f"[{i + 1}/{len(questions)}] {q['qid']} -> {r['answer'] or '(空)'}  "
              f"累计token={LEDGER.total_tokens:,}")

    # answer.csv：首行 summary，后续逐题
    s = LEDGER.summary()
    with open(config.OUTPUT_DIR / "answer.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        w.writerow(["summary", "", s["prompt_tokens"], s["completion_tokens"], s["total_tokens"]])
        for r in results:
            u = LEDGER.per_qid.get(r["qid"], {"prompt_tokens": 0, "completion_tokens": 0})
            w.writerow([r["qid"], r["answer"], u["prompt_tokens"], u["completion_tokens"],
                        u["prompt_tokens"] + u["completion_tokens"]])

    with open(config.OUTPUT_DIR / "evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    LEDGER.dump(config.OUTPUT_DIR / "token_ledger.json")
    print(f"\n完成。总token={s['total_tokens']:,} / 预算 {config.TOKEN_BUDGET_TOTAL:,}")
    print(f"输出: {config.OUTPUT_DIR / 'answer.csv'}")


if __name__ == "__main__":
    main()
