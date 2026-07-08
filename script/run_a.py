#!/usr/bin/env python3
# 运行 A 榜题目，生成 answer.csv（含 summary token 统计行）+ evidence.json
# 支持断点续跑：进度实时写 output/progress.jsonl，重跑自动跳过已完成题目
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


def load_progress(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["qid"]] = r
    return done


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fresh = "--fresh" in sys.argv
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit="):
            limit = int(a.split("=")[1])
    questions = load_questions(args or None)
    if limit:
        questions = questions[:limit]
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = config.OUTPUT_DIR / "progress.jsonl"
    if fresh and progress_path.exists():
        progress_path.unlink()
    done = load_progress(progress_path)
    print(f"共 {len(questions)} 题，已完成 {len(done)} 题")

    # 恢复历史 token 记账
    for r in done.values():
        u = r.get("usage", {})
        LEDGER.prompt_tokens += u.get("prompt_tokens", 0)
        LEDGER.completion_tokens += u.get("completion_tokens", 0)
        LEDGER.per_qid[r["qid"]] = {
            "prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0),
        }

    results = []
    with open(progress_path, "a", encoding="utf-8") as pf:
        for i, q in enumerate(questions):
            if q["qid"] in done:
                results.append(done[q["qid"]])
                continue
            before = dict(LEDGER.per_qid.get(q["qid"], {"prompt_tokens": 0, "completion_tokens": 0}))
            try:
                r = answer_question(q)
            except Exception:
                traceback.print_exc()
                r = {"qid": q["qid"], "answer": "", "doc_ids_used": [], "evidence_ids": []}
            after = LEDGER.per_qid.get(q["qid"], {"prompt_tokens": 0, "completion_tokens": 0})
            r["usage"] = {
                "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
                "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
            }
            r.pop("raw", None)
            results.append(r)
            pf.write(json.dumps(r, ensure_ascii=False) + "\n")
            pf.flush()
            print(f"[{i + 1}/{len(questions)}] {q['qid']} -> {r['answer'] or '(空)'}  "
                  f"累计token={LEDGER.total_tokens:,}")

    s = LEDGER.summary()
    with open(config.OUTPUT_DIR / "answer.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        w.writerow(["summary", "", s["prompt_tokens"], s["completion_tokens"], s["total_tokens"]])
        for r in results:
            u = r.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})
            w.writerow([r["qid"], r["answer"], u["prompt_tokens"], u["completion_tokens"],
                        u["prompt_tokens"] + u["completion_tokens"]])

    evidence = {r["qid"]: {"doc_ids": r["doc_ids_used"], "chunks": r["evidence_ids"]}
                for r in results}
    with open(config.OUTPUT_DIR / "evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    LEDGER.dump(config.OUTPUT_DIR / "token_ledger.json")
    print(f"\n完成。总token={s['total_tokens']:,} / 预算 {config.TOKEN_BUDGET_TOTAL:,}")
    print(f"输出: {config.OUTPUT_DIR / 'answer.csv'}")


if __name__ == "__main__":
    main()
