"""运行金融长文本 Agent，并生成独立于 baseline 的答案、汇总和审计轨迹。

用法：
    python src/run_agent.py --limit 5
    python src/run_agent.py
    python src/run_agent.py --resume
    python src/run_agent.py --domains financial_reports insurance
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent import answer_question_agent
from answer import answer_question
from config import OUTPUT_ROOT, TOKEN_BUDGET, LLMClient, TokenMeter
from run import load_questions, token_score


def _use_baseline_tf(q: dict, baseline_tf: bool) -> bool:
    """简单判断题走低成本 baseline，复合数值合同事实交给 Agent。"""
    if not baseline_tf or q["answer_format"] != "tf":
        return False
    composite_numeric_contract = (
        q["domain"] == "financial_contracts"
        and len(q.get("doc_ids", [])) > 1
        and bool(re.search(r"\d+(?:\.\d+)?\s*%", q["question"]))
    )
    research_fact_check = q["domain"] == "research"
    return not (composite_numeric_contract or research_fact_check)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--qids", nargs="*", default=None,
                    help="仅运行指定 qid，用于针对性回归测试")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--option-rerank", action="store_true")
    ap.add_argument("--llm-planner", action="store_true",
                    help="使用 LLM 规划器；默认采用低成本确定性规划")
    ap.add_argument("--separate-judge", action="store_true",
                    help="记忆压缩后再单独调用一次裁决模型")
    ap.add_argument("--baseline-tf", action="store_true",
                    help="判断题走已验证的 baseline 快速路径")
    ap.add_argument("--output-name", default="answer_agent.csv")
    args = ap.parse_args()

    OUTPUT_ROOT.mkdir(exist_ok=True)
    questions = load_questions(args.domains)
    if args.qids:
        wanted = set(args.qids)
        questions = [q for q in questions if q["qid"] in wanted]
        missing = wanted - {q["qid"] for q in questions}
        if missing:
            ap.error(f"未找到 qid: {', '.join(sorted(missing))}")
    if args.limit:
        questions = questions[:args.limit]

    csv_path = OUTPUT_ROOT / args.output_name
    summary_path = OUTPUT_ROOT / f"{csv_path.stem}_summary.json"
    trace_path = OUTPUT_ROOT / f"{csv_path.stem}_traces.jsonl"
    rows_by_qid: dict[str, dict] = {}
    traces_by_qid: dict[str, dict] = {}
    lock = threading.Lock()
    t0 = time.time()

    if args.resume and csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["qid"] == "summary":
                    continue
                total = int(row.get("total_tokens") or 0)
                if total > 0 and (row.get("answer") or "").strip():
                    rows_by_qid[row["qid"]] = {
                        "qid": row["qid"], "answer": row["answer"],
                        "prompt_tokens": int(row.get("prompt_tokens") or 0),
                        "completion_tokens": int(row.get("completion_tokens") or 0),
                        "total_tokens": total,
                    }
        if trace_path.exists():
            with trace_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        trace = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if trace.get("qid"):
                        traces_by_qid[trace["qid"]] = trace
        print(f"[RESUME] 已完成 {len(rows_by_qid)} 题", flush=True)

    pending = [q for q in questions if q["qid"] not in rows_by_qid]

    def write_outputs() -> None:
        prompt = sum(r["prompt_tokens"] for r in rows_by_qid.values())
        completion = sum(r["completion_tokens"] for r in rows_by_qid.values())
        total = sum(r["total_tokens"] for r in rows_by_qid.values())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["qid", "answer", "prompt_tokens",
                               "completion_tokens", "total_tokens"]
            )
            writer.writeheader()
            writer.writerow({
                "qid": "summary", "answer": "",
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            })
            for q in questions:
                if q["qid"] in rows_by_qid:
                    writer.writerow(rows_by_qid[q["qid"]])
        with trace_path.open("w", encoding="utf-8") as f:
            for q in questions:
                if q["qid"] in traces_by_qid:
                    f.write(json.dumps(traces_by_qid[q["qid"]], ensure_ascii=False) + "\n")

    def worker(q: dict):
        meter = TokenMeter()
        client = LLMClient(meter)
        try:
            if _use_baseline_tf(q, args.baseline_tf):
                answer = answer_question(client, q)
                trace = {
                    "qid": q["qid"],
                    "architecture": "baseline_tf_fast_path",
                    "answer": answer,
                    "review_targets": [],
                }
            else:
                answer, trace = answer_question_agent(
                    client, q,
                    enable_review=not args.no_review,
                    use_option_rerank=args.option_rerank,
                    use_llm_planner=args.llm_planner,
                    separate_judge=args.separate_judge,
                )
        except Exception as exc:
            print(f"[AGENT ERROR] {q['qid']}: {type(exc).__name__}: {exc}", flush=True)
            # 不伪造答案；让断点续跑通过 total_tokens=0/空答案识别失败题。
            answer, trace = "", {
                "qid": q["qid"], "error": f"{type(exc).__name__}: {exc}"
            }
        return q, answer, trace, meter

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(worker, q): q for q in pending}
        for future in as_completed(futures):
            q, answer, trace, meter = future.result()
            with lock:
                done += 1
                rows_by_qid[q["qid"]] = {
                    "qid": q["qid"], "answer": answer,
                    "prompt_tokens": meter.prompt_tokens,
                    "completion_tokens": meter.completion_tokens,
                    "total_tokens": meter.total_tokens,
                }
                traces_by_qid[q["qid"]] = trace
                total = sum(r["total_tokens"] for r in rows_by_qid.values())
                print(
                    f"[{done}/{len(pending)}] {q['qid']} -> {answer or 'ERROR'} "
                    f"| +{meter.total_tokens} tok | 累计={total} "
                    f"| review={len(trace.get('review_targets', []))}",
                    flush=True,
                )
                write_outputs()

    prompt = sum(r["prompt_tokens"] for r in rows_by_qid.values())
    completion = sum(r["completion_tokens"] for r in rows_by_qid.values())
    total = sum(r["total_tokens"] for r in rows_by_qid.values())
    summary = {
        "architecture": "option_evidence_agent_v3",
        "total_tokens": total,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "token_budget": TOKEN_BUDGET,
        "token_score": round(token_score(total), 4),
        "num_questions": len(questions),
        "num_answered": sum(bool(r["answer"]) for r in rows_by_qid.values()),
        "num_reviewed_questions": sum(
            bool(t.get("review_targets")) for t in traces_by_qid.values()
        ),
        "elapsed_sec": round(time.time() - t0, 1),
        "option_rerank": args.option_rerank,
        "llm_planner": args.llm_planner,
        "separate_judge": args.separate_judge,
        "baseline_tf": args.baseline_tf,
        "conditional_review": not args.no_review,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n==== Agent 完成 ====", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"答案: {csv_path}", flush=True)
    print(f"轨迹: {trace_path}", flush=True)


if __name__ == "__main__":
    main()
