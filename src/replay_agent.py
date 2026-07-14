"""用最新确定性裁决规则重放已有 Agent 轨迹，无需重新调用模型。

该工具不读取标准答案；它只重建全文关键词统计，然后对轨迹中的
``final_verdicts`` 运行与在线 Agent 完全相同的 ``reconcile_verdicts`` 和
``_final_answer``。
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from agent import (_final_answer, build_document_profiles,
                   build_document_term_stats, reconcile_verdicts)
from config import OUTPUT_ROOT, TOKEN_BUDGET
from retrieve import DocRetriever
from run import load_questions, token_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-name", default="answer_agent_v3_final.csv")
    parser.add_argument("--output-name", default="answer_agent_v3_stable.csv")
    args = parser.parse_args()

    input_csv = OUTPUT_ROOT / args.input_name
    input_trace = OUTPUT_ROOT / f"{input_csv.stem}_traces.jsonl"
    input_summary = OUTPUT_ROOT / f"{input_csv.stem}_summary.json"
    output_csv = OUTPUT_ROOT / args.output_name
    output_trace = OUTPUT_ROOT / f"{output_csv.stem}_traces.jsonl"
    output_summary = OUTPUT_ROOT / f"{output_csv.stem}_summary.json"

    if not input_csv.exists() or not input_trace.exists():
        parser.error(f"缺少输入文件: {input_csv} / {input_trace}")

    questions = load_questions(None)
    by_qid = {q["qid"]: q for q in questions}
    rows: dict[str, dict] = {}
    with input_csv.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["qid"] != "summary":
                rows[row["qid"]] = row

    traces: dict[str, dict] = {}
    with input_trace.open(encoding="utf-8") as file:
        for line in file:
            trace = json.loads(line)
            traces[trace["qid"]] = trace

    changed = 0
    replayed_traces = []
    for q in questions:
        qid = q["qid"]
        trace = copy.deepcopy(traces[qid])
        old_answer = rows[qid]["answer"]
        if trace.get("architecture") == "baseline_tf_fast_path":
            new_answer = trace["answer"]
            adjustments = []
        else:
            retriever = DocRetriever(q["domain"], q["doc_ids"])
            profiles = trace.get("document_profiles") or build_document_profiles(retriever)
            term_stats = build_document_term_stats(retriever, q)
            verdicts = copy.deepcopy(trace["final_verdicts"])
            adjustments = reconcile_verdicts(
                q, verdicts, profiles, term_stats
            )
            new_answer = _final_answer(q, verdicts)
            trace["final_verdicts"] = verdicts
            trace["document_term_stats"] = term_stats
        rows[qid]["answer"] = new_answer
        trace["replayed_from_answer"] = old_answer
        trace["replay_adjustments"] = adjustments
        trace["answer"] = new_answer
        changed += int(new_answer != old_answer)
        replayed_traces.append(trace)

    prompt = sum(int(row["prompt_tokens"]) for row in rows.values())
    completion = sum(int(row["completion_tokens"]) for row in rows.values())
    total = sum(int(row["total_tokens"]) for row in rows.values())
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"],
        )
        writer.writeheader()
        writer.writerow({
            "qid": "summary", "answer": "", "prompt_tokens": prompt,
            "completion_tokens": completion, "total_tokens": total,
        })
        for q in questions:
            writer.writerow(rows[q["qid"]])

    with output_trace.open("w", encoding="utf-8") as file:
        for trace in replayed_traces:
            file.write(json.dumps(trace, ensure_ascii=False) + "\n")

    summary = {}
    if input_summary.exists():
        summary = json.loads(input_summary.read_text(encoding="utf-8"))
    summary.update({
        "architecture": "option_evidence_agent_v3_stable",
        "replayed_from": input_csv.name,
        "num_replay_answer_changes": changed,
        "total_tokens": total,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "token_budget": TOKEN_BUDGET,
        "token_score": round(token_score(total), 4),
    })
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"答案: {output_csv}")
    print(f"轨迹: {output_trace}")


if __name__ == "__main__":
    main()
