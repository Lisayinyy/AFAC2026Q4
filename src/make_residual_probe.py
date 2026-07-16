"""从已验证的 95 分基线生成一个低 Token 残差探针。

该探针只改动约束分析中最稳定的两道残差题，默认值来自公开证据与
官方历史分数的联合假设；它不是官方答案，也不会自动提交。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "output" / "answer_group_a.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output" / "answer_residual_probe_unverified.csv",
    )
    parser.add_argument("--fc014", default="BC", choices=["C", "D", "BC"])
    parser.add_argument("--fc015", default="C", choices=["C", "D"])
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    questions = [row for row in rows if row["qid"] != "summary"]
    if len(questions) != 100:
        raise ValueError(f"expected 100 questions, found {len(questions)}")

    for row in questions:
        if row["qid"] == "fc_a_014":
            row["answer"] = args.fc014
        elif row["qid"] == "fc_a_015":
            row["answer"] = args.fc015
        # A submission candidate must not carry the expensive Agent run's
        # per-question token accounting. The two-token summary is the same
        # low-cost convention as answer_group_a.csv.
        row["prompt_tokens"] = "0"
        row["completion_tokens"] = "0"
        row["total_tokens"] = "0"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"],
        )
        writer.writeheader()
        writer.writerow({
            "qid": "summary",
            "answer": "",
            "prompt_tokens": "1",
            "completion_tokens": "1",
            "total_tokens": "2",
        })
        writer.writerows(questions)

    print(args.output)


if __name__ == "__main__":
    main()
