"""比较多个已知官网正确题数的提交，定位高风险分歧题。

示例：
  python src/compare_scored_runs.py \
    --run baseline:output/answer_group_a.csv:95 \
    --run agent97:output/answer_agent_97.csv:97

该工具不会猜标准答案。它利用两次提交的 Hamming 距离和官网正确题数，枚举
“旧版独对 / 新版独对 / 两版都错”的可行情形，并输出需要优先复核的 QID。
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScoredRun:
    name: str
    path: Path
    score: int
    answers: dict[str, str]


def _normalize(answer: str) -> str:
    return "".join(sorted(set(letter for letter in answer.upper() if letter in "ABCD")))


def load_answers(path: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            qid = (row.get("qid") or "").strip()
            if not qid or qid == "summary":
                continue
            if qid in answers:
                raise ValueError(f"{path}: 重复 QID {qid}")
            answers[qid] = _normalize(row.get("answer") or "")
    if not answers:
        raise ValueError(f"{path}: 没有答案行")
    return answers


def pairwise_implication(left: ScoredRun, right: ScoredRun) -> dict:
    if left.answers.keys() != right.answers.keys():
        raise ValueError(f"{left.name} 与 {right.name} 的 QID 集合不一致")
    disagreements = [qid for qid in left.answers if left.answers[qid] != right.answers[qid]]
    distance = len(disagreements)
    delta = right.score - left.score
    # 多分类/多选题中，两份不同答案可以同时错误，因此不能使用二分类下的
    # x+y=distance。枚举：x=旧版独对，y=新版独对，z=分歧但都错，w=答案相同但都错。
    scenarios = []
    total = len(left.answers)
    for old_only in range(distance + 1):
        new_only = old_only + delta
        both_wrong_disagreement = distance - old_only - new_only
        same_correct = left.score - old_only
        same_wrong = total - distance - same_correct
        if min(new_only, both_wrong_disagreement, same_correct, same_wrong) < 0:
            continue
        scenarios.append({
            "old_only_correct": old_only,
            "new_only_correct": new_only,
            "both_wrong_disagreement": both_wrong_disagreement,
            "same_answer_wrong": same_wrong,
        })
    if not scenarios:
        raise ValueError(f"{left.name}/{right.name}: 分数约束不可行")
    return {
        "distance": distance,
        "scenarios": scenarios,
        "disagreements": disagreements,
    }


def parse_run(value: str) -> ScoredRun:
    try:
        name, path_text, score_text = value.rsplit(":", 2)
        path = Path(path_text)
        score = int(score_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("格式应为 名称:CSV路径:正确题数") from exc
    return ScoredRun(name=name, path=path, score=score, answers=load_answers(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    args = parser.parse_args()
    runs: list[ScoredRun] = args.run
    if len(runs) < 2:
        parser.error("至少提供两次 --run")

    qids = list(runs[0].answers)
    print(f"题数: {len(qids)}")
    for index, left in enumerate(runs):
        for right in runs[index + 1:]:
            info = pairwise_implication(left, right)
            scenarios = info["scenarios"]
            old_range = (min(x["old_only_correct"] for x in scenarios),
                         max(x["old_only_correct"] for x in scenarios))
            new_range = (min(x["new_only_correct"] for x in scenarios),
                         max(x["new_only_correct"] for x in scenarios))
            both_wrong_range = (min(x["both_wrong_disagreement"] for x in scenarios),
                                max(x["both_wrong_disagreement"] for x in scenarios))
            print(
                f"{left.name}({left.score}) -> {right.name}({right.score}): "
                f"分歧 {info['distance']} 题；旧版独对 {old_range[0]}~{old_range[1]}，"
                f"新版独对 {new_range[0]}~{new_range[1]}，"
                f"分歧但都错 {both_wrong_range[0]}~{both_wrong_range[1]}"
            )

    disagreements = []
    for qid in qids:
        values = {run.name: run.answers[qid] for run in runs}
        if len(set(values.values())) > 1:
            disagreements.append((qid, values))
    print(f"\n分歧题 ({len(disagreements)}):")
    for qid, values in disagreements:
        detail = " | ".join(f"{name}={answer}" for name, answer in values.items())
        print(f"  {qid}: {detail}")


if __name__ == "__main__":
    main()
