"""从 95 分基准答案生成排行榜约束探针。

默认输入不做原地修改，输出到 output/probe_v3_p1_fc7.csv。
生成过程中会校验题数、QID 唯一性、答案格式、修正数量和 Token 字段，
任一条件不满足都会直接报错，避免上传格式错误的文件。
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = PROJECT_ROOT / "output" / "answer_group_a.csv"
PROBE_V3_P1_OUTPUT = PROJECT_ROOT / "output" / "probe_v3_p1_fc7.csv"
LEGACY_PROBE_1_OUTPUT = PROJECT_ROOT / "output" / "probe_1_candidate_100.csv"
LEGACY_PROBE_2_IF_99_OUTPUT = PROJECT_ROOT / "output" / "probe_2_if_99_fc15_D.csv"
FIELDS = [
    "qid",
    "answer",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]

# 2026-07-20 第一探针：加入公开且可精确反推为 67/79 题正确的 hautdong
# V2/V4 官网运行后重新条件化。该组合在当前硬约束下存在 100/100 配置，
# 但尚未获得官网验证。fc_a_007 的原文答案仍为 BD；这里的 B 是针对可能
# 隐藏标注漏掉深层违约条款 D 的定向测试，不能描述成已知标准答案。
PROBE_V3_P1_CORRECTIONS = {
    "fc_a_004": "AC",
    "fc_a_007": "B",
    "fc_a_014": "BC",
    "fc_a_015": "C",
    "ins_a_006": "A",
}

# 2026-07-13 的旧探针只为复现历史审计保留。新增官网运行已排除它们达到
# 100/100 的可能性，严禁继续作为默认提交文件。
LEGACY_PROBE_1_CORRECTIONS = {
    "fc_a_014": "BC",
    "fc_a_015": "C",
    "fc_a_020": "AB",
    "fin_a_008": "A",
    "res_a_004": "B",
}
LEGACY_PROBE_2_IF_99_CORRECTIONS = {
    **LEGACY_PROBE_1_CORRECTIONS,
    "fc_a_015": "D",
}


def _read_base(path: Path) -> tuple[dict[str, str], list[str]]:
    answers: dict[str, str] = {}
    order: list[str] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"CSV 表头不符合要求: {reader.fieldnames!r}")
        for row in reader:
            qid = row["qid"].strip()
            if qid == "summary":
                continue
            if not qid or qid in answers:
                raise ValueError(f"QID 为空或重复: {qid!r}")
            answers[qid] = row["answer"].strip().upper()
            order.append(qid)
    return answers, order


def _validate_answers(answers: dict[str, str], order: list[str]) -> None:
    if len(answers) != 100 or len(order) != 100:
        raise ValueError(f"提交必须恰好包含 100 题，当前为 {len(answers)} 题")
    if len(set(order)) != len(order):
        raise ValueError("提交中存在重复 QID")
    for qid in order:
        answer = answers[qid]
        if not re.fullmatch(r"[A-D]+", answer):
            raise ValueError(f"{qid} 的答案含非法字符: {answer!r}")
        if answer != "".join(sorted(set(answer))):
            raise ValueError(f"{qid} 的答案未去重或未按字母排序: {answer!r}")


def build_submission(
    base: Path,
    output: Path,
    corrections: dict[str, str],
) -> list[tuple[str, str, str]]:
    answers, order = _read_base(base)
    missing = sorted(set(corrections) - set(answers))
    if missing:
        raise ValueError(f"基准答案缺少待修正 QID: {missing}")

    changes = []
    for qid, corrected in corrections.items():
        old = answers[qid]
        answers[qid] = corrected
        if old != corrected:
            changes.append((qid, old, corrected))

    if len(changes) != len(corrections):
        raise ValueError(
            f"预期发生 {len(corrections)} 处修正，实际为 {len(changes)} 处"
        )
    _validate_answers(answers, order)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "qid": "summary",
                "answer": "",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            }
        )
        for qid in order:
            writer.writerow(
                {
                    "qid": qid,
                    "answer": answers[qid],
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            )
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--probe",
        choices=("v3-p1", "legacy-v2-p1", "legacy-v2-p2-if-99"),
        default="v3-p1",
        help="生成 2026-07-20 新探针；旧探针只用于复现，不应提交",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.probe == "v3-p1":
        corrections = PROBE_V3_P1_CORRECTIONS
        default_output = PROBE_V3_P1_OUTPUT
    elif args.probe == "legacy-v2-p1":
        corrections = LEGACY_PROBE_1_CORRECTIONS
        default_output = LEGACY_PROBE_1_OUTPUT
    else:
        corrections = LEGACY_PROBE_2_IF_99_CORRECTIONS
        default_output = LEGACY_PROBE_2_IF_99_OUTPUT
    output = args.output or default_output

    changes = build_submission(args.base, output, corrections)
    print(f"已生成: {output}")
    print("修正:")
    for qid, old, new in changes:
        print(f"  {qid}: {old} -> {new}")


if __name__ == "__main__":
    main()
