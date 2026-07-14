"""用标准答案给 answer.csv 打分,按领域/题型拆解正确率。"""
import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GT_DEFAULT = PROJECT_ROOT / "output" / "answer_group_a.csv"
Q_DIR_DEFAULT = PROJECT_ROOT.parent / "public_dataset_upload" / "questions" / "group_a"
Q_FILES = {
    "insurance": "insurance_questions.json",
    "financial_reports": "financial_reports_questions.json",
    "financial_contracts": "financial_contracts_questions.json",
    "regulatory": "regulatory_questions.json",
    "research": "research_questions.json",
}


def norm(ans: str, fmt: str) -> str:
    letters = [c for c in ans.upper() if c in "ABCD"]
    if fmt in ("mcq", "tf"):
        return letters[0] if letters else ""
    return "".join(sorted(set(letters)))


def load_csv(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["qid"] == "summary":
                continue
            d[r["qid"]] = r["answer"]
    return d


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("ground_truth", nargs="?", type=Path, default=GT_DEFAULT)
    parser.add_argument("--questions-dir", type=Path, default=Q_DIR_DEFAULT)
    args = parser.parse_args()

    # qid -> (domain, fmt)
    meta = {}
    for dom, fn in Q_FILES.items():
        path = args.questions_dir / fn
        if not path.is_file():
            parser.error(f"题目文件不存在: {path}；请通过 --questions-dir 指定题目目录")
        for q in json.loads(path.read_text(encoding="utf-8")):
            meta[q["qid"]] = (dom, q["answer_format"])

    pred = load_csv(args.prediction)
    gt = load_csv(args.ground_truth)

    # 统计
    from collections import defaultdict
    by_dom = defaultdict(lambda: [0, 0])   # [correct, total]
    by_fmt = defaultdict(lambda: [0, 0])
    total_c = total_n = 0
    wrong = []
    for qid, g in gt.items():
        dom, fmt = meta.get(qid, ("?", "multi"))
        p = pred.get(qid, "")
        ok = norm(p, fmt) == norm(g, fmt)
        by_dom[dom][1] += 1
        by_fmt[fmt][1] += 1
        total_n += 1
        if ok:
            by_dom[dom][0] += 1
            by_fmt[fmt][0] += 1
            total_c += 1
        else:
            wrong.append((qid, dom, fmt, norm(p, fmt), norm(g, fmt)))

    acc = total_c / total_n if total_n else 0
    print(f"=== 总正确率: {total_c}/{total_n} = {acc:.3f} ===\n")
    print("按领域:")
    for dom, (c, n) in sorted(by_dom.items()):
        print(f"  {dom:20s}: {c}/{n} = {c/n:.2f}")
    print("按题型:")
    for fmt, (c, n) in sorted(by_fmt.items()):
        print(f"  {fmt:6s}: {c}/{n} = {c/n:.2f}")
    print(f"\n错题({len(wrong)}): qid | 领域 | 题型 | 我答 | 正确")
    for qid, dom, fmt, p, g in wrong:
        print(f"  {qid} | {dom[:8]:8s} | {fmt:5s} | {p:5s} | {g}")


if __name__ == "__main__":
    main()
