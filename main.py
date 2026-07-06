"""AFAC2026 统一入口 —— 执行即产出 output/submit.zip。

用法:
    python main.py                 # 有 data/sample 用真实样例集, 否则合成兜底
    python main.py --alpha 0.6     # 调 Wasserstein/DTW 综合权重

审核要求: main.py 为唯一入口, 相对路径, 不硬编码结果。
"""
from __future__ import annotations

import argparse

from src import (
    config,
    data_loader,
    pipeline,
    submit,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="AFAC2026 资金流向识别管线")
    ap.add_argument("--alpha", type=float, default=config.WASSERSTEIN_ALPHA,
                    help="综合距离中 Wasserstein 权重 (0-1)")
    ap.add_argument("--source", choices=["auto", "sample", "snapshot", "synthetic"],
                    default="auto", help="数据来源")
    ap.add_argument("--snapshot-path", default=None,
                    help="原始十档快照文件路径 (xlsx/csv), 配合 --source snapshot")
    ap.add_argument("--train", action="store_true",
                    help="启用弱标签自训练(高置信规则伪标签→logistic);样本不足自动回退规则")
    args = ap.parse_args()

    print("=" * 60)
    print("AFAC2026 · 市场参与者交易行为识别与资金流向分析")
    print("=" * 60)

    # 1) 数据
    if args.source == "snapshot":
        if not args.snapshot_path:
            raise SystemExit("--source snapshot 需配合 --snapshot-path 指定文件")
        df = data_loader.load_snapshot(args.snapshot_path)
        src_tag = f"原始十档快照 ({args.snapshot_path})"
    elif args.source == "synthetic":
        df, src_tag = data_loader.generate_synthetic(), "合成兜底数据(SYNTHETIC)"
    else:  # auto / sample
        df, is_syn = data_loader.load_feature_set()
        src_tag = "合成兜底数据(SYNTHETIC)" if is_syn else f"官方样例集 ({config.SAMPLE_DIR})"
        if is_syn:
            print("      ⚠ 未发现 data/sample/*.csv, 使用合成数据验证管线。")
    print(f"[1/4] 数据加载: {src_tag} | {len(df)} 行, {df['symbol'].nunique()} 只股票")

    # 2) 核心管线 (特征→Task1→Task2→行情阶段→意图一致性)
    res = pipeline.run_once(df, alpha=args.alpha, use_self_training=args.train)
    rep = res["report"]
    print(f"[2/4] Task1 模式聚类: {rep['task1'].get('n_clusters')} 类 | "
          f"silhouette={rep['task1'].get('silhouette')}")
    print("      模式分布:", res["pattern"]["pattern_type"].value_counts().to_dict())
    print(f"[3/4] Task2 资金识别 (method={rep['task2'].get('method')}): {rep['task2']['type_dist']}")
    if rep["task2"].get("method") != "rule":
        print(f"      自训练: 伪标签{res['train_meta'].get('pseudo_n')}条 "
              f"| 与规则一致率={res['train_meta'].get('agree_with_rule')}")
    print("      意图分布:", rep["task2"]["intention_dist"],
          "| 行情阶段:", res["phase"]["market_phase"].value_counts().to_dict())
    if "type_weighted_f1" in rep["task2"]:
        print(f"      合成自检 加权F1(类型)={rep['task2']['type_weighted_f1']}")
    if "intent_pattern_consistency" in rep:
        print(f"      意图-模式一致率={rep['intent_pattern_consistency']}")

    # 4) 生成提交
    pp = submit.write_pattern(res["pattern"])
    qp = submit.write_predict(res["predict"])
    val = submit.validate(pp, qp)
    if not val["ok"]:
        print("[4/4] ✗ 提交校验失败:", val["issues"])
        raise SystemExit(1)
    zp = submit.pack(pp, qp)
    print(f"[4/4] 提交生成 ✓ {val['rows']} 行 | {zp}")
    print("      ->", pp, "\n      ->", qp)
    print("完成。")


if __name__ == "__main__":
    main()
