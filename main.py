"""AFAC2026 统一入口 —— 执行即产出 output/submit.zip。

用法:
    python main.py                 # 有 data/sample 用真实样例集, 否则合成兜底
    python main.py --alpha 0.6     # 调 Wasserstein/DTW 综合权重

审核要求: main.py 为唯一入口, 相对路径, 不硬编码结果。
"""
from __future__ import annotations

import argparse

from src import (
    capital_classifier,
    config,
    data_loader,
    features,
    pattern_clustering,
    submit,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="AFAC2026 资金流向识别管线")
    ap.add_argument("--alpha", type=float, default=config.WASSERSTEIN_ALPHA,
                    help="综合距离中 Wasserstein 权重 (0-1)")
    args = ap.parse_args()

    print("=" * 60)
    print("AFAC2026 · 市场参与者交易行为识别与资金流向分析")
    print("=" * 60)

    # 1) 数据
    df, is_syn = data_loader.load_feature_set()
    src_tag = "合成兜底数据(SYNTHETIC)" if is_syn else f"官方样例集 ({config.SAMPLE_DIR})"
    print(f"[1/5] 数据加载: {src_tag} | {len(df)} 行, {df['symbol'].nunique()} 只股票")
    if is_syn:
        print("      ⚠ 未发现 data/sample/*.csv, 使用合成数据验证管线。放入真实样例集后重跑即为正式提交。")

    # 2) 特征工程
    df_feat = features.build_features(df)
    print(f"[2/5] 特征工程: 派生 {len(features.MODEL_FEATURE_COLS)} 类建模特征")

    # 3) Task1 交易模式识别
    patt = pattern_clustering.run(df_feat, alpha=args.alpha)
    metrics1 = pattern_clustering.quality_metrics(df_feat, patt["cluster"].to_numpy())
    print(f"[3/5] Task1 模式聚类: {metrics1.get('n_clusters')} 类 | "
          f"silhouette={metrics1.get('silhouette', float('nan')):.3f}")
    print("      模式分布:", patt["pattern_type"].value_counts().to_dict())

    # 4) Task2 资金类型 + 意图
    pred = capital_classifier.run(df_feat)
    check = capital_classifier.self_check(df_feat, pred)
    print(f"[4/5] Task2 资金识别: 类型分布 {pred['capital_type'].value_counts().to_dict()}")
    if "type_weighted_f1" in check:
        print(f"      合成自检 加权F1(类型)={check['type_weighted_f1']:.3f}")
    print("      意图分布:", check.get("intention_dist"))

    # 5) 生成提交
    pp = submit.write_pattern(patt)
    qp = submit.write_predict(pred)
    val = submit.validate(pp, qp)
    if not val["ok"]:
        print("[5/5] ✗ 提交校验失败:", val["issues"])
        raise SystemExit(1)
    zp = submit.pack(pp, qp)
    print(f"[5/5] 提交生成 ✓ {val['rows']} 行 | {zp}")
    print("      ->", pp)
    print("      ->", qp)
    print("完成。")


if __name__ == "__main__":
    main()
