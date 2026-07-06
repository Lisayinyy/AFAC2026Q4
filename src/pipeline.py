"""核心管线编排：把已聚合的日级特征跑完 Task1+Task2+阶段+一致性，返回可提交结果。

被 main.py（单次）与 run_batch.py（批量/每日）共用。
"""
from __future__ import annotations

import pandas as pd

from . import (
    capital_classifier,
    evaluation,
    features,
    intention,
    market_phase,
    pattern_clustering,
    self_training,
)


def run_once(df_raw: pd.DataFrame, alpha: float | None = None,
             use_self_training: bool = False, is_daily_features: bool = False) -> dict:
    """完整跑一遍。

    df_raw: 若 is_daily_features=False 则为原始快照(需先聚合)已在外部完成聚合时置 True。
            本函数假定传入的已是**日级特征前身**(含 symbol,date 及原始/派生列)。
    返回 {pattern, predict, phase, report}。
    """
    df_feat = features.build_features(df_raw)

    # Task1
    patt = pattern_clustering.run(df_feat, alpha=alpha)
    # Task2（可自训练）
    if use_self_training:
        pred, tmeta = self_training.run(df_feat, use_self_training=True)
    else:
        pred, tmeta = capital_classifier.run(df_feat), {"method": "rule"}
    # 行情阶段 + 意图一致性校验
    phase = market_phase.run(df_feat)
    pred = intention.reconcile(pred, patt, phase)
    # 离线自检
    report = evaluation.report(df_feat, patt["cluster"].to_numpy(), pred, patt)
    report["task2"]["method"] = tmeta.get("method")

    return {"features": df_feat, "pattern": patt, "predict": pred,
            "phase": phase, "report": report, "train_meta": tmeta}
