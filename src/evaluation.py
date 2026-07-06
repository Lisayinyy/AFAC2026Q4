"""离线评估：无 ground-truth 下的代理指标，监控 Task1/Task2 质量与稳定性。

对齐赛题评估口径：
  - Task1 聚类：类间区分度(质心间距)↑、类内聚合度(簇内方差)↓ → 轮廓/CH。
  - Task2 分类：无标签，用判别置信度分布、类型-意图-模式一致率、时序稳定性代理。
若有合成 _true_type 或外部标签，另算加权 F1。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features


def cluster_metrics(df_feat: pd.DataFrame, labels) -> dict:
    from sklearn.metrics import calinski_harabasz_score, silhouette_score

    X = features.robust_scale(df_feat, features.MODEL_FEATURE_COLS)
    out = {"n_clusters": int(len(set(labels)))}
    if len(set(labels)) >= 2 and X.shape[1] > 0 and len(X) > len(set(labels)):
        try:
            out["silhouette"] = round(float(silhouette_score(X, labels)), 4)
            out["calinski_harabasz"] = round(float(calinski_harabasz_score(X, labels)), 2)
        except ValueError:
            pass
    return out


def type_confidence_stats(pred: pd.DataFrame) -> dict:
    out = {}
    if "type_confidence" in pred.columns:
        c = pred["type_confidence"].astype(float)
        out["type_conf_mean"] = round(float(c.mean()), 4)
        out["type_conf_low_frac"] = round(float((c < 0.5).mean()), 4)  # 低置信占比
    out["type_dist"] = pred["capital_type"].value_counts().to_dict()
    out["intention_dist"] = pred["capital_intention"].value_counts().to_dict()
    return out


def temporal_stability(preds_by_date: dict[str, pd.DataFrame]) -> dict:
    """相邻交易日同股预测漂移率（越低越稳）。preds_by_date: {date: pred_df}。"""
    dates = sorted(preds_by_date)
    if len(dates) < 2:
        return {"note": "需≥2个交易日"}
    drifts = []
    for a, b in zip(dates, dates[1:]):
        pa = preds_by_date[a].set_index("symbol")["capital_type"]
        pb = preds_by_date[b].set_index("symbol")["capital_type"]
        common = pa.index.intersection(pb.index)
        if len(common):
            drifts.append(float((pa[common].values != pb[common].values).mean()))
    return {"type_drift_mean": round(float(np.mean(drifts)), 4) if drifts else float("nan"),
            "n_transitions": len(drifts)}


def weighted_f1(y_true, y_pred, labels=None) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="weighted", labels=labels))


def report(df_feat, labels, pred, pattern=None) -> dict:
    """汇总一次运行的离线自检报告。"""
    from . import intention

    r = {"task1": cluster_metrics(df_feat, labels), "task2": type_confidence_stats(pred)}
    if pattern is not None:
        r["intent_pattern_consistency"] = round(intention.consistency_rate(pred, pattern), 4)
    if "_true_type" in df_feat.columns:
        r["task2"]["type_weighted_f1"] = round(
            weighted_f1(df_feat["_true_type"].reset_index(drop=True),
                        pred["capital_type"].reset_index(drop=True)), 4)
    return r
