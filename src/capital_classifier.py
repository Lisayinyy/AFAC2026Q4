"""Task 2 —— 资金类型（游资/量化）+ 买卖意图识别。

无 ground-truth 标签，采用领域先验加权评分 + 规则意图树，全部由 Level-2
派生特征驱动（禁硬编码）。输出附置信度供内部复核。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def _weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    wsum = 0.0
    for feat, w in weights.items():
        if feat in df.columns:
            score = score + w * _minmax(pd.to_numeric(df[feat], errors="coerce").fillna(0.0))
            wsum += w
    return score / wsum if wsum > 0 else score


def classify_type(df: pd.DataFrame) -> pd.DataFrame:
    """输出 capital_type + type_confidence。"""
    quant = _weighted_score(df, config.QUANT_WEIGHTS)
    hot = _weighted_score(df, config.HOT_MONEY_WEIGHTS)

    out = pd.DataFrame(index=df.index)
    out["quant_score"] = quant
    out["hot_money_score"] = hot
    # softmax 间隔作为置信度
    exp_q, exp_h = np.exp(quant), np.exp(hot)
    p_hot = exp_h / (exp_q + exp_h)
    out["capital_type"] = np.where(hot > quant, "游资", "量化")
    out["type_confidence"] = np.where(hot > quant, p_hot, 1 - p_hot).round(4)
    return out


def classify_intention(df: pd.DataFrame) -> pd.DataFrame:
    """输出 capital_intention + intention_confidence。"""
    net = pd.to_numeric(df.get("net_active", pd.Series(0.0, index=df.index)),
                        errors="coerce").fillna(0.0)
    uni = pd.to_numeric(df.get("ap_unilateral_intensity", pd.Series(0.0, index=df.index)),
                        errors="coerce").fillna(0.0)
    balance = pd.to_numeric(df.get("balance", pd.Series(0.0, index=df.index)),
                            errors="coerce").fillna(0.0)
    regularity = pd.to_numeric(df.get("regularity", pd.Series(0.0, index=df.index)),
                               errors="coerce").fillna(0.0)

    tau, u_tau, b_tau = config.NET_ACTIVE_TAU, config.UNILATERAL_TAU, config.T0_BALANCE_TAU
    intention = pd.Series("中性", index=df.index)
    # T0：买卖均衡 + 量化节奏
    t0_mask = (net.abs() <= b_tau) & (regularity > 0.6)
    intention[t0_mask] = "T0交易"
    # 买入 / 卖出
    buy_mask = (~t0_mask) & (net > tau)
    sell_mask = (~t0_mask) & (net < -tau)
    intention[buy_mask] = "买入"
    intention[sell_mask] = "卖出"

    conf = (net.abs().clip(0, 1) * 0.6 + uni.clip(0, 1) * 0.4).round(4)
    conf[t0_mask] = (balance[t0_mask] * 0.6 + 0.4).round(4)

    out = pd.DataFrame(index=df.index)
    out["capital_intention"] = intention
    out["intention_confidence"] = conf
    return out


def run(df_feat: pd.DataFrame) -> pd.DataFrame:
    """返回含 symbol,date,capital_type,capital_intention 及置信度的结果。"""
    t = classify_type(df_feat)
    i = classify_intention(df_feat)
    res = pd.concat(
        [df_feat[["symbol", "date"]].reset_index(drop=True),
         t.reset_index(drop=True), i.reset_index(drop=True)],
        axis=1,
    )
    return res


def self_check(df_feat: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """若合成数据含 _true_type，报告加权 F1 作为管线自检。"""
    out = {}
    if "_true_type" in df_feat.columns:
        from sklearn.metrics import f1_score

        y_true = df_feat["_true_type"].reset_index(drop=True)
        y_pred = pred["capital_type"].reset_index(drop=True)
        out["type_weighted_f1"] = float(
            f1_score(y_true, y_pred, average="weighted", labels=config.CAPITAL_TYPES)
        )
    out["intention_dist"] = pred["capital_intention"].value_counts().to_dict()
    return out
