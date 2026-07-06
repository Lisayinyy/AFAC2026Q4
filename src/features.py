"""特征工程：在参考特征集上做二次加工，构造更强的判别量。

所有派生特征均由 Level-2 派生字段计算，不引用股票代码（禁硬编码）。
对缺失字段做鲁棒填充，兼容不同数据源。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

EPS = 1e-9


def _get(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """返回带派生特征列的 DataFrame（原列保留）。

    若 df 已含全部派生特征（如来自 snapshot_features 的日级特征），直接返回，
    避免用缺失的原始字段把已算好的派生特征覆盖为默认值。
    """
    derived = {"regularity", "iceberg", "net_active", "edge_concentration", "t0_balance"}
    if derived.issubset(df.columns):
        return df.copy()

    out = df.copy()

    rs_cv = _get(out, "rs_interval_cv")
    out["regularity"] = 1.0 / (1.0 + rs_cv)                    # 越高越像机器
    out["manual_irregularity"] = rs_cv / (1.0 + rs_cv)         # 越高越像手动

    out["iceberg"] = _get(out, "rs_split_similarity") * _get(out, "rs_split_run_ratio")
    out["spoof"] = _get(out, "cb_fast_cancel_ratio") * _get(out, "cb_cancel_order_ratio")
    out["aggression"] = _get(out, "obp_cross_spread_buy") + _get(out, "obp_cross_spread_sell")

    buy = _get(out, "ap_active_buy_pct")
    sell = _get(out, "ap_active_sell_pct")
    # 若无 ap_*，回退到 oss_ 买卖占比
    if "ap_active_buy_pct" not in out.columns:
        buy = _get(out, "oss_buy_amount_pct", 0.5)
        sell = _get(out, "oss_sell_amount_pct", 0.5)
    out["net_active"] = buy - sell
    out["balance"] = 1.0 - (buy - sell).abs()                 # 越高越均衡(T0)

    out["edge_concentration"] = (
        _get(out, "pi_open_30min_amount_pct") + _get(out, "pi_close_10min_amount_pct")
    )
    out["impact_per_amount"] = _get(out, "pi_max_price_impact_pct") / (
        _get(out, "deal_amount") / 1e8 + EPS
    )

    # T0 均衡 + 高换手（用成交/委托比近似换手活跃度）
    turnover = _get(out, "deal_count") / (_get(out, "order_count") + EPS)
    out["t0_balance"] = out["balance"] * np.clip(turnover, 0, 2)

    return out


def robust_scale(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """RobustScaler（抗高频厚尾），返回标准化矩阵。"""
    from sklearn.preprocessing import RobustScaler

    present = [c for c in cols if c in df.columns]
    mat = df[present].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    if mat.shape[1] == 0:
        return np.zeros((len(df), 0))
    return RobustScaler().fit_transform(mat)


def parse_sequence(series: pd.Series) -> list[np.ndarray]:
    """把 'a;b;c' 字符串序列解析为 float 数组列表（用于 DTW）。"""
    seqs = []
    for v in series:
        if isinstance(v, str) and v:
            seqs.append(np.array([float(x) for x in v.split(";") if x != ""]))
        elif isinstance(v, (list, tuple, np.ndarray)):
            seqs.append(np.asarray(v, dtype=float))
        else:
            seqs.append(np.array([0.0]))
    return seqs


# 进入下游模型的数值特征列（派生 + 关键原始）
MODEL_FEATURE_COLS = [
    "regularity", "manual_irregularity", "iceberg", "spoof", "aggression",
    "net_active", "balance", "edge_concentration", "impact_per_amount", "t0_balance",
    "rs_interval_cv", "rs_split_similarity", "cb_fast_cancel_ratio",
    "oss_hot_money_count_pct", "oss_mega_amount_pct",
    "pi_herfindahl_30min", "pi_max_price_impact_pct",
    "ap_unilateral_intensity",
]
