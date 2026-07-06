"""Task 1 —— 交易模式识别（聚类 + 可解释命名）。

流程：分布/序列特征 → 综合距离(Wasserstein+DTW) → 层次聚类(自动选簇数)
      → 按簇质心画像映射到可读 pattern_type + pattern_explanation。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, distance, features


def _select_k(dist_matrix: np.ndarray, k_range: tuple[int, int]) -> int:
    """用轮廓系数在候选簇数中选优（预计算距离矩阵）。"""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = len(dist_matrix)
    lo, hi = k_range
    hi = min(hi, n - 1)
    best_k, best_s = max(lo, 2), -1.0
    for k in range(max(lo, 2), max(lo, 2) + 1) if hi < lo else range(lo, hi + 1):
        if k >= n:
            break
        labels = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage=config.LINKAGE
        ).fit_predict(dist_matrix)
        if len(set(labels)) < 2:
            continue
        try:
            s = silhouette_score(dist_matrix, labels, metric="precomputed")
        except ValueError:
            continue
        if s > best_s:
            best_s, best_k = s, k
    return best_k


def _name_cluster(centroid: dict[str, float]) -> tuple[str, str]:
    """按质心特征画像映射语义标签（规则驱动，不依赖股票代码）。"""
    g = centroid.get
    net = g("net_active", 0.0)
    iceberg = g("iceberg", 0.0)
    impact = g("pi_max_price_impact_pct", 0.0)
    edge = g("edge_concentration", 0.0)
    spoof = g("spoof", 0.0)
    balance = g("balance", 0.0)
    regularity = g("regularity", 0.0)
    deal_amount = g("deal_amount", 0.0)

    if net > 0.15 and iceberg > 0.3 and impact < 1.0:
        return config.PATTERN_RULES[0]           # 大单吸筹
    if balance > 0.8 and regularity > 0.6:
        return config.PATTERN_RULES[1]           # 日内套利(T0)
    if edge > 0.6 and impact > 1.0:
        return config.PATTERN_RULES[2]           # 尾盘突袭
    if spoof > 0.25:
        return config.PATTERN_RULES[3]           # 盘口博弈
    if net < -0.15:
        return config.PATTERN_RULES[4]           # 派发出货
    return config.PATTERN_RULES[5]               # 缩量整理


def run(df_feat: pd.DataFrame, alpha: float | None = None) -> pd.DataFrame:
    """输入已 build_features 的 DataFrame，返回含 pattern_type/explanation 的结果。

    Returns 列: symbol, date, cluster, pattern_type, pattern_explanation
    """
    alpha = config.WASSERSTEIN_ALPHA if alpha is None else alpha
    n = len(df_feat)

    dist_cols = [c for c in config.DISTRIBUTION_COLS if c in df_feat.columns]
    dist_features = (
        df_feat[dist_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        if dist_cols else np.zeros((n, 1))
    )
    if "net_active_seq" in df_feat.columns:
        sequences = features.parse_sequence(df_feat["net_active_seq"])
    else:
        sequences = [np.array([df_feat.iloc[i].get("net_active", 0.0)]) for i in range(n)]

    if n < 3:
        labels = np.zeros(n, dtype=int)
    else:
        D = distance.combined_distance(dist_features, sequences, alpha=alpha)
        k = _select_k(D, config.CLUSTER_K_RANGE)
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage=config.LINKAGE
        ).fit_predict(D)

    df = df_feat.copy()
    df["cluster"] = labels

    # 每簇质心画像 → 命名
    name_map: dict[int, tuple[str, str]] = {}
    numeric = df.select_dtypes(include=[np.number])
    for c in sorted(set(labels)):
        centroid = numeric[df["cluster"] == c].mean(numeric_only=True).to_dict()
        name_map[c] = _name_cluster(centroid)

    df["pattern_type"] = df["cluster"].map(lambda c: name_map[c][0])
    df["pattern_explanation"] = df["cluster"].map(lambda c: name_map[c][1])
    return df[["symbol", "date", "cluster", "pattern_type", "pattern_explanation"]]


def quality_metrics(df_feat: pd.DataFrame, labels: np.ndarray) -> dict:
    """离线自检：轮廓系数 / CH 指数（越大越好）。"""
    from sklearn.metrics import calinski_harabasz_score, silhouette_score

    X = features.robust_scale(df_feat, features.MODEL_FEATURE_COLS)
    out = {}
    if len(set(labels)) >= 2 and X.shape[1] > 0:
        try:
            out["silhouette"] = float(silhouette_score(X, labels))
            out["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
        except ValueError:
            pass
    out["n_clusters"] = int(len(set(labels)))
    return out
