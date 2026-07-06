"""距离计算：Wasserstein（资金结构分布）+ DTW（日内节奏形态）。

赛题 Task1 指定「综合考量 Wasserstein 距离和 DTW 距离」。本模块提供两者的
成对距离矩阵及归一化综合，供聚类使用。DTW 为纯 numpy 实现，无额外重依赖。
"""
from __future__ import annotations

import numpy as np


def _wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    """两个「分布向量」间的 1-Wasserstein 距离。

    把非负向量归一化为概率分布，按累积分布差的 L1 近似 EMD（等权支撑点）。
    """
    u = np.clip(np.asarray(u, dtype=float), 0, None)
    v = np.clip(np.asarray(v, dtype=float), 0, None)
    su, sv = u.sum(), v.sum()
    if su <= 0 or sv <= 0:
        return float(np.abs(u - v).sum())
    u, v = u / su, v / sv
    return float(np.abs(np.cumsum(u) - np.cumsum(v)).sum())


def wasserstein_matrix(dist_features: np.ndarray) -> np.ndarray:
    """dist_features: (n, d) 每行是一个样本的分布型特征向量。"""
    n = len(dist_features)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _wasserstein_1d(dist_features[i], dist_features[j])
            M[i, j] = M[j, i] = d
    return M


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    """经典 DTW（欧氏局部代价），O(len(a)*len(b))。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))  # 长度归一化


def dtw_matrix(sequences: list[np.ndarray]) -> np.ndarray:
    n = len(sequences)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _dtw(sequences[i], sequences[j])
            M[i, j] = M[j, i] = d
    return M


def _normalize(M: np.ndarray) -> np.ndarray:
    mx = M.max()
    return M / mx if mx > 0 else M


def combined_distance(
    dist_features: np.ndarray,
    sequences: list[np.ndarray],
    alpha: float = 0.5,
) -> np.ndarray:
    """D = α·Wasserstein_norm + (1-α)·DTW_norm。"""
    W = _normalize(wasserstein_matrix(dist_features))
    T = _normalize(dtw_matrix(sequences))
    return alpha * W + (1 - alpha) * T
