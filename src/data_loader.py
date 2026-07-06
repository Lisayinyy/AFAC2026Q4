"""数据加载：优先读官方参考特征集；无数据时生成合成数据保证管线可运行。

合成数据刻意模拟「游资 / 量化」两类微观结构，用于验证管线与自检指标，
不参与正式提交（正式提交需放入 data/sample 的真实样例集）。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from . import config


def load_feature_set(sample_dir: str | None = None) -> tuple[pd.DataFrame, bool]:
    """加载参考特征集。

    Returns
    -------
    (df, is_synthetic)
        df: 至少含 config.ID_COLS 及若干特征列的 DataFrame。
        is_synthetic: 是否为合成兜底数据。
    """
    sample_dir = sample_dir or config.SAMPLE_DIR
    files = []
    if os.path.isdir(sample_dir):
        files = sorted(glob.glob(os.path.join(sample_dir, "*.csv")))

    if files:
        frames = [pd.read_csv(f) for f in files]
        df = pd.concat(frames, ignore_index=True)
        df = _normalize_id_cols(df)
        return df, False

    return generate_synthetic(), True


def load_snapshot(path: str) -> pd.DataFrame:
    """加载原始十档快照数据并聚合为日级特征（委托 adapters.XlsxSource）。

    path 可为单个文件 (xlsx/xls/csv) 或目录（合并其中所有快照文件）。
    返回已 build_from_snapshot 的日级特征 DataFrame。
    """
    from . import snapshot_features
    from .adapters import XlsxSource

    raw = XlsxSource(path).load()
    raw = _normalize_id_cols(raw)
    return snapshot_features.build_from_snapshot(raw)


def _normalize_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    """兼容官方可能的列名差异（stock_code/transaction_date）。"""
    rename = {}
    if "symbol" not in df.columns:
        for cand in ("stock_code", "code", "ts_code"):
            if cand in df.columns:
                rename[cand] = "symbol"
                break
    if "date" not in df.columns:
        for cand in ("transaction_date", "trade_date", "trading_date"):
            if cand in df.columns:
                rename[cand] = "date"
                break
    if rename:
        df = df.rename(columns=rename)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.replace("-", "", regex=False)
    return df


def generate_synthetic(n_stocks: int = 40, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """生成模拟游资/量化两类结构的日级特征。"""
    rng = np.random.default_rng(seed)
    rows = []
    date = "20260507"
    for i in range(n_stocks):
        is_quant = i % 2 == 0
        symbol = f"6{rng.integers(0, 99999):05d}"
        if is_quant:
            rs_interval_cv = rng.uniform(0.05, 0.25)          # 机器节拍，低变异
            rs_split_similarity = rng.uniform(0.6, 0.95)      # 冰山拆单
            cb_fast_cancel_ratio = rng.uniform(0.4, 0.8)      # 快速撤单
            oss_hot_money_count_pct = rng.uniform(0.0, 0.15)
            edge = rng.uniform(0.1, 0.35)
            herf = rng.uniform(0.05, 0.2)
            aggression = rng.uniform(0.05, 0.3)
            oss_mega = rng.uniform(0.05, 0.2)
            net = rng.uniform(-0.08, 0.08)                    # 买卖均衡 → T0
        else:
            rs_interval_cv = rng.uniform(0.6, 1.6)            # 手动间歇，高变异
            rs_split_similarity = rng.uniform(0.1, 0.45)
            cb_fast_cancel_ratio = rng.uniform(0.05, 0.35)
            oss_hot_money_count_pct = rng.uniform(0.25, 0.7)
            edge = rng.uniform(0.45, 0.8)                     # 开收盘集中
            herf = rng.uniform(0.3, 0.7)
            aggression = rng.uniform(0.4, 0.9)                # 激进穿价
            oss_mega = rng.uniform(0.3, 0.65)
            net = rng.choice([rng.uniform(0.2, 0.6), rng.uniform(-0.6, -0.2)])

        buy = 0.5 + net / 2
        buy = float(np.clip(buy, 0.02, 0.98))
        sell = 1 - buy
        # 分窗口序列（用于 DTW）
        k = 8
        base = np.linspace(0, 1, k)
        net_seq = (net + rng.normal(0, 0.05, k)).round(4)
        amount_seq = (np.abs(rng.normal(1, 0.3, k)) * (2 + 3 * edge)).round(4)

        rows.append({
            "symbol": symbol,
            "date": date,
            "order_count": int(rng.integers(500, 5000)),
            "deal_count": int(rng.integers(300, 3000)),
            "deal_amount": float(rng.uniform(5e7, 5e8)),
            "rs_interval_cv": rs_interval_cv,
            "rs_split_similarity": rs_split_similarity,
            "rs_split_run_ratio": rng.uniform(0.1, 0.9),
            "cb_fast_cancel_ratio": cb_fast_cancel_ratio,
            "cb_cancel_order_ratio": rng.uniform(0.1, 0.75),
            "oss_hot_money_count_pct": oss_hot_money_count_pct,
            "oss_mega_amount_pct": oss_mega,
            "oss_large_amount_pct": rng.uniform(0.1, 0.3),
            "oss_medium_amount_pct": rng.uniform(0.1, 0.3),
            "oss_small_amount_pct": rng.uniform(0.05, 0.25),
            "oss_buy_amount_pct": buy,
            "oss_sell_amount_pct": sell,
            "ap_active_buy_pct": buy,
            "ap_active_sell_pct": sell,
            "ap_unilateral_intensity": abs(net) + rng.uniform(0, 0.3),
            "ap_active_buy_run_max": int(rng.integers(1, 30)),
            "ap_active_sell_run_max": int(rng.integers(1, 30)),
            "obp_cross_spread_buy": aggression * buy,
            "obp_cross_spread_sell": aggression * sell,
            "pi_max_price_impact_pct": rng.uniform(0.1, 3.0) * (1 + aggression),
            "pi_herfindahl_30min": herf,
            "pi_open_30min_amount_pct": edge * rng.uniform(0.4, 0.6),
            "pi_close_10min_amount_pct": edge * rng.uniform(0.4, 0.6),
            "net_active_seq": ";".join(map(str, net_seq)),
            "amount_seq": ";".join(map(str, amount_seq)),
            "_true_type": "量化" if is_quant else "游资",   # 仅合成自检用
        })
    return pd.DataFrame(rows)
