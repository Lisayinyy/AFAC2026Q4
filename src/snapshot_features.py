"""十档盘口快照 → 日级特征提取器。

输入为官方赛题一训练数据格式（每行一个快照 tick）：
  symbol, dt(交易日), date(epoch ms), hh, price, volume(累计), amount(累计),
  transactions(累计), totalbidvolume, totalaskvolume, bigordervolume,
  bids/asks (10 档 JSON，每档 {price, volume, order:[{volume}...], bigOrderPercent})

按 (symbol, dt) 聚合为日级特征，输出列尽量对齐 features.MODEL_FEATURE_COLS，
使下游 Task1/Task2 无需区分数据来源。所有特征由行情派生，不引用股票代码。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

EPS = 1e-9


def _parse_levels(raw) -> list[dict]:
    """解析 bids/asks JSON 字符串为 [{price, volume, orders:[...], big_pct}]。"""
    if not isinstance(raw, str) or not raw.strip() or raw.strip() == "[]":
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out = []
    for lvl in data:
        if not isinstance(lvl, dict):
            continue
        orders = lvl.get("order") or []
        out.append({
            "price": float(lvl.get("price", 0) or 0),
            "volume": float(lvl.get("volume", 0) or 0),
            "orders": [float(o.get("volume", 0) or 0) for o in orders if isinstance(o, dict)],
            "big_pct": float(lvl.get("bigOrderPercent", 0) or 0),
        })
    return out


def _snapshot_row_features(row: pd.Series) -> dict:
    """单个快照的盘口结构特征。"""
    bids = _parse_levels(row.get("bids"))
    asks = _parse_levels(row.get("asks"))
    f = {}

    tbv = float(row.get("totalbidvolume", 0) or 0)
    tav = float(row.get("totalaskvolume", 0) or 0)
    f["ob_imbalance"] = (tbv - tav) / (tbv + tav + EPS)     # >0 买方深度占优

    # 价差（以一档价）
    best_bid = bids[0]["price"] if bids else 0.0
    best_ask = asks[0]["price"] if asks else 0.0
    mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else float(row.get("price", 0) or 0)
    f["spread"] = (best_ask - best_bid) / (mid + EPS) if (best_bid and best_ask) else 0.0

    # 一档拆单数（order 数组长度）→ 冰山/拆单证据
    bid_orders = bids[0]["orders"] if bids else []
    ask_orders = asks[0]["orders"] if asks else []
    f["l1_split_count"] = len(bid_orders) + len(ask_orders)
    # 一档大单占比
    f["l1_big_pct"] = (bids[0]["big_pct"] if bids else 0.0) + (asks[0]["big_pct"] if asks else 0.0)
    return f


def _minute_of_day(epoch_ms) -> float:
    """epoch ms → 当日分钟数（本地/交易所时区近似，用于开收盘集中度）。"""
    try:
        ts = int(epoch_ms)
    except (TypeError, ValueError):
        return np.nan
    # 转到东八区分钟：ms → s，+8h，取当日分钟
    sec = ts / 1000.0 + 8 * 3600
    return (sec % 86400) / 60.0


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """把快照序列按 (symbol, dt) 聚合成日级特征 DataFrame。"""
    if "dt" in df.columns:
        date_col = "dt"
    elif "tradedate" in df.columns:
        date_col = "tradedate"
    else:
        date_col = "date"

    rows = []
    for (sym, day), g in df.groupby(["symbol", date_col]):
        g = g.sort_values("date") if "date" in g.columns else g
        n = len(g)

        # 逐快照盘口特征
        snap = g.apply(_snapshot_row_features, axis=1, result_type="expand")

        # 累计量 → 增量（volume/amount/transactions 为累计值，做 diff 得分时流量）
        vol = pd.to_numeric(g.get("volume"), errors="coerce").ffill().fillna(0)
        amt = pd.to_numeric(g.get("amount"), errors="coerce").ffill().fillna(0)
        price = pd.to_numeric(g.get("price"), errors="coerce").ffill()
        dvol = vol.diff().clip(lower=0).fillna(0)
        damt = amt.diff().clip(lower=0).fillna(0)

        # 时段集中度（开盘30min / 收盘10min，A股 9:30-11:30,13:00-15:00）
        mins = g["date"].apply(_minute_of_day) if "date" in g.columns else pd.Series(np.nan, index=g.index)
        open_mask = (mins >= 9 * 60 + 30) & (mins < 10 * 60)          # 9:30-10:00
        close_mask = (mins >= 14 * 60 + 50) & (mins <= 15 * 60)       # 14:50-15:00
        total_amt = damt.sum() + EPS
        open_pct = damt[open_mask.values].sum() / total_amt if open_mask.any() else 0.0
        close_pct = damt[close_mask.values].sum() / total_amt if close_mask.any() else 0.0

        # 真实日内序列（供 DTW）：把连续交易时段切成 N 桶，每桶取盘口不平衡均值。
        # A股连续竞价 9:30-11:30 + 13:00-15:00 = 240 分钟，切 8 桶(每桶30min)。
        imbalance_series = snap["ob_imbalance"] if "ob_imbalance" in snap else pd.Series(0.0, index=g.index)
        intraday_seq = _bin_intraday(mins, imbalance_series, n_bins=8)

        # 价格路径特征
        ret = price.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        px_std = float(ret.std() * 100)
        max_impact = float((price.max() - price.min()) / (price.mean() + EPS) * 100)
        vwap = (price * dvol).sum() / (dvol.sum() + EPS)
        vwap_dev = float(abs(price.mean() - vwap) / (vwap + EPS))

        # 成交集中度（赫芬达尔，按分钟聚合成交额）
        if "date" in g.columns:
            minute_bin = (mins // 30).fillna(-1)
            amt_by_bin = damt.groupby(minute_bin.values).sum()
            shares = amt_by_bin / (amt_by_bin.sum() + EPS)
            herf = float((shares ** 2).sum())
        else:
            herf = 0.0

        # 盘口订单拆分/大单
        l1_split = float(snap["l1_split_count"].mean()) if "l1_split_count" in snap else 0.0
        big_pct = float(snap["l1_big_pct"].mean()) if "l1_big_pct" in snap else 0.0
        big_pct = big_pct / 100 if big_pct > 1 else big_pct
        imbalance = float(snap["ob_imbalance"].mean()) if "ob_imbalance" in snap else 0.0

        # 快照节奏规整度：快照间隔的变异系数（近似成交节奏）
        if "date" in g.columns and n > 2:
            iv = g["date"].astype("int64").diff().dropna()
            interval_cv = float(iv.std() / (iv.mean() + EPS)) if iv.mean() > 0 else 0.0
        else:
            interval_cv = 0.0

        # bigordervolume 相对总量占比（大单参与）
        bov = pd.to_numeric(g.get("bigordervolume"), errors="coerce").fillna(0)
        big_order_share = float(bov.mean() / (vol.iloc[-1] / max(n, 1) + EPS)) if len(vol) else 0.0
        big_order_share = float(np.clip(big_order_share, 0, 1))

        # 订单规模结构近似：用一档单笔均量分档（小/中/大）
        avg_order = _avg_order_size(snap)

        net_active = imbalance  # 盘口不平衡近似净主动方向
        regularity = 1.0 / (1.0 + interval_cv)
        iceberg = float(np.clip(l1_split / 6.0, 0, 1)) * (1 if big_pct > 0.3 else 0.6)

        rows.append({
            "symbol": str(sym),
            "date": str(day),
            # 下游 MODEL_FEATURE_COLS 兼容字段
            "regularity": regularity,
            "manual_irregularity": interval_cv / (1 + interval_cv),
            "iceberg": iceberg,
            "spoof": 0.0,                         # 快照无撤单信息，置 0
            "aggression": float(snap.get("spread", pd.Series([0])).mean()) if "spread" in snap else 0.0,
            "net_active": net_active,
            "balance": 1.0 - abs(net_active),
            "edge_concentration": float(open_pct + close_pct),
            "impact_per_amount": max_impact / (total_amt / 1e8 + EPS),
            "t0_balance": (1.0 - abs(net_active)) * regularity,
            "rs_interval_cv": interval_cv,
            "rs_split_similarity": iceberg,
            "cb_fast_cancel_ratio": 0.0,
            "oss_hot_money_count_pct": big_order_share,
            "oss_mega_amount_pct": big_pct,
            "oss_small_amount_pct": float(np.clip(1 - big_pct - big_order_share, 0, 1)),
            "pi_herfindahl_30min": herf,
            "pi_max_price_impact_pct": max_impact,
            "ap_unilateral_intensity": abs(net_active),
            # 三类判别专用派生
            "big_order_pct": big_pct,
            "small_order_pct": float(np.clip(1 - big_pct, 0, 1)),
            "low_big_order": float(np.clip(1 - big_pct, 0, 1)),
            "low_concentration": float(np.clip(1 - (open_pct + close_pct), 0, 1)),
            "direction_noise": float(np.clip(1 - abs(net_active), 0, 1)) * (0.5 if regularity > 0.8 else 1.0),
            # 分布/序列（供 Task1 距离）
            "oss_buy_amount_pct": float(np.clip(0.5 + net_active / 2, 0.02, 0.98)),
            "oss_sell_amount_pct": float(np.clip(0.5 - net_active / 2, 0.02, 0.98)),
            "oss_large_amount_pct": big_pct * 0.6,
            "oss_medium_amount_pct": float(np.clip(1 - big_pct - (1 - big_pct) * 0.6, 0, 1)),
            "px_std_pct": px_std,
            "vwap_dev": vwap_dev,
            "n_snapshots": n,
            # 真实日内盘口不平衡序列（供 Task1 DTW 形态比较）
            "net_active_seq": ";".join(f"{x:.4f}" for x in intraday_seq),
        })

    return pd.DataFrame(rows)


# A股连续竞价分钟区间：9:30-11:30 (570-690) + 13:00-15:00 (780-900)
_TRADING_MINUTES = list(range(9 * 60 + 30, 11 * 60 + 30)) + list(range(13 * 60, 15 * 60))


def _bin_intraday(mins: pd.Series, values: pd.Series, n_bins: int = 8) -> list[float]:
    """把连续交易时段按分钟切成 n_bins 桶，每桶取 values 均值，构成日内序列。"""
    edges = np.linspace(0, len(_TRADING_MINUTES), n_bins + 1).astype(int)
    min_to_bin = {}
    for b in range(n_bins):
        for idx in range(edges[b], edges[b + 1]):
            min_to_bin[_TRADING_MINUTES[idx]] = b

    sums = np.zeros(n_bins)
    cnts = np.zeros(n_bins)
    for m, v in zip(mins.to_numpy(), values.to_numpy()):
        if np.isnan(m):
            continue
        b = min_to_bin.get(int(m))
        if b is not None:
            sums[b] += float(v)
            cnts[b] += 1
    seq = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)
    return [round(float(x), 4) for x in seq]


def _avg_order_size(snap: pd.DataFrame) -> float:
    return 0.0  # 预留：可从 order 数组均量估计，当前未用于下游


def build_from_snapshot(df_raw: pd.DataFrame) -> pd.DataFrame:
    """外部入口：原始快照 DataFrame → 日级特征。"""
    df_raw = df_raw.copy()
    df_raw["symbol"] = df_raw["symbol"].astype(str)
    return aggregate_daily(df_raw)
