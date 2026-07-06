"""行情阶段识别：把每 (股票,日) 的盘面归入一个资金运作阶段。

阶段词表（吸筹→拉升→派发→整理 的资金生命周期视角），由行情派生特征判定，
供 Task2 意图印证与最终解读使用。不引用股票代码。
"""
from __future__ import annotations

import pandas as pd

PHASES = ["吸筹", "试盘", "拉升", "派发", "整理"]


def _phase_row(f: dict) -> tuple[str, float]:
    g = f.get
    net = g("net_active", 0.0)
    iceberg = g("iceberg", 0.0)
    impact = g("pi_max_price_impact_pct", 0.0)
    big = g("big_order_pct", g("oss_mega_amount_pct", 0.0))
    edge = g("edge_concentration", 0.0)
    spoof = g("spoof", 0.0)
    fast_cancel = g("cb_fast_cancel_ratio", 0.0)

    # 拉升：净买 + 高冲击
    if net > 0.1 and impact > 2.0:
        return "拉升", min(1.0, 0.5 + impact / 10)
    # 派发：净卖 + 高冲击/激进
    if net < -0.1 and impact > 1.5:
        return "派发", min(1.0, 0.5 + abs(net))
    # 吸筹：净买 + 冰山/大单 + 低冲击（隐蔽建仓）
    if net > 0.05 and (iceberg > 0.4 or big > 0.3) and impact < 2.0:
        return "吸筹", min(1.0, 0.5 + iceberg)
    # 试盘：高快速撤单/幌骗 + 小幅
    if (spoof > 0.2 or fast_cancel > 0.4) and impact < 1.5:
        return "试盘", 0.6
    # 其余：整理
    return "整理", 0.5


def run(df_feat: pd.DataFrame) -> pd.DataFrame:
    """返回 symbol,date,market_phase,phase_confidence。"""
    recs = df_feat.to_dict("records")
    phases = [_phase_row(r) for r in recs]
    out = df_feat[["symbol", "date"]].copy().reset_index(drop=True)
    out["market_phase"] = [p[0] for p in phases]
    out["phase_confidence"] = [round(p[1], 4) for p in phases]
    return out
