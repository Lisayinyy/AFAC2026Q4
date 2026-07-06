"""意图-模式一致性校验：让 Task1 交易模式、Task2 买卖意图、行情阶段互相印证。

多路信号（pattern_type / capital_intention / market_phase）本应指向一致的资金动作。
当强模式先验与意图冲突且意图置信不高时，按模式先验温和修正意图，提升整体自洽性。
纯特征/标签驱动，不引用股票代码。
"""
from __future__ import annotations

import pandas as pd

# 模式 → 期望意图先验（用于印证/修正）
PATTERN_INTENT_PRIOR = {
    "大单吸筹": "买入",
    "连续小单推升": "买入",
    "压单吸货": "买入",
    "对倒拉升": "买入",
    "盘中诱多": "卖出",     # 诱多后砸盘出货
    "尾盘突袭": "买入",
    "分时脉冲": "买入",
    "涨停板打开": "卖出",   # 反复打开引诱接盘=派发
    "量化T0": "T0交易",
    "日内套利": "T0交易",
    "集合竞价异动": "买入",
    "散户博弈": "中性",
}

# 意图与模式先验冲突时，若意图置信低于此阈值则采纳模式先验
OVERRIDE_CONF = 0.55


def reconcile(pred: pd.DataFrame, pattern: pd.DataFrame,
              phase: pd.DataFrame | None = None) -> pd.DataFrame:
    """输入 Task2 预测 + Task1 模式(+ 行情阶段)，返回修正后的 pred（含一致性标记）。

    pred: 含 symbol,date,capital_type,capital_intention[,intention_confidence]
    pattern: 含 symbol,date,pattern_type
    """
    key = ["symbol", "date"]
    m = pred.merge(pattern[key + ["pattern_type"]], on=key, how="left")
    if phase is not None:
        m = m.merge(phase[key + ["market_phase"]], on=key, how="left")

    prior = m["pattern_type"].map(PATTERN_INTENT_PRIOR)
    conf = m.get("intention_confidence", pd.Series(1.0, index=m.index)).fillna(1.0)

    consistent = (prior.isna()) | (prior == m["capital_intention"])
    # 冲突 + 低置信 → 采纳模式先验
    override = (~consistent) & (conf < OVERRIDE_CONF) & prior.notna()
    m.loc[override, "capital_intention"] = prior[override]

    m["intent_consistent"] = consistent | override
    m["intent_overridden"] = override
    return m


def consistency_rate(pred: pd.DataFrame, pattern: pd.DataFrame) -> float:
    """报告意图与模式先验的一致率（离线自检）。

    pred 若已含 pattern_type（经 reconcile）则直接用，否则按 (symbol,date) 合并。
    """
    key = ["symbol", "date"]
    if "pattern_type" in pred.columns:
        m = pred
    else:
        m = pred.merge(pattern[key + ["pattern_type"]], on=key, how="left")
    prior = m["pattern_type"].map(PATTERN_INTENT_PRIOR)
    mask = prior.notna()
    if mask.sum() == 0:
        return float("nan")
    return float((prior[mask] == m.loc[mask, "capital_intention"]).mean())
