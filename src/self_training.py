"""弱标签自训练：用规则高置信预测作伪标签，训练分类器再回预测全体。

动机：赛题无 ground-truth。规则判别提供先验，但阈值人工。用规则中**高置信**
样本作伪标签训练 logistic/GBDT，让模型从数据分布中学到更平滑的决策边界，
再对全体（含低置信样本）预测，通常比纯阈值更稳。

安全网：样本量不足 / 类别过少 / sklearn 不可用时，优雅回退到规则预测，
保证管线永远可跑、可复现。全程只用行情派生特征，不引用股票代码。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import capital_classifier, config, features

# 触发自训练的最低门槛
MIN_SAMPLES = 30          # 总样本
MIN_PER_CLASS = 5         # 每个伪标签类别最少样本
MIN_PSEUDO = 15           # 伪标签总数下限
CONF_QUANTILE = 0.5       # 取置信度分位数以上作伪标签(top 50%)


def _pseudo_labels(rule_pred: pd.DataFrame) -> pd.Series:
    """从规则预测中挑高置信样本作伪标签，其余为 NaN。"""
    conf = rule_pred["type_confidence"]
    thr = conf.quantile(CONF_QUANTILE)
    labels = rule_pred["capital_type"].where(conf >= thr)
    return labels


def train_predict_type(df_feat: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """返回 (含 capital_type/type_confidence 的结果, 训练元信息)。

    先跑规则，再尝试自训练；不满足条件则返回规则结果。
    """
    rule = capital_classifier.classify_type(df_feat)
    meta = {"method": "rule", "n": len(df_feat)}

    if len(df_feat) < MIN_SAMPLES:
        meta["reason"] = f"样本<{MIN_SAMPLES}, 用规则"
        return rule, meta

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        meta["reason"] = "无 sklearn, 用规则"
        return rule, meta

    pseudo = _pseudo_labels(rule)
    mask = pseudo.notna()
    vc = pseudo[mask].value_counts()
    if mask.sum() < MIN_PSEUDO or (vc < MIN_PER_CLASS).any() or len(vc) < 2:
        meta["reason"] = "高置信伪标签不足, 用规则"
        return rule, meta

    X = features.robust_scale(df_feat, features.MODEL_FEATURE_COLS)
    if X.shape[1] == 0:
        meta["reason"] = "无可用特征, 用规则"
        return rule, meta

    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=config.RANDOM_SEED)
    clf.fit(X[mask.to_numpy()], pseudo[mask].to_numpy())
    proba = clf.predict_proba(X)
    classes = clf.classes_
    pred_idx = proba.argmax(axis=1)

    out = pd.DataFrame(index=df_feat.index)
    out["capital_type"] = classes[pred_idx]
    out["type_confidence"] = proba.max(axis=1).round(4)
    # 与规则不一致且模型置信不高时，保留规则（保守融合）
    low_conf = out["type_confidence"] < 0.5
    out.loc[low_conf, "capital_type"] = rule.loc[low_conf, "capital_type"]
    out.loc[low_conf, "type_confidence"] = rule.loc[low_conf, "type_confidence"]

    agree = float((out["capital_type"].to_numpy() == rule["capital_type"].to_numpy()).mean())
    meta.update({"method": "self_train_logreg", "pseudo_n": int(mask.sum()),
                 "pseudo_dist": vc.to_dict(), "agree_with_rule": round(agree, 3)})
    return out, meta


def run(df_feat: pd.DataFrame, use_self_training: bool = True) -> tuple[pd.DataFrame, dict]:
    """完整 Task2：类型(可自训练) + 意图(规则)。返回 (结果, meta)。"""
    if use_self_training:
        t, meta = train_predict_type(df_feat)
    else:
        t, meta = capital_classifier.classify_type(df_feat), {"method": "rule"}
    i = capital_classifier.classify_intention(df_feat)
    res = pd.concat(
        [df_feat[["symbol", "date"]].reset_index(drop=True),
         t.reset_index(drop=True), i.reset_index(drop=True)],
        axis=1,
    )
    return res, meta
