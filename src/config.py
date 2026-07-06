"""全局配置：路径、超参、可解释映射规则。

所有阈值集中于此，便于迭代校准；判别逻辑本身不依赖股票代码（禁硬编码）。
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# 路径（相对路径，满足赛题代码审核要求）
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
OUTPUT_DIR = os.path.join(ROOT, "output")

PATTERN_FILE = os.path.join(OUTPUT_DIR, "pattern_reco.csv")
PREDICT_FILE = os.path.join(OUTPUT_DIR, "predict_result.csv")
SUBMIT_ZIP = os.path.join(OUTPUT_DIR, "submit.zip")

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 特征分组（前缀即特征族，见 docs/feature_dictionary.md）
# ---------------------------------------------------------------------------
FEATURE_PREFIXES = ["oss_", "rs_", "cb_", "ap_", "obp_", "pd_", "pi_"]
ID_COLS = ["symbol", "date"]

# 用于 Wasserstein「资金结构分布」的字段（订单规模 + 主动买卖分布）
DISTRIBUTION_COLS = [
    "oss_mega_amount_pct", "oss_large_amount_pct",
    "oss_medium_amount_pct", "oss_small_amount_pct",
    "oss_buy_amount_pct", "oss_sell_amount_pct",
]
# 用于 DTW「日内节奏形态」的序列字段
SEQUENCE_COLS = ["net_active_seq", "amount_seq"]

# ---------------------------------------------------------------------------
# Task 1 聚类
# ---------------------------------------------------------------------------
WASSERSTEIN_ALPHA = 0.5          # 综合距离 D = α·W + (1-α)·DTW
CLUSTER_K_RANGE = (3, 8)         # 自动选簇数范围
LINKAGE = "average"

# 模式命名规则：按簇质心画像匹配（阈值为标准化后经验值，可迭代）。
# 词表对齐官方 A 榜提交样例的 pattern_type，每条 = (名称, 说明)。
# _name_cluster 按质心特征挑选最匹配的一条；顺序即优先级。
PATTERN_RULES = [
    ("大单吸筹", "资金大笔挂单买入，短时间内集中扫货"),
    ("对倒拉升", "通过自买自卖制造成交量放大假象，吸引跟风盘"),
    ("压单吸货", "在卖盘挂大单压制股价，同时在低位悄悄吸纳筹码"),
    ("尾盘突袭", "尾盘最后阶段突然放量拉升，制造强势收盘形态"),
    ("集合竞价异动", "在集合竞价阶段大幅拉高或压低，影响开盘价"),
    ("分时脉冲", "短时间内快速拉升后迅速回落，试探上方抛压"),
    ("量化T0", "利用Level2行情高频捕捉盘口价差，自动完成T0回转"),
    ("日内套利", "资金在一定价格区间来回高抛低吸"),
    ("连续小单推升", "以连续小单逐步推高股价，隐蔽建仓"),
    ("盘中诱多", "盘中放量拉升诱导跟风，随后派发出货"),
    ("散户博弈", "成交以小单为主且方向混乱，呈现典型散户博弈特征"),
    ("缩量整理", "全天成交低迷，缺乏主力引导，参与清淡"),
]
# 兜底模式（无明显画像时）
DEFAULT_PATTERN = ("散户博弈", "成交以小单为主且方向混乱，呈现典型散户博弈特征")

# ---------------------------------------------------------------------------
# Task 2 判别阈值
# ---------------------------------------------------------------------------
NET_ACTIVE_TAU = 0.15            # 净主动方向阈值（|买-卖|占比）
UNILATERAL_TAU = 0.55           # 单边强度阈值
T0_BALANCE_TAU = 0.10            # 买卖均衡带宽（判 T0）

# 官方提交样例含三类参与者
CAPITAL_TYPES = ["散户", "游资", "量化"]
INTENTIONS = ["买入", "卖出", "中性", "T0交易"]

# 三类得分权重（领域先验，来自赛题案例；后续可自训练微调）。
# 特征名兼容「参考特征集」与「快照派生特征」两套命名。
QUANT_WEIGHTS = {
    "regularity": 1.0,           # 节奏规整(机器节拍) 1/(1+rs_interval_cv)
    "rs_split_similarity": 0.8,  # 冰山拆单
    "cb_fast_cancel_ratio": 0.7, # 快速撤单试盘
    "t0_balance": 0.6,           # 买卖均衡高换手
    "iceberg": 0.5,              # 拆单强度
}
HOT_MONEY_WEIGHTS = {
    "oss_hot_money_count_pct": 1.0,
    "edge_concentration": 0.9,   # 开盘+收盘集中
    "pi_herfindahl_30min": 0.6,
    "aggression": 0.8,           # 穿价激进
    "oss_mega_amount_pct": 0.7,  # 大单占比
    "manual_irregularity": 0.5,  # 手动间歇
    "big_order_pct": 0.7,        # 快照派生:大单占比
}
# 散户：小单主导、大单占比低、方向混乱、时段分散、无节奏规律
RETAIL_WEIGHTS = {
    "small_order_pct": 1.0,      # 小单占比高
    "low_big_order": 0.9,        # 1 - big_order_pct
    "direction_noise": 0.7,      # 方向混乱(净主动接近0但非量化均衡)
    "low_concentration": 0.6,    # 1 - edge_concentration
    "oss_small_amount_pct": 0.6, # 参考特征集小单占比
}
