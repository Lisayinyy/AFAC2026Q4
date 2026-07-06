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

# 模式命名规则：按簇质心画像匹配（阈值为标准化后经验值，可迭代）
# 每条规则 = (名称, 说明, 条件函数键)，条件在 pattern_clustering 中按质心特征评估
PATTERN_RULES = [
    ("大单吸筹", "资金大笔挂单分批买入，隐蔽建仓"),
    ("日内套利", "资金在一定价格区间来回高抛低吸(T0)"),
    ("尾盘突袭", "特定时段集中拉升/砸盘，制造形态"),
    ("盘口博弈", "反复挂撤试探盘口深度"),
    ("派发出货", "主动砸盘，连续吃对手盘买单"),
    ("缩量整理", "参与清淡，无明显主导资金"),
]

# ---------------------------------------------------------------------------
# Task 2 判别阈值
# ---------------------------------------------------------------------------
NET_ACTIVE_TAU = 0.15            # 净主动方向阈值（|买-卖|占比）
UNILATERAL_TAU = 0.55           # 单边强度阈值
T0_BALANCE_TAU = 0.10            # 买卖均衡带宽（判 T0）

CAPITAL_TYPES = ["游资", "量化"]
INTENTIONS = ["买入", "卖出", "中性", "T0交易"]

# 量化 / 游资 得分权重（领域先验，来自赛题案例；后续可自训练微调）
QUANT_WEIGHTS = {
    "regularity": 1.0,           # 1/(1+rs_interval_cv)
    "rs_split_similarity": 0.8,  # 冰山拆单
    "cb_fast_cancel_ratio": 0.7, # 快速撤单试盘
    "t0_balance": 0.6,           # 买卖均衡高换手
}
HOT_MONEY_WEIGHTS = {
    "oss_hot_money_count_pct": 1.0,
    "edge_concentration": 0.8,   # 开盘+收盘集中
    "pi_herfindahl_30min": 0.6,
    "aggression": 0.7,           # 穿价激进
    "oss_mega_amount_pct": 0.5,  # 大单占比
    "manual_irregularity": 0.6,  # rs_interval_cv 高（手动间歇）
}
