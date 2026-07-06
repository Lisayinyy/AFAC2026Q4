# 参考特征集字段字典

官方基于 Level-2 数据（逐笔委托/成交/撤单/十档盘口）构建的参考特征集。每行对应「某股票 × 某时间窗口」的聚合特征。字段按前缀分组，前缀即特征族，是建模时的天然分组。

## 标识字段

| 字段名 | 说明 |
|---|---|
| `date` | 日期 |
| `symbol` | 股票代码 |
| `window_start` / `window_end` | 窗口开始 / 结束时间 |
| `window_start_dt` / `window_end_dt` | 窗口起止日期时间（精确） |

## 计数与金额（订单/成交规模）

| 字段名 | 说明 |
|---|---|
| `order_count` | 订单数量（委托笔数） |
| `order_count_all` | 所有订单数量（含撤单） |
| `cancel_count_all` | 撤单总笔数 |
| `deal_count` | 成交笔数 |
| `deal_amount` | 成交金额 |
| `total_deal_amount_all` | 全部成交金额（总计） |
| `signal_deal_buy_amount` | 信号成交买入金额 |
| `signal_deal_sell_amount` | 信号成交卖出金额 |

## oss_* 订单规模分布（Order Size Structure）

| 字段名 | 说明 |
|---|---|
| `oss_mega_amount_pct` / `oss_mega_count_pct` | 超大单成交金额/笔数占比 |
| `oss_large_amount_pct` / `oss_large_count_pct` | 大单成交金额/笔数占比 |
| `oss_medium_amount_pct` / `oss_medium_count_pct` | 中单成交金额/笔数占比 |
| `oss_small_amount_pct` / `oss_small_count_pct` | 小单成交金额/笔数占比 |
| `oss_hot_money_count_pct` | 游资成交笔数占比 |
| `oss_buy_amount_pct` / `oss_sell_amount_pct` | 主动买入/卖出金额占比（OSS 分类） |
| `oss_mega_buy_pct` | 超大单中主动买入金额占比 |

## rs_* 成交节奏（Rhythm / Split）

区分「机器节拍」(量化) 与「手动间歇」(游资) 的核心族。

| 字段名 | 说明 |
|---|---|
| `rs_interval_mean_ms` / `rs_interval_median_ms` | 订单/成交间隔均值/中位数（毫秒） |
| `rs_interval_cv` | 订单/成交间隔变异系数（量化极低） |
| `rs_burst_ratio` | 订单/成交爆发比率 |
| `rs_buy_interval_cv` / `rs_sell_interval_cv` | 买入/卖出订单间隔变异系数 |
| `rs_split_similarity` | 订单拆分相似度（冰山拆单越高） |
| `rs_split_run_ratio` | 订单拆分连续运行比率 |

## cb_* 撤单行为（Cancel Behavior）

| 字段名 | 说明 |
|---|---|
| `cb_cancel_order_count` | 撤单笔数 |
| `cb_cancel_order_ratio` | 撤单率（撤单笔数/总订单） |
| `cb_cancel_volume_ratio` | 撤单量占比（撤单股数/总申报股数） |
| `cb_cancel_amount_ratio` | 撤单金额占比 |
| `cb_fast_cancel_ratio` | 快速撤单占比（量化试盘高） |
| `cb_cancel_interval_cv` | 撤单间隔时间变异系数 |
| `cb_buy_cancel_ratio` / `cb_sell_cancel_ratio` | 买入/卖出委托撤单率 |

## ap_* 主动买卖（Active Participation）

意图方向识别（买/卖/中性）的核心族。

| 字段名 | 说明 |
|---|---|
| `ap_active_buy_pct` / `ap_active_sell_pct` | 主动买入/卖出成交额占比 |
| `ap_active_net_direction` | 主动买卖净方向 |
| `ap_unilateral_intensity` | 主动成交单边强度 |
| `ap_dominant_direction` | 主动成交主导方向（买/卖/均衡） |
| `ap_active_volume_pct` | 主动成交量占比 |
| `ap_active_buy_run_max` / `ap_active_sell_run_max` | 最大连续主动买入/卖出成交笔数 |

## obp_* 盘口挂单（Order Book Placement）

| 字段名 | 说明 |
|---|---|
| `obp_at_best_bid_ratio` / `obp_near_best_bid_ratio` | 最优/靠近最优买价挂单占比 |
| `obp_cross_spread_buy` | 穿越价差买入挂单情况（游资激进） |
| `obp_avg_bid_offset` | 平均买单挂单价格偏移 |
| `obp_at_best_ask_ratio` / `obp_near_best_ask_ratio` | 最优/靠近最优卖价挂单占比 |
| `obp_cross_spread_sell` | 穿越价差卖出挂单情况 |
| `obp_avg_ask_offset` | 平均卖单挂单价格偏移 |

## pd_* 价格发现（Price Discovery，Q1–Q5 / H1–H3 分段）

| 字段名 | 说明 |
|---|---|
| `pd_Q1_ratio` / `pd_Q1_fast_ratio` | Q1 比率 / 快速比率（订单簿不平衡） |
| `pd_Q2_ratio` / `pd_Q2_order_count` | Q2 比率 / 订单数 |
| `pd_Q3_cv` / `pd_Q3_order_count` | Q3 变异系数 / 订单数 |
| `pd_Q4_bid_ratio` / `pd_Q4_ask_ratio` | Q4 买方/卖方挂单比率 |
| `pd_Q5_deal_amount` / `pd_Q5_impact` / `pd_Q5_mean_price` | Q5 成交金额 / 冲击 / 均价 |
| `pd_Q5_effective_threshold` / `pd_Q5_large_threshold` | Q5 有效/大单阈值 |
| `pd_H1_buy_pct` / `pd_H1_sell_pct` / `pd_H1_uni` | H1 买入/卖出占比 / 单一性 |
| `pd_H1_deal_amount` / `pd_H1_mega_threshold` | H1 成交金额 / 超大单阈值 |
| `pd_H2_price_chg` | H2 价格变动 |
| `pd_H3_cross_buy` / `pd_H3_cross_sell` | H3 穿越价差买入/卖出 |
| `pd_H3_deal_amount` / `pd_H3_mega_threshold` | H3 成交金额 / 超大单阈值 |

## pi_* 价格冲击（Price Impact）

| 字段名 | 说明 |
|---|---|
| `pi_max_price_impact_pct` | 最大价格冲击百分比 |
| `pi_price_std_pct` | 价格标准差百分比（波动率） |
| `pi_vwap_deviation` | 成交均价与 VWAP 偏离度 |
| `pi_herfindahl_5min` / `pi_herfindahl_30min` | 5/30 分钟成交集中度（赫芬达尔指数） |
| `pi_peak_amount_ratio` | 成交量峰值比率 |
| `pi_open_30min_amount_pct` | 开盘 30 分钟内成交额占比 |
| `pi_close_10min_amount_pct` | 收盘前 10 分钟内成交额占比 |

## 建模视角速查

| 判别目标 | 主要看的特征族 |
|---|---|
| 量化 vs 游资 | `rs_interval_cv`(低→量化)、`cb_fast_cancel_ratio`(高→量化试盘)、`rs_split_similarity`(高→冰山)、`oss_hot_money_count_pct`(高→游资)、`pi_open_30min/close_10min`(高→游资集中)、`pi_herfindahl`(高→游资) |
| 买/卖/中性意图 | `ap_active_net_direction`、`ap_dominant_direction`、`oss_buy_amount_pct`/`oss_sell_amount_pct`、`ap_active_buy_run_max`/`sell_run_max`、`signal_deal_buy/sell_amount` |
| 吸筹 | 净买入 + 冰山拆单(`rs_split_similarity`高) + 隐蔽(低`pi_max_price_impact`) |
| 出货 | 净卖出 + 激进(`ap_active_sell_run_max`高、`obp_cross_spread_sell`) |
| 拉升 | 高`pi_max_price_impact_pct` + 穿价买入 + 集中 |
| 对倒/T0 | 买卖均衡(`ap_dominant_direction`=均衡) + 高换手 + 量化节奏 |
| 试盘 | 高`cb_fast_cancel_ratio` + 小单 |
