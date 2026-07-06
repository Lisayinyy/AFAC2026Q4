# 方案设计 · Solution Design

## 0. 设计约束回顾

- **无监督为主**：官方无 ground-truth 标签，评分靠组委会真实市场验证。Task 1 天然无监督；Task 2 无法直接有监督训练，采用「领域规则弱标签 + 无监督结构 + 可选自训练」。
- **禁硬编码**：所有判别必须由 Level-2 派生特征驱动，不得基于 `stock_code` 打标或随机填充。
- **可复现**：`main.py` 一键跑通，相对路径，依赖入 `init_env.sh`。

## 1. 数据层

输入优先级：
1. 官方参考特征集（`data/sample/*.csv`，~90 字段，见 `feature_dictionary.md`）——直接建模。
2. Level-2 原始四表（逐笔委托/成交/撤单/十档快照）——若拿到，走 `features.py` 自行构建同名/兼容特征。
3. **合成兜底**：无数据时 `data_loader.generate_synthetic()` 生成模拟游资/量化两类微观结构，保证管线可运行、可测。

窗口聚合：赛题要求「全天交易汇总」的结论。若特征集是分窗口（intraday window）粒度，先按 `(symbol, date)` 聚合到日级（均值/加权/序列保留）再判别；序列本身保留用于 DTW。

## 2. 特征工程（`features.py`）

在参考特征基础上做二次加工，构造更强的判别量：

- **节奏规整度** `regularity = 1/(1+rs_interval_cv)`：越高越像机器（量化）。
- **拆单强度** `iceberg = rs_split_similarity * rs_split_run_ratio`：冰山吸筹信号。
- **撤单博弈** `spoof = cb_fast_cancel_ratio * cb_cancel_order_ratio`：试盘/幌骗。
- **激进度** `aggression = obp_cross_spread_buy + obp_cross_spread_sell`（标准化）：主动穿价。
- **净主动方向** `net_active = ap_active_buy_pct - ap_active_sell_pct`。
- **时段集中度** `edge_concentration = pi_open_30min_amount_pct + pi_close_10min_amount_pct`：游资开收盘突袭。
- **冲击效率** `impact_per_amount = pi_max_price_impact_pct / (deal_amount+eps)`：小钱大波动→拉抬。

全部标准化（`RobustScaler`，抗高频厚尾）后进入下游。缺失特征按族均值 / 0 填充并记录缺失掩码，保证不同数据源鲁棒。

## 3. Task 1 — 交易模式识别（`pattern_clustering.py`）

**目标**：每 (stock, date) 归入一个交易模式，输出 `pattern_type` + `pattern_explanation`。

**距离**（赛题指定综合 Wasserstein + DTW，见 `distance.py`）：
- **Wasserstein**：把每个样本的一组关键分布型特征（订单规模分布 `oss_*`、主动买卖分布）视为分布，算 1-Wasserstein，刻画「资金结构」差异。
- **DTW**：把日内序列（分窗口的净主动、成交额曲线）视为时间序列，算 DTW，刻画「节奏形态」差异。
- **综合距离** `D = α·W_norm + (1-α)·DTW_norm`（α 默认 0.5，可调）。

**聚类**：基于预计算综合距离矩阵做**层次聚类**（average linkage）或谱聚类；簇数用轮廓系数 / Calinski-Harabasz 在候选范围自动选优（对齐评估：类间区分度↑、类内聚合度↑）。

**模式命名**（可解释，非硬编码——按簇质心的特征画像自动映射语义）：

| 画像特征组合 | pattern_type | pattern_explanation |
|---|---|---|
| 净买入 + 冰山拆单 + 低冲击 | 大单吸筹 | 资金大笔挂单分批买入，隐蔽建仓 |
| 买卖均衡 + 高换手 + 机器节奏 | 日内套利 | 资金在价格区间来回高抛低吸(T0) |
| 尾盘/开盘集中 + 高冲击 | 尾盘突袭 | 特定时段集中拉升/砸盘制造形态 |
| 高快速撤单 + 小单 | 盘口博弈 | 反复挂撤试探盘口深度 |
| 净卖出 + 激进穿价 | 派发出货 | 主动砸盘、连续吃对手盘 |
| 低量 + 无主导 | 缩量整理 | 参与清淡，无明显主导资金 |

命名器读簇质心 → 匹配画像规则 → 产出稳定语义标签，规则集中在 `config.py` 便于迭代。

## 4. Task 2 — 资金类型 + 意图识别（`capital_classifier.py`）

### 4.1 capital_type：散户 / 游资 / 量化（三类）

> **实测修正**：官方 A 榜提交样例中 `capital_type` 为**三类**（散户/游资/量化），非赛题目标页所述的两类。已据此实现三类判别。

为每类构造得分，取 argmax，softmax 给置信度：

```
quant_score  ∝  regularity↑ + rs_split_similarity↑ + cb_fast_cancel_ratio↑
                + t0_balance↑（买卖均衡高换手）
hot_money_score ∝ oss_hot_money_count_pct↑ + edge_concentration↑ + big_order_pct↑
                + pi_herfindahl↑ + aggression↑ + 手动间歇↑
retail_score ∝ small_order_pct↑ + (1-big_order_pct)↑ + direction_noise↑
                + (1-edge_concentration)↑   # 小单主导、方向混乱、时段分散
```

判别：`capital_type = argmax(retail, hot_money, quant)`；权重初值来自案例领域先验（`config.py`），后续可用聚类结构 + 高置信弱标签自训练微调，保持特征驱动、可复现。

### 4.2 capital_intention：买入 / 卖出 / 中性 / T0交易

规则树（阈值集中在 `config.py`）：
- `net_active > +τ` 且 单边强度高 → **买入**（吸筹/拉升类）
- `net_active < -τ` 且 单边强度高 → **卖出**（出货/派发类）
- 买卖均衡 + 量化节奏 + 高换手 → **T0交易**
- 其余 → **中性**

意图与 Task1 模式相互印证（如「大单吸筹」→买入，「派发出货」→卖出，「日内套利」→T0），做一致性校验后输出。

### 4.3 置信度

每条输出附 `type_confidence` / `intention_confidence`（供内部排序与人工复核，提交 csv 只保留官方要求的 4 字段）。

## 5. 提交（`submit.py`）

- 生成 `pattern_reco.csv`（stock_code, transaction_date, pattern_type, pattern_explanation）。
- 生成 `predict_result.csv`（stock_code, transaction_date, capital_type, capital_intention）。
- `transaction_date` 统一 `YYYYMMDD`；`stock_code` 用样例集真实代码。
- 打包 `output/submit.zip`；校验字段名/顺序/行数与股票日期覆盖完整性。

## 6. 迭代路线（“一直完善”）

- [x] v0：管线骨架 + 合成数据跑通 + 规则版 Task1/Task2。
- [x] v1：接入真实数据（官方**原始十档快照** `snapshot_features.py`）；修正 `capital_type` 为**三类**（散户/游资/量化）；`pattern_type` 词表对齐官方样例（大单吸筹/对倒拉升/压单吸货/集合竞价异动/分时脉冲/量化T0/散户博弈…）；股票代码保留 `.SH/.SZ` 后缀；真实数据端到端验证通过。
- [x] v2：Wasserstein/DTW 距离矩阵 + 层次聚类调簇数，稳健化模式命名；从快照 `order` 数组挖掘拆单/冰山更细特征。
- [x] v3：多文件批量快照加载（目录/glob）；从 L1 `order` 数组挖掘细粒度拆单/冰山（等量重复度、最大单占比、订单量熵）+ 全10档加权盘口不平衡与深度斜率；量化/游资/散户判别**弱标签自训练**（高置信规则伪标签 → logistic，样本不足自动回退规则）。合成实测自训练把类型 F1 从 0.81 提到 0.91。
- [x] v4：意图-模式一致性图谱（`intention.py`）+ 行情阶段识别（`market_phase.py`：吸筹/试盘/拉升/派发/整理）+ 离线评估代理指标（`evaluation.py`）。
- [x] v5-框架：**数据适配层** `src/adapters/`（恒生 Hundsun 三种接入 + 十档盘口 JSON 组装 + 字段映射 `config/hundsun_schema.json`）；`run_batch.py` 批量/每日提交（股票池×交易日 → 每日 submit.zip + 汇总报告，缺数据自动跳过）；`pipeline.py` 统一编排；`tests/` 冒烟测试。
- [ ] 待数据：接入恒生真实 100 只标的快照后，校准阈值、跑多样本聚类与自训练真实效果、A 榜每日移动加权监控。

### v4/v5 模块说明

- **行情阶段**（`market_phase.py`）：由净主动、冰山、冲击、时段集中、幌骗等特征判 吸筹/试盘/拉升/派发/整理，附置信度。
- **意图一致性**（`intention.py`）：模式→意图先验（如 大单吸筹→买入、涨停板打开→卖出、量化T0→T0交易）。意图与模式冲突且置信 <0.55 时按模式先验温和修正，提升自洽性（合成实测一致率 0.75→0.925）。
- **离线评估**（`evaluation.py`）：Task1 轮廓/CH；Task2 置信度分布、类型分布、意图-模式一致率、（多日）时序漂移率；有标签则加权 F1。
- **数据适配层**（`adapters/`）：`SnapshotSource` 抽象，`HundsunSource` 支持注入查询函数 / SQLAlchemy DSN / 导出目录，自动从分列十档拼 `bids/asks` JSON。见 `docs/hundsun_setup.md`。

### 自训练机制（v3）

`src/self_training.py`：跑规则判别 → 取置信度 top50% 作伪标签 → 训练 balanced logistic（特征 `MODEL_FEATURE_COLS`）→ 回预测全体；模型低置信（<0.5）处保留规则结果做保守融合。安全网：总样本 <30、伪标签 <15、任一类 <5、无 sklearn 时自动回退纯规则，保证永远可跑可复现。全程仅用行情派生特征。

## 附：真实数据结构（实测）

- **训练/样例数据**：官方赛题一训练数据 = **单只股票单日的十档盘口快照序列**（如 603997.SH @ 20260507，4937 个快照 tick）。每行含 price/volume/amount/transactions（累计）、totalbid/askvolume、bigordervolume，及 `bids`/`asks`（10 档 JSON，每档含 `order` 拆单数组与 `bigOrderPercent`）。
- **股票池**：`股票样本.xlsx` = 100 只沪市标的（`代码.SH` + 简称），为 A/B 榜测试universe。
- **提交样例**：`pattern_reco.csv` / `predict_result.csv` 四字段，代码带交易所后缀，日期 `YYYYMMDD`。
- 快照数据无逐笔撤单信息，故撤单类特征（`cb_*`）在快照路径置 0，改由盘口 `order` 拆单数组与 `bigOrderPercent` 近似大单/冰山行为。

## 7. 评估自检（离线代理指标）

无标签下用代理指标监控：
- Task1：轮廓系数、Calinski-Harabasz、类内综合距离方差（越小越聚合）、类间质心距离（越大越区分）。
- Task2：判别得分间隔分布（间隔越大越自信）、类型/意图与模式标签一致率、时序稳定性（相邻交易日同股预测漂移率）。
