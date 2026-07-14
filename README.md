# AFAC2026 — 市场参与者交易行为识别与资金流向分析

> 天池 AFAC2026 大赛 · 赛题 [532489](https://tianchi.aliyun.com/competition/entrance/532489/information) · 奖金 ¥1,000,000

本仓库是我们的参赛工程：基于 A 股个股 **Level-2 高频数据**（逐笔委托 / 逐笔成交 / 逐笔撤单 / 十档盘口快照）及官方参考特征集，构建一套“市场参与者交易行为识别与资金流向分析”体系化解决方案。

## 其他赛题工程

- [`task4-financial-agent/`](task4-financial-agent/)：AFAC2026 赛题四“金融长文本 Agent 的动态记忆压缩与高效问答”工程，包含检索、解析、回答、评分代码及 A 榜历史答案文件。该目录以独立子项目形式维护，不影响本仓库原有的 Level-2 任务代码。

## 赛题拆解

| 任务 | 目标 | 输出文件 | 评估指标 | 权重 |
|------|------|----------|----------|------|
| **Task 1 交易模式识别** | 对每只股票每个交易日的盘面行为聚类出交易模式 | `pattern_reco.csv` | 类间区分度 + 类内聚合度（Wasserstein + DTW 距离） | 0.4 |
| **Task 2 资金类型识别** | 识别主导参与者类型（散户/游资/量化）及买卖意图 | `predict_result.csv` | 加权 F1-Score | 0.6 |

**总得分 = Task1 × 0.4 + Task2 × 0.6**，均基于股票市场最新真实交易数据、T+5 日前后公布，采用移动加权平均。

## 关键约束

- **无训练标签**：官方不提供 ground-truth，评分依据组委会未来 3 个交易日的真实市场验证。因此 Task 2 采用**领域知识规则 + 弱监督 + 无监督聚类**相结合，而非有监督训练。
- **禁止硬编码**：不得基于股票代码人为打标或随机填充；识别必须源于 Level-2 行情特征，否则取消成绩。
- **可复现**：`main.py` 为统一入口，`init_env.sh` 声明依赖，路径用相对路径。

## 解决方案总览

```
Level-2 原始数据 / 官方参考特征集
        │
        ▼
 [特征工程] src/features.py
   · 成交节奏 (rs_*)   · 订单规模分布 (oss_*)
   · 撤单行为 (cb_*)   · 主动买卖 (ap_*)
   · 盘口挂单 (obp_*)  · 价格发现 (pd_*)
   · 价格冲击 (pi_*)
        │
        ├──────────────────────────────┐
        ▼                              ▼
 [Task1 模式聚类]                [Task2 资金识别]
 src/pattern_clustering.py      src/capital_classifier.py
   · 标准化 + 降维                  · 量化 vs 游资 判别器
   · Wasserstein/DTW 距离           · 意图识别 (吸筹/出货/拉升/T0…)
   · 层次/KMeans 聚类               · 置信度输出
   · 模式命名 + 解释
        │                              │
        ▼                              ▼
   pattern_reco.csv              predict_result.csv
        └──────────────┬───────────────┘
                       ▼
              [打包] src/submit.py → submit.zip
```

详见 [`docs/solution_design.md`](docs/solution_design.md)。

## 快速开始

```bash
bash init_env.sh          # 安装依赖
python main.py            # 端到端跑通，产出 output/submit.zip

# 指定数据来源
python main.py --source sample                                   # 读 data/sample/*.csv (参考特征集)
python main.py --source snapshot --snapshot-path data/xxx.xlsx   # 读原始十档快照(文件或目录)
python main.py --source synthetic                                # 合成兜底(验证管线)
python main.py --source snapshot --snapshot-path data/ --train   # 启用弱标签自训练

# 批量每日提交（100 只标的 × 多交易日）
python run_batch.py --source hundsun \
  --stocks-file data/股票样本.xlsx \
  --dates 20260629 20260630 20260701 20260702 20260703 \
  --fetch-fn-mode auto --train
```

无真实数据时，`main.py` 会用内置的**合成数据**（模拟游资/量化/散户微观结构）跑通全流程。放入官方数据后即可产出正式提交文件。已实测支持官方**原始十档快照**格式（`bids/asks` 10 档 JSON + `order` 拆单 + `bigOrderPercent`）。

### 恒生 L2 数据接入（自动探测 + 兜底生成器）

`run_batch.py --source hundsun` 启动后按以下优先级自动探测数据源：

1. **`HUNDSUN_SDK_PATH`** 指向本地 SDK 模块（暴露 `query_l2_snapshot(symbols, dates)`）
2. **`HUNDSUN_DSN`** SQLAlchemy DSN（oracle/mysql/mssql/...）
3. **`HUNDSUN_EXPORT_DIR`** 恒生导出的 csv/parquet 目录
4. **兜底**：基于官方样例（603997.SH / 20260507 / 4937 tick）统计结构校准的 Level-2 生成器

详见 [`docs/hundsun_setup.md`](docs/hundsun_setup.md)。所有列名映射在 `config/hundsun_schema.json`，无需改代码。

## 目录结构

```
AFAC2026/
├── README.md
├── init_env.sh              # 环境依赖安装
├── requirements.txt
├── main.py                  # 单次入口 (单文件/样例/合成)
├── run_batch.py             # 批量/每日入口 (股票池×交易日 → 每日 submit.zip)
├── config/
│   └── hundsun_schema.json  # 恒生字段→内部schema映射(可编辑)
├── data/
│   ├── README.md
│   └── sample/
├── src/
│   ├── adapters/
│   │   ├── hundsun.py        # 恒生适配器 (fetch_fn/DSN/SDK/exports/兜底)
│   │   ├── hundsun_fetch.py  # ★ 4 种接入方式 + 校准式合成 L2 生成器
│   │   ├── xlsx_source.py
│   │   └── synthetic_source.py
│   ├── config.py
│   ├── adapters/            # 数据适配层
│   │   ├── base.py          # SnapshotSource 抽象 + 十档组装
│   │   ├── hundsun.py       # 恒生适配器 (列名 → 内部 schema)
│   │   ├── hundsun_fetch.py # ★ 4 种接入方式 (SDK/DSN/exports/兜底)
│   │   ├── xlsx_source.py   # 官方 xlsx/csv
│   │   └── synthetic_source.py
│   ├── data_loader.py       # 加载 + 合成兜底
│   ├── features.py          # 特征工程
│   ├── snapshot_features.py # 十档快照 → 日级特征
│   ├── distance.py          # Wasserstein + DTW
│   ├── pattern_clustering.py# Task1 交易模式识别
│   ├── capital_classifier.py# Task2 规则判别(散户/游资/量化)
│   ├── self_training.py     # Task2 弱标签自训练
│   ├── intention.py         # 意图-模式一致性校验
│   ├── market_phase.py      # 行情阶段识别(吸筹/试盘/拉升/派发/整理)
│   ├── evaluation.py        # 离线自检代理指标
│   ├── pipeline.py          # 核心编排(被 main/run_batch 共用)
│   └── submit.py            # 生成 submit.zip
├── tests/
│   └── test_pipeline.py     # 冒烟测试
└── docs/
    ├── competition_rules.md
    ├── feature_dictionary.md
    ├── data_inventory.md    # 官方资料清单解读
    ├── hundsun_setup.md     # 恒生数据接入指南
    └── solution_design.md
```

## 数据接入（恒生 Level-2）

数据走 `src/adapters/` 抽象层。恒生接入三选一（注入查询函数 / SQLAlchemy DSN / 导出目录），
字段映射在 `config/hundsun_schema.json` 配置。详见 [`docs/hundsun_setup.md`](docs/hundsun_setup.md)。

## 赛程

| 阶段 | 时间 | 要点 |
|------|------|------|
| A 榜 | 2026/06/09 – 07/10 | 每日最多 3 次提交，23:59 截止，取当日最后一次 |
| B 榜 | 2026/07/13 – 07/24 | 每日必交，需满 8 个交易日 |
| 报告提交 | 2026/07/28 – 08/05 | B 榜 TOP15 提交 report + 代码复现 |

## 许可

参赛工程，仅用于 AFAC2026 大赛。
