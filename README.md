# AFAC2026 — 市场参与者交易行为识别与资金流向分析

> 天池 AFAC2026 大赛 · 赛题 [532489](https://tianchi.aliyun.com/competition/entrance/532489/information) · 奖金 ¥1,000,000

本仓库是我们的参赛工程：基于 A 股个股 **Level-2 高频数据**（逐笔委托 / 逐笔成交 / 逐笔撤单 / 十档盘口快照）及官方参考特征集，构建一套“市场参与者交易行为识别与资金流向分析”体系化解决方案。

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
python main.py --source snapshot --snapshot-path data/xxx.xlsx   # 读原始十档快照
python main.py --source synthetic                                # 合成兜底(验证管线)
```

无真实数据时，`main.py` 会用内置的**合成数据**（模拟游资/量化/散户微观结构）跑通全流程。放入官方数据后即可产出正式提交文件。已实测支持官方**原始十档快照**格式（`bids/asks` 10 档 JSON + `order` 拆单 + `bigOrderPercent`）。

## 目录结构

```
AFAC2026/
├── README.md
├── init_env.sh              # 环境依赖安装
├── requirements.txt
├── main.py                  # 统一入口
├── data/
│   ├── README.md            # 数据获取与放置说明
│   └── sample/              # 样例特征集(需自行下载放入)
├── src/
│   ├── config.py            # 路径与超参
│   ├── data_loader.py       # 数据加载 + 合成兜底 + 快照加载
│   ├── features.py          # 特征工程(参考特征集路径)
│   ├── snapshot_features.py # 十档快照 → 日级特征
│   ├── distance.py          # Wasserstein + DTW 距离
│   ├── pattern_clustering.py# Task1
│   ├── capital_classifier.py# Task2 (散户/游资/量化)
│   └── submit.py            # 生成 submit.zip
├── docs/
│   ├── competition_rules.md # 完整赛题规则(存档)
│   ├── feature_dictionary.md# 参考特征集字段字典
│   └── solution_design.md   # 方案设计
└── output/                  # 运行产物(gitignore)
```

## 赛程

| 阶段 | 时间 | 要点 |
|------|------|------|
| A 榜 | 2026/06/09 – 07/10 | 每日最多 3 次提交，23:59 截止，取当日最后一次 |
| B 榜 | 2026/07/13 – 07/24 | 每日必交，需满 8 个交易日 |
| 报告提交 | 2026/07/28 – 08/05 | B 榜 TOP15 提交 report + 代码复现 |

## 许可

参赛工程，仅用于 AFAC2026 大赛。
