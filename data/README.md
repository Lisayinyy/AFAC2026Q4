# 数据目录

## 放置方式

把官方参考特征集 CSV 放入 `data/sample/` 即可，`main.py` 会自动加载：

```
data/sample/
├── sample_feature.csv      # 官方样例集 (2026/05/07, 1 只股票)
├── test_A_*.csv            # 测试 A 集 (100 只, 06/08-07/10)
└── test_B_*.csv            # 测试 B 集 (100 只, 07/13-07/24)
```

加载器 `src/data_loader.py` 会：
1. 读取 `data/sample/*.csv` 并 concat；
2. 自动兼容列名差异（`stock_code→symbol`、`transaction_date→date`）；
3. 若目录为空，则生成**合成兜底数据**跑通管线（不用于正式提交）。

## 数据来源

- **官方参考特征集**：赛题详情页顶部「数据下载」区，注册报名后可下载（ossutil 命令或直接下载）。字段字典见 [`../docs/feature_dictionary.md`](../docs/feature_dictionary.md)。
- **Level-2 原始数据**（逐笔委托/成交/撤单/十档快照）：官方提示可通过淘宝、闲鱼、百度网盘等公开渠道自行获取。若使用原始数据，需在 `src/features.py` 扩展从四表构建特征的逻辑。

## 注意

- 本目录内容默认 `.gitignore`（数据不入库，见根目录 `.gitignore`）。
- 数据仅用于 AFAC2026 参赛，遵守赛题规则与相关法律法规。
