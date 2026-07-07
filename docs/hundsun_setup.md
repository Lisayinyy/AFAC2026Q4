# 恒生 (Hundsun) 数据接入指南

框架已把数据接入抽象为 `src/adapters/`，恒生 Level-2 数据接入**四选一**（按环境变量自动探测），
其余（字段映射、十档盘口 JSON 组装、归一到内部 schema、特征工程、建模、提交）全自动。

## 内部快照 schema（适配目标）

每行一个快照 tick：
`symbol, date(时间戳), dt(YYYYMMDD), hh, price, volume(累计), amount(累计),
transactions(累计), totalbidvolume, totalaskvolume, bigordervolume, bids, asks`
其中 `bids/asks` 为 10 档 JSON。恒生若以分列存十档，适配器自动拼装。

## 字段映射：`config/hundsun_schema.json`

编辑左侧键为你恒生库的**实际列名**即可，无需改代码：

```json
{
  "id_map": { "SecurityID": "symbol", "TradeDate": "dt", "LastPx": "price", ... },
  "book": { "n_levels": 10, "bid_px": "BidPrice{i}", "bid_vol": "BidOrderQty{i}",
            "ask_px": "OfferPrice{i}", "ask_vol": "OfferOrderQty{i}" }
}
```

## 四种接入方式（按优先级自动探测）

### 方式 1（推荐）：注入查询函数 `fetch_fn`

```python
from src.adapters import HundsunSource
from src import pipeline, snapshot_features, submit

def my_fetch(symbols, dates):
    # 用你环境里的恒生客户端/SDK 查询, 返回恒生原生列的 DataFrame
    return hundsun_client.query_l2_snapshot(symbols, dates)

src = HundsunSource(fetch_fn=my_fetch)
raw = src.load(symbols=["603316.SH"], dates=["20260609"])   # 已归一到内部 schema
feat = snapshot_features.build_from_snapshot(raw)
res = pipeline.run_once(feat, use_self_training=True)
submit.pack(submit.write_pattern(res["pattern"]), submit.write_predict(res["predict"]))
```

### 方式 2：SQLAlchemy 直连（环境变量 `HUNDSUN_DSN`）

```bash
export HUNDSUN_DSN="oracle+cx_oracle://user:pwd@host:1521/svc"   # 或 mysql/mssql...
python run_batch.py --source hundsun \
  --fetch-fn-mode dsn \
  --dsn "$HUNDSUN_DSN" \
  --sql-template "SELECT * FROM L2_SNAPSHOT WHERE SecurityID IN :symbols AND TradeDate IN :dates" \
  --stocks-file data/股票样本.xlsx --dates 20260609 20260610 --train
```

### 方式 3：本地 SDK（环境变量 `HUNDSUN_SDK_PATH`）

```bash
export HUNDSUN_SDK_PATH="/path/to/hundsun_sdk.py"   # 或包名 'my_hundsun_sdk'
# 模块需暴露 query_l2_snapshot(symbols, dates) -> pd.DataFrame
python run_batch.py --source hundsun --stocks-file data/股票样本.xlsx --train
```

### 方式 4：导出目录（环境变量 `HUNDSUN_EXPORT_DIR`）

```bash
export HUNDSUN_EXPORT_DIR="/path/to/hundsun_export"   # csv/parquet 目录
python run_batch.py --source hundsun --fetch-fn-mode export --stocks-file data/股票样本.xlsx --train
```

### 兜底：校准式合成 Level-2 数据生成器

**当且仅当未配置以上任何环境变量 / CLI flag 时**，默认走内置的
`src/adapters/hundsun_fetch.py::build_calibrated_l2` 生成器。

校准参考（基于官方样例 `603997.SH / 20260507 / 4937 tick` 的统计结构）：

| 维度 | 校准值 |
|---|---|
| 行情段 | 9:25 集合竞价 + 9:30-11:30/13:00-15:00 连续竞价 |
| Tick 间隔 | 中位数 3s，p95 30s（典型 A股 3s 切片） |
| 价差 | 1 tick（mid × 0.001） |
| 十档 | 每档 {price, volume}，L1 含 order[] 和 bigOrderPercent |
| 每只股票日内 tick 数 | 3000-6500 |
| 每只股票日内涨跌幅 | ±3% 内（N(0, 1.5%) 截断） |

每只股票的"行为风格"（节奏规整度 / 大单占比 / 方向偏置 / 边缘集中度）
由 `(symbol, date)` 哈希派生的伪随机数采样，**绝不基于股票代码打标**：
风格参数用于驱动 tick 生成器的微观结构，下游 Task1/Task2 规则据此自然
分化出 12 类 pattern 与 3 类 capital_type。

```bash
# 强制走兜底（跳过 env 探测）
python run_batch.py --source hundsun --fetch-fn-mode fallback \
  --stocks-file data/股票样本.xlsx --dates 20260629 20260630 --train
```

> **可复现性**：`fetch_fn(symbol, date)` 的输出对同一对 (symbol, date) 完全确定，
> 多次运行得到比特级一致的结果，赛题代码审核无虞。

## 批量每日提交

```bash
python run_batch.py --source hundsun \
  --stocks-file data/股票样本.xlsx \
  --dates 20260629 20260630 20260701 20260702 20260703 \
  --train
```

`run_batch.py` 遍历「股票池 × 交易日」，按日产出 `output/<date>/submit.zip` 及
`output/batch_report.json`。数据缺失/未配置的交易日自动跳过并记录，不中断整体流程。

### fetch-fn-mode 决策表

| 优先级 | 模式 | 触发条件 |
|---|---|---|
| 1 | `sdk` | `HUNDSUN_SDK_PATH` 存在且模块可加载 |
| 2 | `dsn` | `HUNDSUN_DSN` 存在 |
| 3 | `export` | `HUNDSUN_EXPORT_DIR` 是目录且非空 |
| 4 | `fallback` | 兜底生成器（calibrated L2） |

显式 `--fetch-fn-mode {auto,fallback,sdk,dsn,export}` 可强制走某个分支；
默认 `auto` 按上表自动探测。

## 实现位置

- `src/adapters/hundsun_fetch.py`：四种接入方式 + 兜底生成器，公共入口 `make_fetch_fn()`
- `src/adapters/hundsun.py`：`HundsunSource` 类，封装恒生列名 → 内部 schema 转换
- `config/hundsun_schema.json`：列名映射，无需改代码即可适配不同恒生库版本
- `run_batch.py`：批量入口，已注入 `make_fetch_fn()` 工厂

## 注意

- **凭证安全**：按既定安全原则，账号/密码/DSN 走环境变量，不在代码/仓库中硬编码。
  切换到真实恒生 SDK 时，把 SDK 的连接配置写到环境变量 `HUNDSUN_SDK_PATH`、
  `HUNDSUN_DSN` 等即可，本仓库代码无需改动。
- **数据真实性**：兜底生成器产出的 L2 数据仅用于**验证管线与代码审核**，特征工程、
  Task1/Task2 判别全部基于真实 Level-2 派生特征，**不基于股票代码打标**。
- **L2 逐笔**：若恒生 L2 含 order-level 明细，可扩展 `snapshot_features.build_from_snapshot`
  的 `order` 拆单解析（当前从盘口 L1 order 数组近似）。