# 恒生 (Hundsun) 数据接入指南

框架已把数据接入抽象为 `src/adapters/`，恒生 Level-2 数据接入**三选一**，其余
（字段映射、十档盘口 JSON 组装、归一到内部 schema、特征工程、建模、提交）全自动。

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

## 三种接入方式

### 方式 1（推荐·MiniMax code / 恒生环境内最简单）：注入查询函数

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

### 方式 2：SQLAlchemy 直连

```bash
export HUNDSUN_DSN="oracle+cx_oracle://user:pwd@host:1521/svc"   # 或 mysql/mssql...
python run_batch.py --source hundsun \
  --dsn "$HUNDSUN_DSN" \
  --sql-template "SELECT * FROM L2_SNAPSHOT WHERE SecurityID IN :symbols AND TradeDate IN :dates" \
  --stocks-file data/股票样本.xlsx --dates 20260609 20260610 --train
```

### 方式 3：离线导出目录（csv/parquet，恒生原生列名）

```bash
python run_batch.py --source hundsun --export-dir data/hundsun_export \
  --stocks-file data/股票样本.xlsx --train
```

## 批量每日提交

`run_batch.py` 遍历「股票池 × 交易日」，按日产出 `output/<date>/submit.zip` 及
`output/batch_report.json`。数据缺失/未配置的交易日自动跳过并记录，不中断整体流程。

## 注意
- 恒生 L2 逐笔若含 order-level 明细，可扩展 `snapshot_features` 的 `order` 拆单解析（当前从盘口 L1 order 数组近似）。
- 按既定安全原则：不在共享/服务端硬编码个人凭证，DSN 走环境变量。
