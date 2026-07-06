"""恒生 (Hundsun) Level-2 数据库适配器。

设计目标：**接入口就绪**——你在 MiniMax code / 恒生环境里，只需三选一提供数据获取方式，
其余（字段映射、十档盘口组装、归一到内部 schema）本适配器自动完成。

三种接入方式（优先级从高到低）：
  1) 注入查询函数 fetch_fn(symbols, dates) -> pd.DataFrame   —— 最灵活，推荐
  2) SQLAlchemy DSN（环境变量 HUNDSUN_DSN）+ SQL 模板        —— 直连数据库
  3) 本地导出文件（csv/parquet 目录，恒生原生列名）          —— 离线兜底

字段映射见 config/hundsun_schema.json（可编辑，无需改代码）：
  - id_map:   恒生列 → 内部列 (symbol/dt/hh/price/volume/...)
  - book:     十档价量列名模板 (bidpx1.., bidvol1.., askpx1.., askvol1..) 与档数
恒生 Level-2 通常以分列存十档（不是 JSON），本适配器据模板拼成内部 bids/asks JSON。
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

from .base import SnapshotSource, assemble_book_json, to_internal

_DEFAULT_MAPPING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "hundsun_schema.json",
)


def _load_mapping(path: str | None) -> dict:
    path = path or _DEFAULT_MAPPING_PATH
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return _FALLBACK_MAPPING


# 恒生/交易所 Level-2 快照常见列名的默认映射（可被 config/hundsun_schema.json 覆盖）
_FALLBACK_MAPPING = {
    "id_map": {
        "SecurityID": "symbol", "TradeDate": "dt", "DataTimeStamp": "date",
        "TradeTime": "hh", "LastPx": "price", "TotalVolumeTrade": "volume",
        "TotalValueTrade": "amount", "NumTrades": "transactions",
        "TotalBidQty": "totalbidvolume", "TotalOfferQty": "totalaskvolume",
        "BigOrderVolume": "bigordervolume",
    },
    "book": {
        "n_levels": 10,
        "bid_px": "BidPrice{i}", "bid_vol": "BidOrderQty{i}",
        "ask_px": "OfferPrice{i}", "ask_vol": "OfferOrderQty{i}",
    },
}


class HundsunSource(SnapshotSource):
    produces_daily_features = False

    def __init__(self, fetch_fn=None, dsn: str | None = None,
                 sql_template: str | None = None, export_dir: str | None = None,
                 mapping_path: str | None = None):
        self.fetch_fn = fetch_fn
        self.dsn = dsn or os.environ.get("HUNDSUN_DSN")
        self.sql_template = sql_template
        self.export_dir = export_dir
        self.mapping = _load_mapping(mapping_path)

    # ---- 三种原始数据获取方式 ----
    def _fetch_raw(self, symbols, dates) -> pd.DataFrame:
        if self.fetch_fn is not None:
            return self.fetch_fn(symbols, dates)
        if self.export_dir:
            return self._read_exports(symbols, dates)
        if self.dsn and self.sql_template:
            return self._query_sql(symbols, dates)
        raise RuntimeError(
            "HundsunSource 未配置数据获取方式。请三选一:\n"
            "  1) HundsunSource(fetch_fn=你的查询函数)  # 在 MiniMax code/恒生环境里最简单\n"
            "  2) HundsunSource(dsn='...', sql_template='SELECT ... WHERE SecurityID IN :symbols AND TradeDate IN :dates')\n"
            "  3) HundsunSource(export_dir='data/hundsun_export')  # 恒生导出的 csv/parquet 目录\n"
        )

    def _read_exports(self, symbols, dates) -> pd.DataFrame:
        files = []
        for ext in ("*.parquet", "*.csv"):
            files.extend(glob.glob(os.path.join(self.export_dir, ext)))
        if not files:
            raise FileNotFoundError(f"恒生导出目录无文件: {self.export_dir}")
        frames = []
        for fp in sorted(files):
            frames.append(pd.read_parquet(fp) if fp.endswith(".parquet") else pd.read_csv(fp))
        return pd.concat(frames, ignore_index=True)

    def _query_sql(self, symbols, dates) -> pd.DataFrame:
        try:
            from sqlalchemy import create_engine, text
        except ImportError as e:
            raise RuntimeError("SQL 直连需要 sqlalchemy: pip install sqlalchemy") from e
        engine = create_engine(self.dsn)
        with engine.connect() as conn:
            params = {}
            if symbols:
                params["symbols"] = tuple(str(s) for s in symbols)
            if dates:
                params["dates"] = tuple(str(d) for d in dates)
            return pd.read_sql(text(self.sql_template), conn, params=params)

    # ---- 归一到内部 schema ----
    def load(self, symbols=None, dates=None) -> pd.DataFrame:
        raw = self._fetch_raw(symbols, dates)
        return self.normalize(raw)

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        """恒生原生列 → 内部 schema，并从分列十档组装 bids/asks JSON。"""
        m = self.mapping
        df = to_internal(raw, m.get("id_map", {}))

        book = m.get("book", {})
        n = int(book.get("n_levels", 10))
        bid_px, bid_vol = book.get("bid_px", "BidPrice{i}"), book.get("bid_vol", "BidOrderQty{i}")
        ask_px, ask_vol = book.get("ask_px", "OfferPrice{i}"), book.get("ask_vol", "OfferOrderQty{i}")

        # 若已是 JSON 的 bids/asks 列则直接用；否则从分列组装
        if "bids" not in df.columns:
            df["bids"] = raw.apply(
                lambda r: _assemble(r, bid_px, bid_vol, n), axis=1)
        if "asks" not in df.columns:
            df["asks"] = raw.apply(
                lambda r: _assemble(r, ask_px, ask_vol, n), axis=1)

        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str)
        if "dt" in df.columns:
            df["dt"] = df["dt"].astype(str).str.replace("-", "", regex=False)
        return df


def _assemble(row, px_tmpl: str, vol_tmpl: str, n: int) -> str:
    import json as _json
    levels = []
    for i in range(1, n + 1):
        px = row.get(px_tmpl.format(i=i))
        vol = row.get(vol_tmpl.format(i=i))
        if px is None or (isinstance(px, float) and pd.isna(px)) or float(px or 0) <= 0:
            continue
        levels.append({"price": float(px), "volume": float(vol or 0)})
    return _json.dumps(levels, ensure_ascii=False)
