"""数据源抽象接口 + 内部快照 schema 定义与 10 档盘口组装工具。"""
from __future__ import annotations

import abc
import json

import pandas as pd

# 内部统一快照 schema（列名）
INTERNAL_SNAPSHOT_COLS = [
    "symbol", "date", "dt", "hh",
    "price", "volume", "amount", "transactions",
    "totalbidvolume", "totalaskvolume", "bigordervolume",
    "bids", "asks",
]


class SnapshotSource(abc.ABC):
    """快照数据源抽象基类。子类实现 load() 返回内部 schema 的原始快照 DataFrame。"""

    @abc.abstractmethod
    def load(self, symbols: list[str] | None = None,
             dates: list[str] | None = None) -> pd.DataFrame:
        """返回归一到内部 schema 的快照 DataFrame（每行一个 tick）。

        symbols/dates 为 None 时由具体源决定（如返回全部可用数据）。
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


def assemble_book_json(row: pd.Series, side: str, n_levels: int = 10,
                       px_fmt: str = "{side}px{i}", vol_fmt: str = "{side}vol{i}") -> str:
    """把分列的 N 档价量组装成内部 bids/asks JSON 字符串。

    side: 'bid' | 'ask'（或 'offer'）。列名模板可配置，默认 bidpx1..10 / bidvol1..10。
    多数 Level-2 库以分列存十档；本函数把它们拼成内部 JSON（无逐单拆分时 order 省略）。
    """
    levels = []
    for i in range(1, n_levels + 1):
        pxc = px_fmt.format(side=side, i=i)
        volc = vol_fmt.format(side=side, i=i)
        px = row.get(pxc)
        vol = row.get(volc)
        if px is None or (isinstance(px, float) and pd.isna(px)) or float(px or 0) <= 0:
            continue
        lvl = {"price": float(px), "volume": float(vol or 0)}
        levels.append(lvl)
    return json.dumps(levels, ensure_ascii=False)


def to_internal(df: pd.DataFrame, colmap: dict[str, str]) -> pd.DataFrame:
    """按 colmap(源列→内部列) 重命名，仅保留内部 schema 存在的列。"""
    present = {src: dst for src, dst in colmap.items() if src in df.columns}
    out = df.rename(columns=present)
    return out
