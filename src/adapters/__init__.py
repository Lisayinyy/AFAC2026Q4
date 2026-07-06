"""数据适配层：统一不同数据源到内部快照 schema。

内部快照 schema（每行一个快照 tick，snapshot_features 消费）：
  symbol, date(epoch ms 或可比时间戳), dt(交易日 YYYYMMDD), hh,
  price, volume(累计), amount(累计), transactions(累计),
  totalbidvolume, totalaskvolume, bigordervolume,
  bids/asks (10 档 JSON: [{"price","volume","order":[{"volume"}...],"bigOrderPercent"}])

各数据源实现 SnapshotSource.load(...) 返回**已归一到内部 schema** 的原始快照 DataFrame，
再由 snapshot_features.build_from_snapshot 聚合成日级特征。
"""
from __future__ import annotations

from .base import SnapshotSource, INTERNAL_SNAPSHOT_COLS
from .synthetic_source import SyntheticSource
from .xlsx_source import XlsxSource
from .hundsun import HundsunSource

__all__ = [
    "SnapshotSource", "INTERNAL_SNAPSHOT_COLS",
    "SyntheticSource", "XlsxSource", "HundsunSource",
    "get_source",
]


def get_source(kind: str, **kwargs) -> SnapshotSource:
    """工厂：按名称返回数据源实例。

    kind: 'hundsun' | 'xlsx' | 'synthetic'
    """
    kind = (kind or "").lower()
    if kind == "hundsun":
        return HundsunSource(**kwargs)
    if kind in ("xlsx", "snapshot", "file"):
        return XlsxSource(**kwargs)
    if kind == "synthetic":
        return SyntheticSource(**kwargs)
    raise ValueError(f"未知数据源: {kind}")
