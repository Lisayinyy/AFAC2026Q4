"""合成数据源：包装 data_loader.generate_synthetic 为 SnapshotSource。

注意：合成源直接产出**日级特征**（非原始快照），用于验证下游管线。
main/批量流程对合成源特殊处理（跳过 snapshot_features）。
"""
from __future__ import annotations

import pandas as pd

from .base import SnapshotSource


class SyntheticSource(SnapshotSource):
    produces_daily_features = True  # 直接产日级特征

    def __init__(self, n_stocks: int = 40, seed: int = 42):
        self.n_stocks = n_stocks
        self.seed = seed

    def load(self, symbols=None, dates=None) -> pd.DataFrame:
        from ..data_loader import generate_synthetic
        return generate_synthetic(n_stocks=self.n_stocks, seed=self.seed)
