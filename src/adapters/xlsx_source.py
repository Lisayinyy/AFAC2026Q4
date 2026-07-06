"""官方 xlsx/csv 快照数据源（文件或目录）。"""
from __future__ import annotations

import glob
import os

import pandas as pd

from .base import SnapshotSource


class XlsxSource(SnapshotSource):
    produces_daily_features = False  # 产原始快照，需 snapshot_features 聚合

    def __init__(self, path: str):
        self.path = path

    def _files(self) -> list[str]:
        if os.path.isdir(self.path):
            out = []
            for ext in ("*.xlsx", "*.xls", "*.csv"):
                out.extend(glob.glob(os.path.join(self.path, ext)))
            return sorted(out)
        return [self.path] if os.path.isfile(self.path) else []

    def load(self, symbols=None, dates=None) -> pd.DataFrame:
        files = self._files()
        if not files:
            raise FileNotFoundError(f"未找到快照文件: {self.path}")
        frames = []
        for fp in files:
            if fp.lower().endswith((".xlsx", ".xls")):
                frames.append(pd.read_excel(fp))
            else:
                frames.append(pd.read_csv(fp))
        df = pd.concat(frames, ignore_index=True)
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str)
        if symbols:
            df = df[df["symbol"].isin([str(s) for s in symbols])]
        return df
