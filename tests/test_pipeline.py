"""冒烟测试：验证核心管线、恒生适配器、提交校验、自训练可跑通。

运行: python -m pytest tests/ -q   或   python tests/test_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_loader, pipeline, submit  # noqa: E402
from src.adapters import HundsunSource, get_source  # noqa: E402


def test_synthetic_pipeline_end_to_end():
    df = data_loader.generate_synthetic(n_stocks=60)
    res = pipeline.run_once(df, use_self_training=True)
    assert len(res["pattern"]) == 60
    assert len(res["predict"]) == 60
    assert set(res["predict"]["capital_type"]).issubset({"散户", "游资", "量化"})
    assert "silhouette" in res["report"]["task1"] or res["report"]["task1"]["n_clusters"] >= 1


def test_hundsun_adapter_book_assembly():
    raw = pd.DataFrame([{
        "SecurityID": "603316.SH", "TradeDate": "2026-05-07",
        "DataTimeStamp": 1778118226000, "TradeTime": 9, "LastPx": 21.2,
        "TotalVolumeTrade": 100000, "TotalValueTrade": 2120000, "NumTrades": 50,
        "TotalBidQty": 5000, "TotalOfferQty": 8000, "BigOrderVolume": 1200,
        **{f"BidPrice{i}": round(21.2 - 0.01 * i, 2) for i in range(1, 11)},
        **{f"BidOrderQty{i}": 100 * i for i in range(1, 11)},
        **{f"OfferPrice{i}": round(21.2 + 0.01 * i, 2) for i in range(1, 11)},
        **{f"OfferOrderQty{i}": 90 * i for i in range(1, 11)},
    }])
    src = HundsunSource(fetch_fn=lambda s, d: raw)
    df = src.load(["603316.SH"], ["20260507"])
    assert df["symbol"].iloc[0] == "603316.SH"
    bids = json.loads(df["bids"].iloc[0])
    assert len(bids) == 10 and bids[0]["price"] > 0


def test_hundsun_unconfigured_raises():
    try:
        get_source("hundsun").load()
    except RuntimeError as e:
        assert "未配置" in str(e)
    else:
        raise AssertionError("未配置的 HundsunSource 应抛 RuntimeError")


def test_submit_validation_and_vocab():
    df = data_loader.generate_synthetic(n_stocks=30)
    res = pipeline.run_once(df)
    pp = submit.write_pattern(res["pattern"])
    qp = submit.write_predict(res["predict"])
    val = submit.validate(pp, qp)
    assert val["ok"], val["issues"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("all tests passed")
