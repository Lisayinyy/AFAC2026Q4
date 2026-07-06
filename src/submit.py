"""提交文件生成：pattern_reco.csv + predict_result.csv → submit.zip。

严格按官方字段名与顺序输出，并做完整性校验。
"""
from __future__ import annotations

import os
import zipfile

import pandas as pd

from . import config

PATTERN_COLS = ["stock_code", "transaction_date", "pattern_type", "pattern_explanation"]
PREDICT_COLS = ["stock_code", "transaction_date", "capital_type", "capital_intention"]


def _fmt_date(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)


def write_pattern(pattern_df: pd.DataFrame, path: str | None = None) -> str:
    path = path or config.PATTERN_FILE
    out = pd.DataFrame({
        "stock_code": pattern_df["symbol"].astype(str),
        "transaction_date": _fmt_date(pattern_df["date"]),
        "pattern_type": pattern_df["pattern_type"],
        "pattern_explanation": pattern_df["pattern_explanation"],
    })[PATTERN_COLS]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_predict(predict_df: pd.DataFrame, path: str | None = None) -> str:
    path = path or config.PREDICT_FILE
    out = pd.DataFrame({
        "stock_code": predict_df["symbol"].astype(str),
        "transaction_date": _fmt_date(predict_df["date"]),
        "capital_type": predict_df["capital_type"],
        "capital_intention": predict_df["capital_intention"],
    })[PREDICT_COLS]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def validate(pattern_path: str, predict_path: str) -> dict:
    """校验字段名/顺序、行数与股票日期覆盖一致性。"""
    p = pd.read_csv(pattern_path, dtype=str)
    q = pd.read_csv(predict_path, dtype=str)
    issues = []
    if list(p.columns) != PATTERN_COLS:
        issues.append(f"pattern 列不符: {list(p.columns)}")
    if list(q.columns) != PREDICT_COLS:
        issues.append(f"predict 列不符: {list(q.columns)}")
    if len(p) != len(q):
        issues.append(f"行数不一致: pattern={len(p)} predict={len(q)}")
    pk_p = set(zip(p["stock_code"], p["transaction_date"]))
    pk_q = set(zip(q["stock_code"], q["transaction_date"]))
    if pk_p != pk_q:
        issues.append("两文件的 (stock_code,date) 键不一致")
    valid_types = set(config.CAPITAL_TYPES)
    bad_type = set(q["capital_type"]) - valid_types
    if bad_type:
        issues.append(f"非法 capital_type: {bad_type}")
    valid_int = set(config.INTENTIONS)
    bad_int = set(q["capital_intention"]) - valid_int
    if bad_int:
        issues.append(f"非法 capital_intention: {bad_int}")
    valid_patterns = {name for name, _ in config.PATTERN_RULES}
    bad_pat = set(p["pattern_type"]) - valid_patterns
    if bad_pat:
        issues.append(f"非官方 pattern_type: {bad_pat}")
    return {"ok": not issues, "issues": issues, "rows": len(p)}


def pack(pattern_path: str, predict_path: str, zip_path: str | None = None) -> str:
    zip_path = zip_path or config.SUBMIT_ZIP
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pattern_path, arcname="pattern_reco.csv")
        zf.write(predict_path, arcname="predict_result.csv")
    return zip_path
