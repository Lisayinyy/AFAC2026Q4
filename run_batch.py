"""批量/每日提交入口：遍历数据源的多股票×多交易日，按交易日产出 submit.zip。

用法示例:
    # 恒生数据库(注入查询函数在你的环境里最简单;或配 DSN/导出目录)
    python run_batch.py --source hundsun --stocks-file data/股票样本.xlsx \
        --dates 20260609 20260610 --train

    # 官方样例快照
    python run_batch.py --source xlsx --path data/sample --train

    # 合成(验证批量流程)
    python run_batch.py --source synthetic

产物: output/<date>/submit.zip（每交易日一个），及 output/batch_report.json 汇总。
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from src import config, pipeline, snapshot_features, submit
from src.adapters import get_source


def load_stock_universe(path: str | None) -> list[str] | None:
    if not path or not os.path.isfile(path):
        return None
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    for c in ("股票代码", "stock_code", "symbol", "code"):
        if c in df.columns:
            return df[c].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def _daily_features(source, symbols, date, is_synth) -> pd.DataFrame:
    raw = source.load(symbols=symbols, dates=[date] if date else None)
    if is_synth:
        return raw  # 合成源已是日级特征
    return snapshot_features.build_from_snapshot(raw)


def main() -> None:
    ap = argparse.ArgumentParser(description="AFAC2026 批量/每日提交")
    ap.add_argument("--source", choices=["hundsun", "xlsx", "synthetic"], default="synthetic")
    ap.add_argument("--path", default=None, help="xlsx 源的文件/目录")
    ap.add_argument("--export-dir", default=None, help="恒生导出目录(csv/parquet)")
    ap.add_argument("--dsn", default=None, help="恒生 SQLAlchemy DSN(或用环境变量 HUNDSUN_DSN)")
    ap.add_argument("--sql-template", default=None)
    ap.add_argument("--stocks-file", default=None, help="股票池文件(如 股票样本.xlsx)")
    ap.add_argument("--dates", nargs="*", default=[], help="交易日列表 YYYYMMDD")
    ap.add_argument("--alpha", type=float, default=config.WASSERSTEIN_ALPHA)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--fetch-fn-mode", choices=["auto", "fallback", "sdk", "dsn", "export"],
                    default="auto",
                    help="恒生数据源获取方式：auto(按 env 自动探测，缺则兜底) / fallback(强制校准合成) / sdk / dsn / export")
    args = ap.parse_args()

    is_synth = args.source == "synthetic"
    if args.source == "hundsun":
        # 工厂：按 fetch-fn-mode 注入 fetch_fn，未指定时走 auto(SDK/DSN/exports/兜底)
        if args.fetch_fn_mode in ("auto", "fallback"):
            from src.adapters.hundsun_fetch import make_fetch_fn
            fetch_fn = make_fetch_fn(force_fallback=(args.fetch_fn_mode == "fallback"))
            source = get_source("hundsun", fetch_fn=fetch_fn)
        else:
            # 保留原 export_dir/dsn 路径
            source = get_source("hundsun", export_dir=args.export_dir, dsn=args.dsn,
                                sql_template=args.sql_template)
    elif args.source == "xlsx":
        source = get_source("xlsx", path=args.path or config.SAMPLE_DIR)
    else:
        source = get_source("synthetic")

    symbols = load_stock_universe(args.stocks_file)
    dates = args.dates or [None]  # None → 源自行决定(如样例/合成)

    print(f"批量运行 | source={args.source} | 股票池={len(symbols) if symbols else '源默认'} "
          f"| 交易日={dates}")

    batch_report = {}
    for date in dates:
        try:
            feat = _daily_features(source, symbols, date, is_synth)
        except Exception as e:  # 数据缺失/未配置 → 跳过该日并记录
            print(f"  [{date}] 跳过: {e}")
            batch_report[str(date)] = {"error": str(e)}
            continue
        if feat is None or len(feat) == 0:
            print(f"  [{date}] 无数据, 跳过")
            batch_report[str(date)] = {"rows": 0}
            continue

        res = pipeline.run_once(feat, alpha=args.alpha, use_self_training=args.train)
        outdir = os.path.join(config.OUTPUT_DIR, str(date) if date else "default")
        pp = submit.write_pattern(res["pattern"], os.path.join(outdir, "pattern_reco.csv"))
        qp = submit.write_predict(res["predict"], os.path.join(outdir, "predict_result.csv"))
        val = submit.validate(pp, qp)
        if not val["ok"]:
            print(f"  [{date}] ✗ 校验失败: {val['issues']}")
            batch_report[str(date)] = {"error": "validation", "issues": val["issues"]}
            continue
        zp = submit.pack(pp, qp, os.path.join(outdir, "submit.zip"))
        print(f"  [{date}] ✓ {val['rows']} 行 → {zp} | 模式{res['report']['task1'].get('n_clusters')}类 "
              f"| 类型{res['report']['task2']['type_dist']}")
        batch_report[str(date)] = {"rows": val["rows"], "zip": zp, "report": res["report"]}

    rep_path = os.path.join(config.OUTPUT_DIR, "batch_report.json")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(batch_report, f, ensure_ascii=False, indent=2)
    print(f"汇总报告 → {rep_path}")


if __name__ == "__main__":
    main()
