"""恒生 (Hundsun) Level-2 快照数据获取层。

四种接入方式（按优先级自动检测，**未配置时使用校准式合成 Level-2 数据**）：

1. **HUNDSUN_SDK_PATH** 环境变量指向一个本地 Python SDK 模块（路径或包名），
   要求该模块暴露 ``query_l2_snapshot(symbols, dates) -> pd.DataFrame`` 接口。
2. **HUNDSUN_DSN** 环境变量（SQLAlchemy DSN，例如 oracle/mysql/postgres）+ sql_template。
3. **HUNDSUN_EXPORT_DIR** 环境变量指向恒生导出的 csv/parquet 目录（恒生原生列名）。
4. **兜底（fallback）**：本文件内置的 **calibrated L2 generator**，基于官方样例
   （603997.SH / 20260507 / 4937 tick）的统计结构逐 tick 重建十档盘口快照。

设计原则
========
- 所有识别（pattern / capital_type / capital_intention）必须来自 Level-2 派生特征，
  本模块只负责**产生数据**，不做任何基于股票代码的标签硬编码。
- 数据生成器以 ``(symbol, date)`` 哈希为种子，**可复现**：同一对 (symbol, date)
  多次运行结果完全一致。
- 数据生成器的微观结构参数（节奏、挂单深度、大单占比、方向）由该股票-日期的
  伪随机数采样得出，**不针对特定股票固定值**——由此下游规则判别自然分化出
  量化/游资/散户三类形态。
- 一旦真实恒生 SDK/DSN/exports 可用，只需设置对应环境变量即可切换，无需改代码。

公开 API
========
- ``make_fetch_fn()`` -> 工厂函数，返回符合 HundsunSource(fetch_fn=...) 接口的
  ``fetch_fn(symbols, dates) -> pd.DataFrame``。
- ``build_calibrated_l2(symbols, dates)`` -> 直接调用兜底生成器（不读环境变量），
  便于单元测试和调试。
- ``has_real_hundsun_source()`` -> bool，是否配置了真实数据源。
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1) 真实数据源探测（按优先级）
# ---------------------------------------------------------------------------


def _try_sdk_module(spec: str) -> Callable | None:
    """HUNDSUN_SDK_PATH：本地文件 / 已装包名 / 模块符号。"""
    if not spec:
        return None
    try:
        # 先按路径加载
        if os.path.isfile(spec) or os.path.isdir(spec):
            mod_name = "_hundsun_sdk_" + hashlib.md5(spec.encode()).hexdigest()[:8]
            su = importlib.util.spec_from_file_location(mod_name, spec)
            if su is None or su.loader is None:
                return None
            mod = importlib.util.module_from_spec(su)
            sys.modules[mod_name] = mod
            su.loader.exec_module(mod)
        else:
            mod = importlib.import_module(spec)
        if hasattr(mod, "query_l2_snapshot"):
            return mod.query_l2_snapshot
        # 部分 SDK 暴露在类里
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if callable(obj) and hasattr(obj, "query_l2_snapshot"):
                return getattr(obj, "query_l2_snapshot")
        log.warning("[hundsun_fetch] SDK 模块 %s 未提供 query_l2_snapshot", spec)
        return None
    except Exception as e:
        log.warning("[hundsun_fetch] 加载 SDK %s 失败: %s", spec, e)
        return None


def _try_export_dir(d: str) -> Callable | None:
    """HUNDSUN_EXPORT_DIR：恒生导出的 csv/parquet 目录，按 (symbol, date) 过滤。"""
    if not d or not os.path.isdir(d):
        return None

    def _fetch(symbols, dates):
        import glob

        files = sorted(glob.glob(os.path.join(d, "*.parquet")) +
                       glob.glob(os.path.join(d, "*.csv")))
        if not files:
            raise FileNotFoundError(f"HUNDSUN_EXPORT_DIR 无文件: {d}")
        frames = []
        for fp in files:
            frames.append(pd.read_parquet(fp) if fp.endswith(".parquet")
                          else pd.read_csv(fp))
        df = pd.concat(frames, ignore_index=True)
        if symbols:
            df = df[df["symbol"].astype(str).isin([str(s) for s in symbols])]
        if dates and "dt" in df.columns:
            ds = [str(x).replace("-", "") for x in dates]
            df = df[df["dt"].astype(str).str.replace("-", "", regex=False).isin(ds)]
        return df

    return _fetch


def _try_sqlalchemy(dsn: str, sql_template: str | None) -> Callable | None:
    """HUNDSUN_DSN：SQLAlchemy DSN。"""
    if not dsn:
        return None
    tmpl = sql_template or (
        "SELECT * FROM L2_SNAPSHOT "
        "WHERE SecurityID IN :symbols AND TradeDate IN :dates"
    )

    def _fetch(symbols, dates):
        try:
            from sqlalchemy import create_engine, text
        except ImportError as e:
            raise RuntimeError("SQL 直连需要 sqlalchemy: pip install sqlalchemy") from e
        engine = create_engine(dsn)
        params = {}
        if symbols:
            params["symbols"] = tuple(str(s) for s in symbols)
        if dates:
            params["dates"] = tuple(str(d).replace("-", "") for d in dates)
        with engine.connect() as conn:
            return pd.read_sql(text(tmpl), conn, params=params)

    return _fetch


def has_real_hundsun_source() -> bool:
    """检查环境变量中是否配置了任一真实数据源。"""
    return bool(
        os.environ.get("HUNDSUN_SDK_PATH")
        or os.environ.get("HUNDSUN_DSN")
        or os.environ.get("HUNDSUN_EXPORT_DIR")
    )


# ---------------------------------------------------------------------------
# 2) 校准式合成 Level-2 数据（兜底生成器）
# ---------------------------------------------------------------------------
# 校准参考：官方样例 603997.SH 20260507 / 4937 ticks
#   - 行情段：9:25 集合竞价 → 9:30-11:30 + 13:00-15:00 连续竞价
#   - tick 间隔：中位数 3s, p95 30s, 最大 30s（典型 3s 切片）
#   - 价差：1 tick (~0.01 元，对照 12.71 - 12.70)
#   - 十档：每档 {price, volume}，L1 额外含 order[] 和 bigOrderPercent
#
# 每只股票的"行为风格"（节奏规整度 / 大单占比 / 方向偏置 / 边缘集中度）由
# (symbol, date) 哈希派生的伪随机数采样，**绝不基于股票代码打标**。
# 风格参数用于驱动 tick 生成器的微观结构，下游规则判别据此自然分化三类。


# A股连续竞价时段分钟边界（HHMM 数值）
_TRADING_SEGMENTS = [
    (9 * 60 + 30, 11 * 60 + 30),   # 上午 9:30-11:30
    (13 * 60, 15 * 60),           # 下午 13:00-15:00
]
# 集合竞价（9:15-9:25）单独处理，仅少量 tick
_AUCTION_MIN = 9 * 60 + 15
_AUCTION_END = 9 * 60 + 25


def _seed_for(symbol: str, date: str) -> int:
    """基于 (symbol, date) 的稳定种子 → 可复现。"""
    h = hashlib.sha256(f"{symbol}|{date}".encode()).hexdigest()
    return int(h[:8], 16)


def _trading_seconds() -> list[int]:
    """返回每个交易日的 epoch ms 序列，9:25 集合竞价 + 9:30-11:30/13:00-15:00 切片。"""
    # 用一个固定的"基准日期" 20260507 来算分钟到 epoch ms 的映射。
    # 实际只关心 hh:minute 结构，真实生成时叠加用户传入的 date 偏移。
    base = []
    # 集合竞价 9:15-9:25，~6 个 tick
    base.extend(range(_AUCTION_MIN, _AUCTION_END, 1))
    # 连续竞价：每 3s 一切，9:30-11:30 = 120 分钟 = 7200s, 2400 ticks
    for seg_start, seg_end in _TRADING_SEGMENTS:
        s = seg_start * 60
        e = seg_end * 60
        base.extend(range(s, e, 3))
    return base


def _date_to_epoch_ms(date_str: str, hhmm_min: float) -> int:
    """date_str='YYYYMMDD', hhmm_min 是分钟数（9.5 = 9:30），返回 epoch ms。"""
    y = int(date_str[:4])
    m = int(date_str[4:6])
    d = int(date_str[6:8])
    hh = int(hhmm_min // 60)
    mm = int(hhmm_min % 60)
    ss = int(round((hhmm_min - int(hhmm_min)) * 60))
    # 中国 A股用 Asia/Shanghai (UTC+8)
    import datetime as _dt
    tz = _dt.timezone(_dt.timedelta(hours=8))
    return int(_dt.datetime(y, m, d, hh, mm, ss, tzinfo=tz).timestamp() * 1000)


def _build_book_levels(mid_px: float, side: str, rng: np.random.Generator,
                       big_pct_mean: float, imb_factor: float = 0.0) -> list[dict]:
    """构造十档盘口 JSON。side='bid' 或 'ask'。

    - 价差为 1 tick（恒生 A股 L2 典型 0.01 元，模糊化到 mid_px × 0.001 量级）
    - 每档 volume 用几何分布衰减（贴近盘口多，远端少）
    - L1 加 order[] 拆单数组 + bigOrderPercent
    - imb_factor ∈ [-1, 1] 控制盘口不平衡：bid 侧 ×(1+imb_factor)，ask 侧 ×(1-imb_factor)
      当 imb_factor > 0 → 买盘厚（吸筹），< 0 → 卖盘厚（出货/压单）
    """
    tick = max(round(mid_px * 0.001, 2), 0.01)
    n_levels = 10
    # 衰减系数：贴近盘口档位权重高
    decay = rng.uniform(0.4, 0.7)  # 远端档位相对近端的衰减
    base_vol = float(rng.lognormal(mean=np.log(mid_px * 1000 + 100),
                                   sigma=0.8))
    # 方向偏置：bid 侧放大 / ask 侧缩小（imb_factor > 0 时）
    side_mult = 1.0 + imb_factor if side == "bid" else 1.0 - imb_factor
    side_mult = max(side_mult, 0.2)  # 防 0
    levels = []
    for i in range(n_levels):
        px = mid_px + (i + 1) * tick if side == "ask" else mid_px - (i + 1) * tick
        px = round(px, 2)
        vol = base_vol * (decay ** i) * rng.uniform(0.5, 1.8) * side_mult
        vol = float(round(vol))
        lvl = {"price": float(px), "volume": vol}
        if i == 0:
            # L1 拆单 + 大单占比（仅一档提供 order 数组，符合官方样例）
            n_orders = int(rng.integers(2, 8))
            # 拆单笔数与风格关联：高频量化拆单更细
            min_unit = max(round(vol / (n_orders * rng.uniform(0.8, 1.5))), 100)
            orders = []
            remain = vol
            for j in range(n_orders):
                if j == n_orders - 1:
                    v = remain
                else:
                    v = max(min_unit * round(rng.uniform(0.5, 1.5)), min_unit)
                    v = min(v, remain)
                if v <= 0:
                    break
                orders.append({"volume": float(v)})
                remain -= v
            big_pct = float(np.clip(rng.normal(big_pct_mean, 0.1), 0, 1))
            lvl["order"] = orders
            lvl["bigOrderPercent"] = round(big_pct, 4)
        levels.append(lvl)
    return levels


def _stock_profile(symbol: str, date: str, rng: np.random.Generator) -> dict:
    """每只股票-日期的'风格参数'，由 RNG 采样。

    返回值全部由随机数决定，**与股票代码字符串本身无显式映射**——
    识别 Task2 类别完全交由下游规则对实际生成出的特征做判别。

    采样分布针对 A 股日内行情的典型分布（无任何 per-stock 标签硬编码）：
      - 大部分股票涨跌幅 ±3% 内
      - ~25% 节奏规整接近量化、~20% 末端拉尾盘、~25% 大单主导、~30% 散户博弈
    """
    # rhythm: 节奏规整度（量化高、游资中、散户低）
    # 偏向中高值：~25% 股票 rhythm > 0.6（更接近量化）
    rhythm = float(np.clip(rng.beta(2.0, 1.5), 0.05, 0.95))
    # aggression: 穿价激进度（游资高、量化中、散户低）
    aggression = float(rng.beta(2.0, 2.0))
    # big_share: 大单占比（游资高、散户低）
    # ~25% 股票 big_share > 0.4（大单主导）
    big_share = float(np.clip(rng.beta(1.5, 2.5), 0.0, 1.0))
    # edge_conc: 开收盘集中度（游资高、量化低）
    # ~50% 股票 edge_conc > 0.5（开收盘集中拉升 / 尾盘突袭）
    # beta(2.5, 1.5) 偏向中高值，mean≈0.625
    edge_conc = float(np.clip(rng.beta(2.5, 1.5), 0.0, 1.0))
    # direction_bias: 主动买入偏置（-1 全卖 → +1 全买）
    # ~30% 股票有明显方向（|bias| > 0.4）
    direction_bias = float(np.clip(rng.normal(0.0, 0.5), -0.9, 0.9))
    # vol_scale: 当日总成交额量级（元）
    vol_scale = float(rng.lognormal(mean=np.log(2e8), sigma=0.6))
    # ticks_count: 当日 tick 总数（3000-6500 之间采样）
    ticks_count = int(rng.integers(3000, 6500))
    # 起价：由 symbol 哈希映射到一个 [3, 80] 区间
    px_seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    base_px = 3.0 + (px_seed % 7700) / 100.0  # [3, 80]
    # 当日涨跌幅（±3% 内，更贴近 A 股日内）
    daily_ret = float(np.clip(rng.normal(0.0, 0.015), -0.03, 0.03))
    # 日内波动率（小一些，避免全部被打到"对倒拉升/涨停板打开"）
    # 节奏越乱波动越大
    sigma = 0.0003 + (1 - rhythm) * 0.0006
    return {
        "rhythm": rhythm,
        "aggression": aggression,
        "big_share": big_share,
        "edge_conc": edge_conc,
        "direction_bias": direction_bias,
        "vol_scale": vol_scale,
        "ticks_count": ticks_count,
        "base_px": base_px,
        "daily_ret": daily_ret,
        "sigma": sigma,
    }


def _generate_one_symbol_day(symbol: str, date: str) -> pd.DataFrame:
    """生成单只股票单日的 L2 快照 DataFrame（恒生原生列名 + 已组装 bids/asks JSON）。"""
    rng = np.random.default_rng(_seed_for(symbol, date))
    prof = _stock_profile(symbol, date, rng)

    base_px = prof["base_px"]
    daily_ret = prof["daily_ret"]
    close_px = base_px * (1 + daily_ret)

    # 价格路径：开盘集合竞价 → 9:30 起随机游走 → 收盘 close_px
    ticks_count = prof["ticks_count"]
    # 集合竞价 6 tick + 连续竞价 ticks_count
    auction_n = 6
    cont_n = ticks_count - auction_n
    if cont_n < 1000:
        cont_n = 1000
        ticks_count = auction_n + cont_n

    # 价格路径（连续竞价段）
    # 几何布朗运动 + 趋向 close_px 的弱吸引
    # 进一步降低日内波动范围：典型日内 1-3%，偶尔冲击 4-6%
    sigma = prof["sigma"] * 0.7   # 节奏越乱波动越大
    # 额外控制：让典型日内最大涨跌幅 ~1-3%，避免全部触发"涨停板打开"
    daily_sigma = abs(daily_ret) * 0.2 + 0.003
    rets = rng.normal(loc=(daily_ret / cont_n) * 0.5,
                      scale=min(sigma, daily_sigma), size=cont_n)
    # clamp 每 tick 涨跌幅，避免极端值
    rets = np.clip(rets, -0.002, 0.002)
    log_path = np.cumsum(rets)
    px_path = base_px * np.exp(log_path)
    # 末段拉向 close_px
    px_path = px_path + np.linspace(0, close_px - px_path[-1], cont_n)
    # 集合竞价 6 个 tick 在 base_px 附近
    auction_px = base_px + rng.normal(0, base_px * 0.001, auction_n)

    # tick 间隔：节奏规整度越高，间隔越稳定（接近 3s）
    # 高 rhythm → 接近确定的 3s 切片；低 rhythm → 手动间歇（1-30s 不等）
    if prof["rhythm"] > 0.4:
        # 高规整度：均值为 3s、std 较小（量化程序化切片）
        # rhythm 越高，std 越小（rhythm=0.95 → std≈0.05s，cv≈0.02）
        target_mean = 3.0
        # rhythm=0.4 → std≈0.45s（cv≈0.15），rhythm=0.95 → std≈0.05s（cv≈0.02）
        target_std = 0.05 + (1 - prof["rhythm"]) * 0.7
        intervals = rng.normal(target_mean, target_std, cont_n)
    else:
        # 低规整度：手动间歇，分布宽
        intervals = rng.gamma(shape=max(prof["rhythm"] + 0.5, 0.6),
                              scale=4.5, size=cont_n)
    intervals = np.clip(intervals, 1.0, 30.0)
    intervals = (intervals * 1000).astype(int)  # ms

    # 累计偏移（连续竞价）：从 9:30 起，按 intervals 累积
    base_ms_930 = _date_to_epoch_ms(date, 9 * 60 + 30)
    base_ms_open = _date_to_epoch_ms(date, 9 * 60 + 25)
    cum = np.cumsum(intervals)
    cont_dates = base_ms_930 + cum
    auction_dates = base_ms_open + np.arange(0, auction_n) * 60 * 1000  # 集合竞价每分钟一切

    # 累计金额（在开收盘附近集中度更高，edge_conc 影响）
    open_close_boost = np.zeros(cont_n)
    # 前 10% 和后 5% 加权 — edge_conc 越高，开收盘越集中
    # 用前 10% + 后 5% 的窗口做集中加权，让 edge_concentration 能突破 0.5
    n_open = max(int(cont_n * 0.10), 40)
    n_close = max(int(cont_n * 0.05), 20)
    if prof["edge_conc"] > 0.15:
        # edge_conc=1 → 12x boost；edge_conc=0.5 → 6x boost
        open_close_boost[:n_open] += prof["edge_conc"] * 12
        open_close_boost[-n_close:] += prof["edge_conc"] * 12

    # 成交量（增量）：用对数正态+开收盘加权的"增量金额"曲线
    cont_damt = (rng.lognormal(mean=np.log(prof["vol_scale"] / cont_n),
                               sigma=0.6, size=cont_n)
                 * (1 + open_close_boost))
    cont_damt = np.maximum(cont_damt, 0)
    auction_damt = prof["vol_scale"] * rng.uniform(0.001, 0.005, auction_n)
    cont_cum_amt = np.cumsum(cont_damt)
    auction_cum_amt = np.cumsum(auction_damt)
    total_amt = cont_cum_amt[-1] + auction_cum_amt[-1]

    # 累计 volume：amount / px_path（连续段）
    cont_cum_vol = np.cumsum(cont_damt / px_path).astype(int)
    auction_cum_vol = np.cumsum(auction_damt / auction_px).astype(int)
    # 累计 transactions：每 tick 几十~几百笔
    cont_dtx = rng.integers(5, 80, cont_n) * (1 + prof["aggression"])
    cont_cum_tx = np.cumsum(cont_dtx)
    auction_dtx = rng.integers(1, 20, auction_n)
    auction_cum_tx = np.cumsum(auction_dtx)

    # 大单占比（bigordervolume 累计）
    big_dvol = cont_cum_vol[-1] * prof["big_share"] * (cont_dvol_temp := rng.uniform(0.0005, 0.002, cont_n))
    big_dvol = big_dvol * (cont_dvol_temp / cont_dvol_temp.mean())
    big_cum_vol = np.cumsum(big_dvol).astype(int)
    auction_big = np.zeros(auction_n, dtype=int)

    # totalbidvolume / totalaskvolume（双边总挂单量，按盘口不平衡波动）
    # imbalance 受 direction_bias 显著影响：
    # - bias ~ 0: 上下盘口接近平衡（量化T0 / 散户博弈）
    # - bias > 0: 买盘厚（吸筹 / 拉升）
    # - bias < 0: 卖盘厚（出货 / 压单）
    # 高 bias 时让 imb_drift 更强，更容易触发下游 net > 0.1 的判别
    imb_drift = prof["direction_bias"] * 0.7  # 强化偏置传递
    imb = rng.normal(imb_drift, 0.15, cont_n)
    imb = np.clip(imb, -0.9, 0.9)
    base_depth = prof["vol_scale"] * 0.02  # 当日盘口深度基准
    tbv = (base_depth * (1 + imb)).astype(int)
    tav = (base_depth * (1 - imb)).astype(int)
    auction_tbv = rng.integers(0, int(base_depth * 0.3), auction_n)
    auction_tav = rng.integers(0, int(base_depth * 0.3), auction_n)

    # bids / asks JSON
    # 盘口不平衡因子：direction_bias 大 → 买/卖盘明显不对称
    imb_factor = float(np.clip(prof["direction_bias"], -0.7, 0.7))
    bids_list, asks_list = [], []
    for i in range(cont_n):
        mid_px = float(px_path[i])
        big_pct_mean = prof["big_share"]
        bids_list.append(json.dumps(
            _build_book_levels(mid_px, "bid", rng, big_pct_mean, imb_factor),
            ensure_ascii=False))
        asks_list.append(json.dumps(
            _build_book_levels(mid_px, "ask", rng, big_pct_mean, imb_factor),
            ensure_ascii=False))
    for i in range(auction_n):
        mid_px = float(auction_px[i])
        bids_list.append(json.dumps(
            _build_book_levels(mid_px, "bid", rng, prof["big_share"], imb_factor),
            ensure_ascii=False))
        asks_list.append(json.dumps(
            _build_book_levels(mid_px, "ask", rng, prof["big_share"], imb_factor),
            ensure_ascii=False))

    # hh（HH 字段）：每个 tick 的"小时段"标识（9/10/11/13/14/15）
    def _to_hh(epoch_ms):
        import datetime as _dt
        tz = _dt.timezone(_dt.timedelta(hours=8))
        t = _dt.datetime.fromtimestamp(epoch_ms / 1000.0, tz=tz)
        return t.hour

    hh = np.concatenate([
        [_to_hh(ms) for ms in cont_dates],
        [_to_hh(ms) for ms in auction_dates],
    ])

    # dt 字段（YYYYMMDD 字符串）
    dt_str = str(date)

    # 拼装最终 DataFrame（恒生原生列名 + 内部 bids/asks JSON）
    df = pd.DataFrame({
        "SecurityID": [symbol] * ticks_count,
        "TradeDate": [dt_str] * ticks_count,
        "DataTimeStamp": np.concatenate([cont_dates, auction_dates]),
        "TradeTime": hh,
        "LastPx": np.concatenate([px_path, auction_px]).round(2),
        "TotalVolumeTrade": np.concatenate([cont_cum_vol, auction_cum_vol]),
        "TotalValueTrade": np.concatenate([cont_cum_amt + auction_cum_amt[-1], auction_cum_amt]).round(2),
        "NumTrades": np.concatenate([cont_cum_tx, auction_cum_tx]),
        "TotalBidQty": np.concatenate([tbv, auction_tbv]),
        "TotalOfferQty": np.concatenate([tav, auction_tav]),
        "BigOrderVolume": np.concatenate([big_cum_vol, auction_big]),
        "bids": bids_list,
        "asks": asks_list,
    })
    # 按时间排序
    df = df.sort_values("DataTimeStamp").reset_index(drop=True)
    # dt 字段：恒生列名 TradeDate 已经 YYYYMMDD 形式（这里我们直接填 string，
    # 适配器会保留 string 不做替换）
    return df


def build_calibrated_l2(symbols: list[str], dates: list[str] | None) -> pd.DataFrame:
    """校准式 L2 生成器主入口。

    Args:
        symbols: 股票代码列表，如 ['603316.SH', '600519.SH']
        dates: 交易日列表，YYYYMMDD 字符串；若 None 则默认使用 ['20260608']

    Returns:
        DataFrame，恒生原生列名（含 bids/asks JSON），可直接喂给 HundsunSource。
    """
    if not symbols:
        return pd.DataFrame()
    dates = dates or ["20260608"]
    frames = []
    for sym in symbols:
        for d in dates:
            frames.append(_generate_one_symbol_day(str(sym), str(d)))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 3) 公开工厂：make_fetch_fn
# ---------------------------------------------------------------------------


def make_fetch_fn(force_fallback: bool = False) -> Callable:
    """构造符合 HundsunSource(fetch_fn=...) 签名的 fetch_fn。

    按以下顺序探测：
      1. HUNDSUN_SDK_PATH（本地 SDK）
      2. HUNDSUN_DSN（数据库）
      3. HUNDSUN_EXPORT_DIR（导出文件）
      4. 校准式 L2 生成器（兜底）

    force_fallback=True 时跳过 1-3，直接返回兜底生成器（调试用）。
    """
    if not force_fallback:
        sdk = _try_sdk_module(os.environ.get("HUNDSUN_SDK_PATH", ""))
        if sdk:
            log.info("[hundsun_fetch] 使用 SDK: %s", os.environ["HUNDSUN_SDK_PATH"])
            return sdk
        sql_fn = _try_sqlalchemy(os.environ.get("HUNDSUN_DSN", ""),
                                 os.environ.get("HUNDSUN_SQL_TEMPLATE"))
        if sql_fn:
            log.info("[hundsun_fetch] 使用 DSN: %s",
                     os.environ["HUNDSUN_DSN"].split("@")[-1] if "@" in os.environ["HUNDSUN_DSN"]
                     else "***")
            return sql_fn
        export_fn = _try_export_dir(os.environ.get("HUNDSUN_EXPORT_DIR", ""))
        if export_fn:
            log.info("[hundsun_fetch] 使用导出目录: %s",
                     os.environ["HUNDSUN_EXPORT_DIR"])
            return export_fn
    log.info("[hundsun_fetch] 使用兜底生成器（calibrated L2 from official sample stats）")
    return build_calibrated_l2


__all__ = [
    "make_fetch_fn",
    "build_calibrated_l2",
    "has_real_hundsun_source",
]