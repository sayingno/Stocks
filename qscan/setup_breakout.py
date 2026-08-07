"""Detector for Qullamaggie setup #1: the Breakout (continuation) setup.

The setup, in one sentence: a stock that already made a big move up in the last
1-3 months, then consolidated in an orderly, tightening range while riding its
10/20/50-day moving averages, and is now pressing against - or clearing - the
high of that consolidation.

This module turns that sentence into measurable gates:

  1. tradable        price / dollar volume / ADR
  2. prior move      +30% in 1m, or +50% in 3m, or +100% in 6m (configurable)
  3. base found      pivot high 3-60 bars back, depth within limits
  4. orderly         range contracts, lows hold up, volume dries out
  5. MA structure    above the 50MA, stacked and rising, price hugging a MA
  6. coiled          price within striking distance of the pivot
  7. trigger         breakout today, or "setup" (ready and waiting)

Every gate records the value it measured, so a rejected name tells you which
condition it missed rather than just disappearing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import BreakoutConfig
from .indicators import annotate

# Bars searched backwards from the pivot when locating the start of the
# impulse leg that created the base.
IMPULSE_LOOKBACK = 90


@dataclass(frozen=True)
class _Arrays:
    """Column-major numpy view of an annotated frame - fast to index in a loop."""

    date: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    ma_fast: np.ndarray
    ma_mid: np.ndarray
    ma_slow: np.ndarray
    ma_mid_slope: np.ndarray
    ma_slow_slope: np.ndarray
    adr20: np.ndarray
    dollar_vol20: np.ndarray
    vol_avg20: np.ndarray
    range_pct: np.ndarray

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "_Arrays":
        get = lambda c: df[c].to_numpy(dtype=float)  # noqa: E731
        return cls(
            date=df.index.to_numpy(),
            open=get("open"),
            high=get("high"),
            low=get("low"),
            close=get("close"),
            volume=get("volume"),
            ma_fast=get("ma_fast"),
            ma_mid=get("ma_mid"),
            ma_slow=get("ma_slow"),
            ma_mid_slope=get("ma_mid_slope"),
            ma_slow_slope=get("ma_slow_slope"),
            adr20=get("adr20"),
            dollar_vol20=get("dollar_vol20"),
            vol_avg20=get("vol_avg20"),
            range_pct=get("range_pct"),
        )

    def __len__(self) -> int:
        return len(self.close)


def _clamp01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _lsq_slope_pct(values: np.ndarray) -> float:
    """Least-squares slope of `values`, in percent of the mean per bar."""
    n = len(values)
    if n < 3:
        return 0.0
    mean = float(values.mean())
    if not math.isfinite(mean) or mean == 0.0:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    slope = float((x * values).sum() / (x * x).sum())
    return slope / mean * 100.0


def _reject(reason: str, **fields: Any) -> dict[str, Any]:
    return {"passed": False, "reject": reason, **fields}


def evaluate_bar(a: _Arrays, i: int, cfg: BreakoutConfig) -> dict[str, Any]:
    """Evaluate the breakout setup as of bar `i` (which is treated as 'today').

    Returns a dict that always carries `passed` and, when False, a `reject`
    reason naming the first gate that failed.
    """
    warmup = max(cfg.ma_slow, 126, cfg.max_base_len + 5)
    if i < warmup:
        return _reject("warmup")

    close = a.close[i]
    adr = a.adr20[i]
    if not math.isfinite(close) or not math.isfinite(adr):
        return _reject("nan_inputs")

    base_fields: dict[str, Any] = {
        "date": pd.Timestamp(a.date[i]),
        "close": close,
        "adr20": adr,
        "dollar_vol20": a.dollar_vol20[i],
    }

    # ---- 1. tradable --------------------------------------------------------
    if close < cfg.min_price:
        return _reject("price", **base_fields)
    if not (a.dollar_vol20[i] >= cfg.min_dollar_volume):
        return _reject("liquidity", **base_fields)
    if not (cfg.min_adr_pct <= adr <= cfg.max_adr_pct):
        return _reject("adr", **base_fields)

    # ---- 2. the prior move --------------------------------------------------
    best_ratio = 0.0
    move_returns: dict[str, float] = {}
    for n, threshold in cfg.move_lookbacks:
        if i - n < 0:
            continue
        prior = a.close[i - n]
        if prior <= 0 or not math.isfinite(prior):
            continue
        ret = close / prior - 1.0
        move_returns[f"ret_{n}d"] = ret
        if threshold > 0:
            best_ratio = max(best_ratio, ret / threshold)
    base_fields.update(move_returns)
    base_fields["move_ratio"] = best_ratio
    if best_ratio < 1.0:
        return _reject("no_prior_move", **base_fields)

    # ---- 3. locate the base -------------------------------------------------
    # The base is measured on bars strictly before today, so that "today broke
    # out of it" stays a meaningful statement.
    win_start = max(0, i - cfg.max_base_len)
    if i - win_start < cfg.min_base_len:
        return _reject("warmup", **base_fields)
    window_high = a.high[win_start:i]
    pivot_off = int(np.argmax(window_high))
    pivot_pos = win_start + pivot_off
    base_high = float(a.high[pivot_pos])
    base_len = i - pivot_pos  # bars from the pivot bar up to yesterday

    base_fields["pivot"] = base_high
    base_fields["pivot_date"] = pd.Timestamp(a.date[pivot_pos])
    base_fields["base_len"] = base_len

    if base_len < cfg.min_base_len:
        return _reject("base_too_short", **base_fields)
    if base_len > cfg.max_base_len:
        return _reject("base_too_long", **base_fields)

    # Depth is measured after the pivot bar so a single wide impulse candle
    # does not define the low of the base.
    post = slice(pivot_pos + 1, i)
    base_low = float(np.min(a.low[post])) if i - pivot_pos > 1 else float(a.low[pivot_pos])
    base_depth = (base_high - base_low) / base_high if base_high > 0 else 1.0
    base_fields["base_low"] = base_low
    base_fields["base_depth"] = base_depth
    if base_depth > cfg.max_base_depth:
        return _reject("base_too_deep", **base_fields)

    # ---- 4. orderly: contraction, holding lows, volume drying out -----------
    base_slice = slice(pivot_pos, i)
    base_ranges = a.range_pct[base_slice]
    contraction = float("nan")
    if base_len >= 6:
        half = base_len // 2
        first = float(np.nanmean(base_ranges[:half]))
        second = float(np.nanmean(base_ranges[half:]))
        if first > 0:
            contraction = second / first
    base_fields["contraction"] = contraction
    if math.isfinite(contraction) and contraction > cfg.max_contraction_ratio:
        return _reject("no_contraction", **base_fields)

    low_slope = _lsq_slope_pct(a.low[base_slice])
    base_fields["low_slope"] = low_slope
    if low_slope < -0.75:
        return _reject("lows_breaking_down", **base_fields)

    # Impulse leg = from the swing low that preceded the pivot, to the pivot.
    imp_start = max(0, pivot_pos - IMPULSE_LOOKBACK)
    imp_low_pos = imp_start + int(np.argmin(a.low[imp_start : pivot_pos + 1]))
    impulse_gain = base_high / float(a.low[imp_low_pos]) - 1.0
    base_fields["impulse_gain"] = impulse_gain
    base_fields["impulse_start_date"] = pd.Timestamp(a.date[imp_low_pos])

    bars_since_impulse = i - pivot_pos
    if bars_since_impulse > cfg.max_bars_since_impulse:
        return _reject("stale_move", **base_fields)

    imp_vol = a.volume[imp_low_pos : pivot_pos + 1]
    back_half = a.volume[pivot_pos + max(1, base_len // 2) : i]
    vol_ratio = float("nan")
    if imp_vol.size and back_half.size:
        imp_mean = float(np.nanmean(imp_vol))
        if imp_mean > 0:
            vol_ratio = float(np.nanmean(back_half)) / imp_mean
    base_fields["vol_dryup"] = vol_ratio
    if math.isfinite(vol_ratio) and vol_ratio > cfg.max_volume_dryup_ratio:
        return _reject("volume_not_drying_up", **base_fields)

    # ---- 5. moving average structure ---------------------------------------
    ma_f, ma_m, ma_s = a.ma_fast[i], a.ma_mid[i], a.ma_slow[i]
    base_fields.update({"ma_fast": ma_f, "ma_mid": ma_m, "ma_slow": ma_s})
    if not all(math.isfinite(v) for v in (ma_f, ma_m, ma_s)):
        return _reject("nan_ma", **base_fields)
    if cfg.require_above_slow_ma and close < ma_s:
        return _reject("below_slow_ma", **base_fields)
    if cfg.require_ma_stack and not (ma_f > ma_m > ma_s):
        return _reject("ma_not_stacked", **base_fields)
    if cfg.require_rising_mid_ma and not (a.ma_mid_slope[i] > 0):
        return _reject("mid_ma_not_rising", **base_fields)

    # "Surfing": across the base, how tightly did close track one of the MAs?
    base_close = a.close[base_slice]
    surf_dist, surf_ma = float("inf"), ""
    for name, series in (
        (f"ma{cfg.ma_fast}", a.ma_fast),
        (f"ma{cfg.ma_mid}", a.ma_mid),
        (f"ma{cfg.ma_slow}", a.ma_slow),
    ):
        seg = series[base_slice]
        with np.errstate(invalid="ignore", divide="ignore"):
            d = float(np.nanmean(np.abs(base_close - seg) / base_close))
        if math.isfinite(d) and d < surf_dist:
            surf_dist, surf_ma = d, name
    base_fields["surf_ma"] = surf_ma
    base_fields["surf_dist"] = surf_dist
    if surf_dist > cfg.max_surf_distance:
        return _reject("not_surfing_ma", **base_fields)

    # ---- 6. coiled near the pivot ------------------------------------------
    dist_from_pivot = max(0.0, (base_high - close) / base_high)
    base_fields["dist_from_pivot"] = dist_from_pivot

    # ---- 7. trigger ---------------------------------------------------------
    vol_mult = a.volume[i] / a.vol_avg20[i] if a.vol_avg20[i] > 0 else float("nan")
    base_fields["vol_mult"] = vol_mult
    broke_out = (
        a.high[i] > base_high
        and close > base_high
        and (not math.isfinite(vol_mult) or vol_mult >= cfg.breakout_volume_mult)
    )
    if broke_out:
        state = "breakout"
    elif dist_from_pivot <= cfg.max_dist_from_pivot:
        state = "setup"
    else:
        return _reject("far_from_pivot", **base_fields)
    base_fields["state"] = state

    base_fields.update(_trade_plan(a, i, base_high, adr, state, cfg))
    base_fields["score"] = _score(base_fields, cfg)
    base_fields["passed"] = True
    base_fields["reject"] = ""
    return base_fields


def _trade_plan(
    a: _Arrays, i: int, pivot: float, adr: float, state: str, cfg: BreakoutConfig
) -> dict[str, Any]:
    """Entry / stop / size, with Qullamaggie's 'stop no wider than 1 ADR' rule.

    On a real breakout day the entry is the opening-range high (1/5/60 min),
    which daily bars cannot see. The daily proxy is the pivot itself; the stop
    proxy is the low of the day, tightened to at most `max_stop_adr_mult` ADRs.
    """
    entry = pivot * 1.001 if state == "setup" else max(pivot, a.open[i])
    if state == "breakout":
        raw_stop = float(a.low[i])
    else:
        raw_stop = float(np.min(a.low[max(0, i - 2) : i + 1]))

    adr_floor = entry * (1.0 - (adr / 100.0) * cfg.max_stop_adr_mult)
    stop = max(raw_stop, adr_floor)
    risk_ps = entry - stop
    if risk_ps <= 0:
        return {"entry": entry, "stop": stop, "risk_pct": float("nan"), "shares": 0}

    risk_dollars = cfg.account_size * cfg.risk_pct
    shares_by_risk = risk_dollars / risk_ps
    shares_by_cap = (cfg.account_size * cfg.max_position_pct) / entry
    shares = int(min(shares_by_risk, shares_by_cap))

    return {
        "entry": entry,
        "stop": stop,
        "raw_stop": raw_stop,
        "stop_capped_by_adr": adr_floor > raw_stop,
        "risk_pct": risk_ps / entry,
        "risk_in_adr": (risk_ps / entry) / (adr / 100.0) if adr > 0 else float("nan"),
        "shares": shares,
        "position_value": shares * entry,
        "target_2r": entry + 2 * risk_ps,
        "target_3r": entry + 3 * risk_ps,
    }


def _score(f: dict[str, Any], cfg: BreakoutConfig) -> float:
    """Blend the measured qualities into a 0-100 ranking score."""
    depth_span = max(1e-9, cfg.max_base_depth - cfg.ideal_base_depth)
    contraction = f.get("contraction")
    vol_dryup = f.get("vol_dryup")
    base_len = f.get("base_len", 0)

    parts = {
        "momentum": _clamp01((f["move_ratio"] - 0.8) / 1.2),
        "tightness": _clamp01((cfg.max_base_depth - f["base_depth"]) / depth_span),
        "contraction": (
            _clamp01((cfg.max_contraction_ratio - contraction) / max(1e-9, cfg.max_contraction_ratio - 0.45))
            if contraction is not None and math.isfinite(contraction)
            else 0.6
        ),
        "volume": (
            _clamp01((cfg.max_volume_dryup_ratio - vol_dryup) / max(1e-9, cfg.max_volume_dryup_ratio - 0.35))
            if vol_dryup is not None and math.isfinite(vol_dryup)
            else 0.6
        ),
        "surf": _clamp01((cfg.max_surf_distance - f["surf_dist"]) / max(1e-9, cfg.max_surf_distance - 0.015)),
        "proximity": _clamp01((cfg.max_dist_from_pivot - f["dist_from_pivot"]) / max(1e-9, cfg.max_dist_from_pivot)),
        "adr": _clamp01((f["adr20"] - cfg.min_adr_pct) / max(1e-9, 8.0 - cfg.min_adr_pct)),
        "base_len": 1.0 if 5 <= base_len <= 40 else _clamp01(1.0 - abs(base_len - 22) / 45.0),
    }
    weights = {
        "momentum": 0.25,
        "tightness": 0.15,
        "contraction": 0.12,
        "volume": 0.10,
        "surf": 0.13,
        "proximity": 0.10,
        "adr": 0.10,
        "base_len": 0.05,
    }
    return round(100.0 * sum(parts[k] * w for k, w in weights.items()), 1)


def scan_symbol(
    df: pd.DataFrame,
    cfg: BreakoutConfig,
    symbol: str = "",
    positions: Iterable[int] | None = None,
    keep_rejects: bool = False,
) -> list[dict[str, Any]]:
    """Run the detector over one symbol.

    `positions` defaults to the last bar only. Pass a range to sweep history.
    """
    ann = annotate(df, cfg)
    if len(ann) == 0:
        return []
    a = _Arrays.from_frame(ann)
    idx = list(positions) if positions is not None else [len(a) - 1]

    hits: list[dict[str, Any]] = []
    for i in idx:
        if i < 0 or i >= len(a):
            continue
        res = evaluate_bar(a, i, cfg)
        if res["passed"] or keep_rejects:
            res["symbol"] = symbol
            res.setdefault("date", pd.Timestamp(a.date[i]))
            res["bar"] = i
            hits.append(res)
    return hits


def dedupe_signals(hits: list[dict[str, Any]], cooldown: int = 10) -> list[dict[str, Any]]:
    """Collapse a run of consecutive signals into the first bar of each cluster.

    A base that stays valid for two weeks would otherwise emit ten near-identical
    rows; for a historical study you want one row per distinct setup.
    """
    out: list[dict[str, Any]] = []
    last_by_symbol: dict[str, int] = {}
    for h in sorted(hits, key=lambda r: (r.get("symbol", ""), r["bar"])):
        sym = h.get("symbol", "")
        prev = last_by_symbol.get(sym)
        if prev is None or h["bar"] - prev > cooldown:
            out.append(h)
        last_by_symbol[sym] = h["bar"]
    return sorted(out, key=lambda r: (r["date"], r.get("symbol", "")))
