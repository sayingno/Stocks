"""Setup #2: the Episodic Pivot.

A stock that has been dead for months gaps up hard on a genuine surprise —
earnings, a contract, a drug result — and opens a new chapter. The gap is the
market repricing the business in one print, and the move that follows is
institutions building positions they could not build quietly.

Where this differs from the breakout is instructive, because on two axes it is
the exact opposite:

                        breakout                episodic pivot
    prior move          +30-100% required       must have gone *nowhere*
    volume in the base  must dry up             must explode on the day
    the trigger         clears a pivot          gaps clean out of a base

The rest — liquidity, ADR, the opening-range entry, the stop at the low of the
day capped at 1 ADR, selling a third to a half into the first burst and
trailing the remainder on a moving average — is identical.

The dormancy requirement is the part people skip, and it is the part that
matters: a 12% gap in something already up 200% this quarter is a blow-off, not
an episodic pivot.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .common import LONG, band_score, clamp01, reject, tradable, trade_plan, weighted
from .config import EPConfig
from .setup_breakout import _Arrays

NAME = "ep"
DIRECTION = LONG


def evaluate_bar(a: _Arrays, i: int, cfg: EPConfig, rs: np.ndarray | None = None) -> dict[str, Any]:
    """Evaluate the episodic pivot as of bar `i`."""
    warmup = max(cfg.ma_slow, cfg.dormancy_lookback + 25)
    if i < warmup:
        return reject("warmup")

    close = a.close[i]
    adr = a.adr20[i]
    prev_close = a.close[i - 1]

    f: dict[str, Any] = {
        "setup": NAME,
        "date": pd.Timestamp(a.date[i]),
        "close": close,
        "adr20": adr,
        "dollar_vol20": a.dollar_vol20[i],
    }

    # ---- tradable ------------------------------------------------------------
    # Liquidity is measured on the *prior* 20 days: the gap day's own volume is
    # enormous by construction, so including it would let a one-day spike in a
    # normally untradable stock through the filter.
    prior_dollar_vol = float(np.nanmean(a.close[i - 20 : i] * a.volume[i - 20 : i]))
    f["dollar_vol20"] = prior_dollar_vol
    bad = tradable(close, prior_dollar_vol, adr, cfg)
    if bad:
        return reject(bad, **f)

    # ---- the gap -------------------------------------------------------------
    if prev_close <= 0 or not math.isfinite(prev_close):
        return reject("nan_inputs", **f)
    gap = a.open[i] / prev_close - 1.0
    f["gap"] = gap
    if gap < cfg.min_gap:
        return reject("gap_too_small", **f)

    # ---- volume --------------------------------------------------------------
    prior_avg_vol = float(np.nanmean(a.volume[i - 20 : i]))
    vol_mult = a.volume[i] / prior_avg_vol if prior_avg_vol > 0 else float("nan")
    f["vol_mult"] = vol_mult
    if not math.isfinite(vol_mult) or vol_mult < cfg.min_volume_mult:
        return reject("no_volume_surge", **f)

    # ---- dormancy: it must have been going nowhere --------------------------
    # Measured to the bar *before* the gap, so the gap itself cannot satisfy the
    # very condition it is supposed to be breaking.
    look = cfg.dormancy_lookback
    window_close = a.close[i - look : i]
    window_high = float(np.max(a.high[i - look : i]))
    window_low = float(np.min(a.low[i - look : i]))

    prior_gain = prev_close / float(window_close[0]) - 1.0 if window_close[0] > 0 else float("inf")
    f["prior_gain"] = prior_gain
    if prior_gain > cfg.max_prior_gain:
        return reject("already_extended", **f)

    band = (window_high - window_low) / window_high if window_high > 0 else 1.0
    f["dormant_band"] = band
    if band > cfg.max_dormant_band:
        return reject("not_dormant", **f)

    # ---- run-up into the event, per horizon ---------------------------------
    # Measured to the bar before the gap, so the gap itself is never part of
    # "what did it look like going in".
    for label, bars in (("3d", 3), ("1w", 5), ("1m", 21), ("3m", 63)):
        bounds = getattr(cfg, f"perf_{label}", None)
        prior = a.close[i - 1 - bars] if i - 1 - bars >= 0 else float("nan")
        value = (prev_close / prior - 1.0) if prior > 0 and math.isfinite(prior) else float("nan")
        f[f"pre_ret_{label}"] = value
        if bounds is None:
            continue
        lo_b, hi_b = bounds
        if not math.isfinite(value):
            return reject(f"no_perf_{label}", **f)
        if lo_b is not None and value < lo_b:
            return reject(f"perf_{label}_too_low", **f)
        if hi_b is not None and value > hi_b:
            return reject(f"perf_{label}_too_high", **f)

    # ---- the gap must clear the base ----------------------------------------
    f["base_high"] = window_high
    if close < window_high:
        return reject("still_inside_base", **f)

    # ---- it has to hold the gap ---------------------------------------------
    day_range = a.high[i] - a.low[i]
    close_pos = (close - a.low[i]) / day_range if day_range > 0 else 0.5
    f["close_position"] = close_pos
    if close_pos < cfg.min_close_position:
        return reject("closed_weak", **f)
    if cfg.require_close_above_open and close < a.open[i]:
        return reject("closed_red", **f)

    # ---- structure -----------------------------------------------------------
    ma_s = a.ma_slow[i]
    f["ma_slow"] = ma_s
    if math.isfinite(ma_s) and close < ma_s:
        return reject("below_slow_ma", **f)

    if rs is not None and i < len(rs):
        f["rs_rank"] = float(rs[i])
        if cfg.min_rs_rank is not None and not (float(rs[i]) >= cfg.min_rs_rank):
            return reject("weak_rs", **f)
    elif cfg.min_rs_rank is not None:
        return reject("no_rs_data", **f)

    # ---- trade ---------------------------------------------------------------
    # Daily bars cannot see the opening range, so the day's high stands in for
    # the 1/5/60-minute opening-range high he actually buys.
    #
    # The stop needs care here. The ADR cap exists to stop you risking an absurd
    # amount, but on an episodic pivot the 20-day ADR describes the *dormant*
    # stock that no longer exists — the gap has just repriced its volatility.
    # Capping at 3% when the stock now swings 8% a day guarantees you are shaken
    # out by noise on the next bar. So the reference volatility is the wider of
    # the trailing ADR and the day's own range, which in practice puts the stop
    # at the low of the day: exactly where he says it goes.
    day_range_pct = (a.high[i] - a.low[i]) / a.low[i] * 100.0 if a.low[i] > 0 else adr
    stop_adr = max(adr, day_range_pct)
    f["stop_adr_ref"] = stop_adr
    f["state"] = "ep"
    f.update(
        trade_plan(entry=float(a.high[i]), raw_stop=float(a.low[i]), adr_pct=stop_adr, direction=LONG, cfg=cfg)
    )
    f["score"] = _score(f, cfg)
    f["passed"] = True
    f["reject"] = ""
    return f


def _score(f: dict[str, Any], cfg: EPConfig) -> float:
    parts = {
        # A bigger gap is a bigger surprise, but past ~40% it stops meaning more.
        "gap": band_score(f["gap"], best=0.40, worst=cfg.min_gap),
        "volume": band_score(f["vol_mult"], best=10.0, worst=cfg.min_volume_mult),
        # Deader beforehand is better.
        "dormancy": band_score(f["dormant_band"], best=0.10, worst=cfg.max_dormant_band),
        "not_extended": band_score(f["prior_gain"], best=-0.10, worst=cfg.max_prior_gain),
        "close_strength": clamp01(f["close_position"]),
        "adr": band_score(f["adr20"], best=8.0, worst=cfg.min_adr_pct),
    }
    return weighted(
        parts,
        {"gap": 0.24, "volume": 0.22, "dormancy": 0.18, "not_extended": 0.14,
         "close_strength": 0.12, "adr": 0.10},
    )
