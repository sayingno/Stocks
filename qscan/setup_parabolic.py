"""Setup #3: the parabolic short.

A stock goes vertical — several hundred percent in a few weeks, gapping, closing
at the highs, miles above every moving average — and then cracks. The short is
taken on the first real sign of weakness, not on the way up.

This is the setup that hurts people, and it is worth being explicit about why:

* **Timing is everything.** Shorting strength is how accounts die. The detector
  refuses to fire until the day itself shows the break — a reversal off the
  highs, a close below the prior low, or a red day after a run of green.
* **Risk is uncapped in principle.** A long can only go to zero; a short can go
  to any number. The 1x ADR stop rule matters more here, not less.
* **Borrow may not exist.** The scanner cannot see locate availability or
  borrow cost. A signal here is not the same as a fill.

Mechanically it is the mirror of the other two: entry on the break of the day's
low, stop above the day's high capped at 1 ADR, cover into the flush.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .common import SHORT, band_score, clamp01, reject, tradable, trade_plan, weighted
from .config import ParabolicConfig
from .setup_breakout import _Arrays

NAME = "parabolic"
DIRECTION = SHORT


def evaluate_bar(a: _Arrays, i: int, cfg: ParabolicConfig, rs: np.ndarray | None = None) -> dict[str, Any]:
    """Evaluate the parabolic short as of bar `i`."""
    warmup = max(cfg.ma_slow, cfg.run_lookback + 25)
    if i < warmup:
        return reject("warmup")

    close = a.close[i]
    adr = a.adr20[i]

    f: dict[str, Any] = {
        "setup": NAME,
        "date": pd.Timestamp(a.date[i]),
        "close": close,
        "adr20": adr,
        "dollar_vol20": a.dollar_vol20[i],
    }

    bad = tradable(close, a.dollar_vol20[i], adr, cfg)
    if bad:
        return reject(bad, **f)

    # ---- the parabolic run ---------------------------------------------------
    # Measured from the lowest low in the lookback to the highest high, so a
    # move that happened inside the window still counts even if today has
    # already given some back.
    win_low_pos = i - cfg.run_lookback + int(np.argmin(a.low[i - cfg.run_lookback : i + 1]))
    run_high = float(np.max(a.high[win_low_pos : i + 1]))
    run_low = float(a.low[win_low_pos])
    run_gain = run_high / run_low - 1.0 if run_low > 0 else 0.0
    f["run_gain"] = run_gain
    f["run_bars"] = i - win_low_pos
    if run_gain < cfg.min_run_gain:
        return reject("not_parabolic", **f)
    if f["run_bars"] > cfg.max_run_bars:
        return reject("run_too_slow", **f)

    # ---- extension from the mean --------------------------------------------
    ma_m = a.ma_mid[i]
    if not math.isfinite(ma_m) or ma_m <= 0:
        return reject("nan_ma", **f)
    extension = close / ma_m - 1.0
    f["extension"] = extension
    f["ma_mid"] = ma_m
    if extension < cfg.min_extension:
        return reject("not_extended", **f)

    # ---- climax volume -------------------------------------------------------
    # The baseline has to be the volume *before* the run began. Using a trailing
    # 20 days would compare the run against itself and always read ~1x.
    base_from = max(0, win_low_pos - 20)
    baseline_vol = float(np.nanmean(a.volume[base_from:win_low_pos])) if win_low_pos > base_from else float("nan")
    run_vol = float(np.nanmean(a.volume[win_low_pos : i + 1]))
    vol_mult = run_vol / baseline_vol if baseline_vol > 0 else float("nan")
    f["vol_mult"] = vol_mult
    if math.isfinite(vol_mult) and vol_mult < cfg.min_volume_mult:
        return reject("no_climax_volume", **f)

    # ---- the crack: today must actually show weakness ------------------------
    # Without this the scanner would hand you a short every day of the ramp.
    day_range = a.high[i] - a.low[i]
    close_pos = (close - a.low[i]) / day_range if day_range > 0 else 0.5
    reversal = close_pos <= cfg.max_close_position and a.high[i] >= run_high * (1 - 0.02)
    broke_prior_low = a.low[i] < a.low[i - 1] and close < a.close[i - 1]
    red_after_green = close < a.open[i] and _green_streak(a, i - 1) >= cfg.min_green_streak

    f["close_position"] = close_pos
    f["reversal_bar"] = bool(reversal)
    f["broke_prior_low"] = bool(broke_prior_low)
    f["red_after_green"] = bool(red_after_green)

    if not (reversal or broke_prior_low or red_after_green):
        return reject("no_weakness_yet", **f)

    f["days_off_high"] = int(i - (win_low_pos + int(np.argmax(a.high[win_low_pos : i + 1]))))
    if f["days_off_high"] > cfg.max_days_off_high:
        return reject("too_late", **f)

    if rs is not None and i < len(rs):
        f["rs_rank"] = float(rs[i])

    # ---- trade ---------------------------------------------------------------
    # Short the break of the day's low; stop above the day's high, capped at 1 ADR.
    f["state"] = "short"
    f.update(trade_plan(entry=float(a.low[i]), raw_stop=float(a.high[i]), adr_pct=adr, direction=SHORT, cfg=cfg))
    f["score"] = _score(f, cfg)
    f["passed"] = True
    f["reject"] = ""
    return f


def _green_streak(a: _Arrays, i: int) -> int:
    """How many consecutive up-closes end at bar i."""
    n = 0
    while i - n >= 1 and a.close[i - n] > a.close[i - n - 1]:
        n += 1
        if n > 30:
            break
    return n


def _score(f: dict[str, Any], cfg: ParabolicConfig) -> float:
    confirmations = sum((f["reversal_bar"], f["broke_prior_low"], f["red_after_green"]))
    parts = {
        "run": band_score(f["run_gain"], best=3.0, worst=cfg.min_run_gain),
        "extension": band_score(f["extension"], best=1.0, worst=cfg.min_extension),
        "volume": band_score(f["vol_mult"], best=6.0, worst=cfg.min_volume_mult),
        # Selling into the top of the range is the whole game; a weak close is better.
        "weakness": clamp01(1.0 - f["close_position"]),
        "confirmation": confirmations / 3.0,
        "timing": band_score(float(f["days_off_high"]), best=0.0, worst=float(cfg.max_days_off_high)),
        "adr": band_score(f["adr20"], best=12.0, worst=cfg.min_adr_pct),
    }
    return weighted(
        parts,
        {"run": 0.22, "extension": 0.20, "volume": 0.12, "weakness": 0.16,
         "confirmation": 0.12, "timing": 0.10, "adr": 0.08},
    )
