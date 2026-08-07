"""What happened after the signal.

Two views of the same signal:

  forward_returns   plain +5/+10/+20 day returns and best/worst excursion
  simulate_trade    Qullamaggie's actual management - stop at the low of the
                    day (capped at 1 ADR), sell a third-to-a-half into the
                    first burst of strength, stop to breakeven, trail the rest
                    on the 10 or 20 day moving average

`simulate_trade` is deliberately pessimistic about fills: gaps through the stop
fill at the open, never at the stop price.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeRules:
    trigger_window: int = 10  # bars a "setup" gets to actually break out
    partial_days: int = 4  # sell into strength 3-5 days in
    partial_at_r: float = 3.0  # ...or sooner if it prints this many R
    partial_fraction: float = 0.5  # sell half
    breakeven_after_partial: bool = True
    trail_ma: int = 20  # exit remainder on first close below this MA
    max_hold: int = 120
    max_stop_adr_mult: float = 1.0


def forward_returns(ann: pd.DataFrame, bar: int, horizons=(5, 10, 20, 60)) -> dict[str, Any]:
    close = ann["close"].to_numpy(dtype=float)
    high = ann["high"].to_numpy(dtype=float)
    low = ann["low"].to_numpy(dtype=float)
    n = len(close)
    base = close[bar]
    out: dict[str, Any] = {}
    for h in horizons:
        j = bar + h
        out[f"fwd_{h}d"] = (close[j] / base - 1.0) if j < n else float("nan")
    window = slice(bar + 1, min(n, bar + 21))
    if window.start < n:
        out["mfe_20d"] = float(np.max(high[window])) / base - 1.0
        out["mae_20d"] = float(np.min(low[window])) / base - 1.0
    else:
        out["mfe_20d"] = out["mae_20d"] = float("nan")
    return out


def simulate_trade(
    ann: pd.DataFrame,
    signal: dict[str, Any],
    rules: TradeRules = TradeRules(),
) -> dict[str, Any]:
    """Walk the trade forward bar by bar. Returns R multiple and exit reason."""
    o = ann["open"].to_numpy(dtype=float)
    h = ann["high"].to_numpy(dtype=float)
    lo = ann["low"].to_numpy(dtype=float)
    c = ann["close"].to_numpy(dtype=float)
    ma_col = f"ma_{'mid' if rules.trail_ma >= 20 else 'fast'}"
    trail = ann[ma_col].to_numpy(dtype=float) if ma_col in ann else np.full(len(c), np.nan)
    n = len(c)

    bar = int(signal["bar"])
    entry_trigger = float(signal["entry"])
    adr = float(signal["adr20"]) / 100.0

    # ---- find the fill ------------------------------------------------------
    if signal.get("state") == "breakout":
        fill_bar, fill = bar, entry_trigger
    else:
        fill_bar, fill = None, None
        for j in range(bar + 1, min(n, bar + 1 + rules.trigger_window)):
            if h[j] >= entry_trigger:
                fill_bar, fill = j, max(entry_trigger, o[j])
                break
        if fill_bar is None:
            return {"outcome": "no_trigger", "r_multiple": float("nan"), "days_held": 0}

    stop = max(lo[fill_bar], fill * (1.0 - adr * rules.max_stop_adr_mult))
    risk = fill - stop
    if risk <= 0:
        return {"outcome": "bad_risk", "r_multiple": float("nan"), "days_held": 0}

    remaining = 1.0
    realised_r = 0.0
    took_partial = False
    mfe_r = mae_r = 0.0
    exit_reason = "max_hold"
    exit_bar = min(n - 1, fill_bar + rules.max_hold)

    for j in range(fill_bar + 1, min(n, fill_bar + 1 + rules.max_hold)):
        mfe_r = max(mfe_r, (h[j] - fill) / risk)
        mae_r = min(mae_r, (lo[j] - fill) / risk)

        # 1. stop first - a gap through it fills at the open
        if lo[j] <= stop:
            px = min(o[j], stop) if o[j] < stop else stop
            realised_r += remaining * (px - fill) / risk
            remaining, exit_reason, exit_bar = 0.0, ("stopped" if not took_partial else "trail_stop"), j
            break

        # 2. sell into the first burst of strength
        if not took_partial:
            hit_target = h[j] >= fill + rules.partial_at_r * risk
            hit_time = (j - fill_bar) >= rules.partial_days
            if hit_target or hit_time:
                px = fill + rules.partial_at_r * risk if hit_target else c[j]
                realised_r += rules.partial_fraction * (px - fill) / risk
                remaining -= rules.partial_fraction
                took_partial = True
                if rules.breakeven_after_partial:
                    stop = max(stop, fill)
                continue

        # 3. trail the rest on the moving average
        if took_partial and np.isfinite(trail[j]) and c[j] < trail[j]:
            realised_r += remaining * (c[j] - fill) / risk
            remaining, exit_reason, exit_bar = 0.0, "ma_trail", j
            break

    if remaining > 0:  # ran out of data or hit max hold
        last = min(n - 1, fill_bar + rules.max_hold)
        realised_r += remaining * (c[last] - fill) / risk
        exit_bar = last

    return {
        "outcome": exit_reason,
        "fill_date": ann.index[fill_bar],
        "fill": fill,
        "sim_stop": stop,
        "r_multiple": round(realised_r, 3),
        "mfe_r": round(mfe_r, 2),
        "mae_r": round(mae_r, 2),
        "days_held": int(exit_bar - fill_bar),
        "exit_date": ann.index[exit_bar],
        "exit_price": float(c[exit_bar]),
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate stats over a set of simulated trades."""
    df = pd.DataFrame(rows)
    traded = df[df["outcome"].notna() & (df["outcome"] != "no_trigger")] if "outcome" in df else df
    r = pd.to_numeric(traded.get("r_multiple"), errors="coerce").dropna() if len(traded) else pd.Series(dtype=float)
    if r.empty:
        return {"signals": len(df), "trades": 0}
    wins = r[r > 0]
    losses = r[r <= 0]
    return {
        "signals": len(df),
        "trades": int(len(r)),
        "no_trigger": int((df["outcome"] == "no_trigger").sum()) if "outcome" in df else 0,
        "win_rate": round(float(len(wins) / len(r)), 3),
        "avg_r": round(float(r.mean()), 3),
        "median_r": round(float(r.median()), 3),
        "total_r": round(float(r.sum()), 1),
        "avg_win_r": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_r": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "best_r": round(float(r.max()), 2),
        "worst_r": round(float(r.min()), 2),
        "expectancy_r": round(float(r.mean()), 3),
    }


def rules_from_dict(d: dict[str, Any]) -> TradeRules:
    known = set(asdict(TradeRules()))
    return TradeRules(**{k: v for k, v in d.items() if k in known})
