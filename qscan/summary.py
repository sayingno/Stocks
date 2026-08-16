"""Inventory the price database: what is in it, and is any of it usable?

The first thing to do with a freshly ingested vendor dump is not to scan it —
it is to find out what you actually got. A dump can look fine and still be
missing a year, be padded with delisted shells, or carry a thousand symbols too
illiquid to trade. Every one of those quietly changes what a backtest means.

Produces two things: a per-symbol table you can sort and filter, and an overall
picture with the distributions that decide which thresholds are even sensible
for this market. A $5 price floor is reasonable in USD and nonsense in KRW.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# A bar-to-bar move beyond this, with no split adjustment to explain it, is
# more likely an unadjusted corporate action than a real day.
#
# The threshold has to sit below 0.50, because a 2:1 split — the most common
# ratio there is — lands exactly on a 50% drop. 0.45 also catches 3:1, 4:1 and
# 10:1; it misses 3:2 (a 33% step), which is the price of not firing on every
# biotech that halves on a failed trial. This is a warning to go and look, not
# a rejection, so erring toward a few false positives is the right side.
SUSPECT_JUMP = 0.45


@dataclass
class DatasetSummary:
    per_symbol: pd.DataFrame = field(default_factory=pd.DataFrame)
    overview: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def describe_symbol(symbol: str, df: pd.DataFrame) -> dict[str, Any]:
    """One row per ticker: span, coverage, liquidity, and anything suspicious."""
    if df is None or df.empty:
        return {"symbol": symbol, "bars": 0}

    close = df["close"]
    volume = df["volume"]
    first, last = df.index[0], df.index[-1]

    # Sessions actually present vs weekdays in the span. A market with many
    # holidays will sit well below 1.0 legitimately, so this is a relative
    # measure across the dataset, not an absolute quality bar.
    weekdays = int(np.busday_count(first.date(), last.date())) + 1
    coverage = len(df) / weekdays if weekdays > 0 else np.nan

    step = (close / close.shift(1)).dropna()
    jumps = int(((step - 1).abs() > SUSPECT_JUMP).sum())
    biggest_gap = int(df.index.to_series().diff().dt.days.max() or 0)

    dollar_vol = (close * volume).tail(60)
    rng = (df["high"] / df["low"] - 1.0).tail(60)

    return {
        "symbol": symbol,
        "bars": len(df),
        "first": first.date(),
        "last": last.date(),
        "years": round((last - first).days / 365.25, 2),
        "coverage": round(float(coverage), 3) if np.isfinite(coverage) else None,
        "max_calendar_gap_days": biggest_gap,
        "last_close": round(float(close.iloc[-1]), 4),
        "median_close": round(float(close.median()), 4),
        "median_dollar_vol": round(float(dollar_vol.median()), 0) if len(dollar_vol) else None,
        "median_adr_pct": round(float(rng.median() * 100), 2) if len(rng) else None,
        "zero_volume_days": int((volume <= 0).sum()),
        "suspect_jumps": jumps,
    }


def _pct(series: pd.Series, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    return {f"p{int(q * 100)}": round(float(s.quantile(q)), 4) for q in qs}


def summarise(frames: dict[str, pd.DataFrame], min_bars: int = 200) -> DatasetSummary:
    """Build the inventory across every symbol in the database."""
    rows = [describe_symbol(sym, df) for sym, df in sorted(frames.items())]
    per = pd.DataFrame(rows)
    if per.empty:
        return DatasetSummary(overview={"symbols": 0})

    usable = per[per["bars"] >= min_bars]
    warnings: list[str] = []

    latest = pd.to_datetime(per["last"]).max()
    stale = per[pd.to_datetime(per["last"]) < latest - pd.Timedelta(days=30)]
    if len(stale):
        warnings.append(
            f"{len(stale)} symbols stop more than 30 days before the dataset's last date "
            f"({latest.date()}) — delisted names, or a partial dump"
        )
    thin = per[per["bars"] < min_bars]
    if len(thin):
        warnings.append(f"{len(thin)} symbols have under {min_bars} bars and cannot be scanned")
    jumpy = per[per["suspect_jumps"] > 0]
    if len(jumpy):
        warnings.append(
            f"{len(jumpy)} symbols contain a >{SUSPECT_JUMP:.0%} single-bar move — "
            f"check whether the file is split-adjusted"
        )
    if per["coverage"].notna().any() and float(per["coverage"].median()) < 0.5:
        warnings.append(
            "median session coverage is below 50% of weekdays — the dump may be "
            "missing days, or this market trades far fewer sessions than a 5-day week"
        )

    overview = {
        "symbols": len(per),
        "usable_symbols": len(usable),
        "total_bars": int(per["bars"].sum()),
        "date_range": [str(pd.to_datetime(per["first"]).min().date()),
                       str(pd.to_datetime(per["last"]).max().date())],
        "median_history_years": round(float(per["years"].median()), 2),
        "median_coverage": round(float(per["coverage"].median()), 3) if per["coverage"].notna().any() else None,
        "last_close": _pct(per["last_close"]),
        "median_dollar_vol": _pct(per["median_dollar_vol"]),
        "median_adr_pct": _pct(per["median_adr_pct"]),
        "symbols_with_zero_volume_days": int((per["zero_volume_days"] > 0).sum()),
        "symbols_with_suspect_jumps": int((per["suspect_jumps"] > 0).sum()),
    }
    return DatasetSummary(per_symbol=per, overview=overview, warnings=warnings)


def suggest_thresholds(summary: DatasetSummary) -> dict[str, Any]:
    """Threshold starting points derived from this dataset, not from US defaults.

    The shipped defaults assume USD and US market structure. On a CNY or KRW
    dataset a $5 floor and a $3M dollar-volume floor are meaningless, and a 10%
    daily price limit caps ADR so the 3.5% ADR floor may exclude the entire
    market. These are the dataset's own quantiles — a place to start arguing
    from, not an answer.
    """
    o = summary.overview
    if not o.get("symbols"):
        return {}
    price = o.get("last_close", {})
    dv = o.get("median_dollar_vol", {})
    adr = o.get("median_adr_pct", {})
    return {
        "min_price": price.get("p25"),
        "min_dollar_volume": dv.get("p50"),
        "min_adr_pct": adr.get("p50"),
        "note": (
            "the 25th percentile price and median dollar volume / ADR of this "
            "dataset — sanity-check against the market's currency and price limits"
        ),
    }
