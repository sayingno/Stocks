"""Earnings calendar: load it, and answer "how far is this bar from a report?".

No calendar can be fetched from here — every exchange and vendor host is
blocked — so this reads one you supply. `tools/fetch_earnings.py` produces the
file on a machine that has network.

Expected shape, extra columns ignored:

    symbol,date
    600519,2024-04-27
    600519,2024-08-09

When no calendar exists, `infer_from_prices` marks bars that look like a
reaction to a scheduled disclosure — a large gap on heavy volume, spaced at
least a quarter apart. It is a **proxy**, and every field it produces is
labelled `inferred` so it can never be mistaken for the real thing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL_ALIASES = ("symbol", "ticker", "code", "sym", "secid", "stock_code")
DATE_ALIASES = (
    "date", "earnings_date", "report_date", "reportdate", "ann_date",
    "announcement_date", "disclosure_date", "period_end", "报告日期", "公告日期", "披露日期",
)


def load(path: str | Path) -> dict[str, list[pd.Timestamp]]:
    """Read a calendar CSV into {symbol: [dates]}, sorted and de-duplicated."""
    # dtype=str throughout: pandas would read 000001 as the integer 1,
    # silently unmatching every zero-padded ticker from its price file.
    df = pd.read_csv(path, dtype=str)
    lower = {str(c).strip().lower(): c for c in df.columns}

    sym_col = next((lower[a] for a in SYMBOL_ALIASES if a in lower), None)
    date_col = next((lower[a] for a in DATE_ALIASES if a in lower), None)
    if sym_col is None or date_col is None:
        raise ValueError(
            f"calendar needs a symbol and a date column; found {list(df.columns)}"
        )

    out: dict[str, list[pd.Timestamp]] = {}
    dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    for sym, when in zip(df[sym_col], dates):
        if pd.isna(when):
            continue
        key = str(sym).strip().upper()
        if not key:
            continue
        out.setdefault(key, []).append(pd.Timestamp(when).normalize())

    return {k: sorted(set(v)) for k, v in out.items()}


def infer_from_prices(
    ann: pd.DataFrame,
    min_gap: float = 0.06,
    min_vol_mult: float = 2.5,
    min_spacing: int = 40,
) -> list[pd.Timestamp]:
    """Guess disclosure reaction days from price action alone.

    A proxy for when no calendar is available. Real calendars beat this every
    time: a scheduled report that lands with a 1% move is invisible here, and a
    takeover rumour looks identical to a blowout quarter.
    """
    if len(ann) < 30:
        return []
    o = ann["open"].to_numpy(float)
    c = ann["close"].to_numpy(float)
    v = ann["volume"].to_numpy(float)
    avg = pd.Series(v).rolling(20, min_periods=10).mean().to_numpy(float)

    picks: list[pd.Timestamp] = []
    last = -10**9
    for i in range(20, len(ann)):
        if i - last < min_spacing or c[i - 1] <= 0 or not np.isfinite(avg[i]) or avg[i] <= 0:
            continue
        if abs(o[i] / c[i - 1] - 1.0) >= min_gap and v[i] / avg[i] >= min_vol_mult:
            picks.append(pd.Timestamp(ann.index[i]).normalize())
            last = i
    return picks


def distances(index: pd.DatetimeIndex, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Trading-bar distance from each bar to the nearest report either side.

    Returns `days_since_earnings` (>= 0, bars since the last report on or before
    this bar) and `days_to_next_earnings` (> 0, bars until the next one). NaN
    where there is no report on that side — which matters: "no next report
    known" is not the same as "the next report is far away", and a filter that
    conflates them will quietly drop the whole tail of your sample.
    """
    out = pd.DataFrame(index=index, columns=["days_since_earnings", "days_to_next_earnings"], dtype=float)
    if not dates:
        return out

    marks = np.array([index.searchsorted(d) for d in sorted(dates)], dtype=int)
    marks = marks[(marks >= 0) & (marks < len(index))]
    if marks.size == 0:
        return out

    positions = np.arange(len(index))
    prev_idx = np.searchsorted(marks, positions, side="right") - 1
    next_idx = np.searchsorted(marks, positions, side="left")

    since = np.where(prev_idx >= 0, positions - marks[np.clip(prev_idx, 0, len(marks) - 1)], np.nan)
    ahead = np.where(
        next_idx < len(marks), marks[np.clip(next_idx, 0, len(marks) - 1)] - positions, np.nan
    )
    out["days_since_earnings"] = since
    out["days_to_next_earnings"] = ahead
    return out


def attach(ann: pd.DataFrame, dates: list[pd.Timestamp] | None) -> pd.DataFrame:
    """Add the two distance columns to an annotated frame."""
    out = ann.copy()
    d = distances(out.index, dates or [])
    out["days_since_earnings"] = d["days_since_earnings"]
    out["days_to_next_earnings"] = d["days_to_next_earnings"]
    return out
