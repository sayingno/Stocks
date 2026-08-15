"""Event study: what did the big movers look like *before* they moved?

This answers a different question from the scanner. The scanner asks "is this a
setup today?". This asks "across all the times a catalyst landed, what separated
the ones that ran from the ones that didn't?"

The trap it is built to avoid
-----------------------------
Selecting on the outcome and then describing the winners tells you what winners
looked like. It does **not** tell you the signature predicts anything. If the big
post-earnings movers were flat for three months beforehand, that only matters if
the non-movers were *not* also flat for three months — and most of the market is
flat for three months.

So every comparison here carries the base-rate cohort alongside, and the column
worth reading is the lift between them. A `t` statistic accompanies each row:
treat |t| < 2 as noise regardless of how large the gap looks.

Two views
---------
`compare_cohorts`  movers vs non-movers on each antecedent — the headline
`bucket_by`        slice one antecedent into buckets, show P(mover) per bucket
                   against the overall base rate — the decision-useful view
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# Antecedents recorded on every event. All are known strictly before the bar.
DEFAULT_ANTECEDENTS: tuple[str, ...] = (
    "ret_3d", "ret_1w", "ret_1m", "ret_3m", "ret_6m",
    "adr20", "dollar_vol20", "dist_from_52w_high", "vol_ratio_20",
)


@dataclass(frozen=True)
class StudyConfig:
    # ---- what counts as an event -------------------------------------------
    event: str = "gap"  # "gap" | "earnings" | "all"
    min_gap: float = 0.05  # for event="gap": open vs prior close
    earnings_window: int = 1  # for event="earnings": bars after the date to look at

    # ---- what makes an event a "mover" -------------------------------------
    outcome_horizon: int = 10  # bars forward
    outcome_threshold: float = 0.20  # forward return that defines a mover

    # ---- housekeeping -------------------------------------------------------
    antecedents: tuple[str, ...] = DEFAULT_ANTECEDENTS
    min_history: int = 130  # bars needed before an event is usable
    min_price: float = 1.0
    min_dollar_volume: float = 0.0
    cooldown: int = 5  # bars before the same symbol can raise another event

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class StudyResult:
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    cohorts: pd.DataFrame = field(default_factory=pd.DataFrame)
    buckets: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# event collection
# --------------------------------------------------------------------------
def _extra_features(ann: pd.DataFrame) -> pd.DataFrame:
    """Antecedents that aren't already columns on the annotated frame."""
    out = pd.DataFrame(index=ann.index)
    high_252 = ann["high"].rolling(252, min_periods=60).max()
    out["dist_from_52w_high"] = ann["close"] / high_252 - 1.0
    avg = ann["volume"].rolling(20, min_periods=10).mean()
    out["vol_ratio_20"] = ann["volume"] / avg.replace(0, np.nan)
    return out


def collect_events(
    ann: pd.DataFrame,
    symbol: str,
    cfg: StudyConfig,
    earnings_dates: Iterable[pd.Timestamp] | None = None,
    start: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    """Find every candidate event in one symbol and record its antecedents.

    Antecedents are read at the bar *before* the event, so nothing describing
    the event itself leaks into the "what did it look like going in" picture.
    """
    n = len(ann)
    if n < cfg.min_history + cfg.outcome_horizon + 2:
        return []

    extra = _extra_features(ann)
    o = ann["open"].to_numpy(float)
    c = ann["close"].to_numpy(float)
    h = ann["high"].to_numpy(float)
    lo = ann["low"].to_numpy(float)
    dv = ann["dollar_vol20"].to_numpy(float)

    if cfg.event == "earnings":
        wanted = set(pd.DatetimeIndex(earnings_dates or []).normalize())
        candidate_bars = [
            i for i in range(cfg.min_history, n)
            # The reaction bar is the first session on/after the disclosure.
            if any(pd.Timestamp(ann.index[j]).normalize() in wanted
                   for j in range(max(0, i - cfg.earnings_window), i + 1))
        ]
    elif cfg.event == "all":
        candidate_bars = list(range(cfg.min_history, n))
    else:  # "gap"
        candidate_bars = [
            i for i in range(cfg.min_history, n)
            if c[i - 1] > 0 and (o[i] / c[i - 1] - 1.0) >= cfg.min_gap
        ]

    events: list[dict[str, Any]] = []
    last_bar = -10**9
    for i in candidate_bars:
        if i - last_bar <= cfg.cooldown:
            continue
        if i + cfg.outcome_horizon >= n:
            continue
        if c[i] < cfg.min_price or not (dv[i] >= cfg.min_dollar_volume):
            continue
        date = pd.Timestamp(ann.index[i])
        if start is not None and date < start:
            continue

        prev = i - 1  # antecedents as of the close before the event
        row: dict[str, Any] = {
            "symbol": symbol,
            "date": date,
            "bar": i,
            "gap": o[i] / c[i - 1] - 1.0 if c[i - 1] > 0 else np.nan,
            "event_day_return": c[i] / c[i - 1] - 1.0 if c[i - 1] > 0 else np.nan,
        }
        for name in cfg.antecedents:
            src = ann if name in ann.columns else extra
            row[name] = float(src[name].iloc[prev]) if name in src.columns else np.nan

        # Outcome measured from the event close forward.
        j = i + cfg.outcome_horizon
        row["fwd_return"] = c[j] / c[i] - 1.0
        row["fwd_mfe"] = float(np.max(h[i + 1 : j + 1])) / c[i] - 1.0
        row["fwd_mae"] = float(np.min(lo[i + 1 : j + 1])) / c[i] - 1.0
        row["is_mover"] = bool(row["fwd_return"] >= cfg.outcome_threshold)
        events.append(row)
        last_bar = i

    return events


# --------------------------------------------------------------------------
# cohort comparison
# --------------------------------------------------------------------------
def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch's t for two samples of unequal variance. |t| < 2 ~ noise."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denom = math.sqrt(va / len(a) + vb / len(b))
    if denom == 0 or not math.isfinite(denom):
        return float("nan")
    return float((a.mean() - b.mean()) / denom)


def compare_cohorts(events: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    """Movers vs non-movers on each antecedent, with lift and a noise guard."""
    if events.empty or "is_mover" not in events:
        return pd.DataFrame()

    movers = events[events["is_mover"]]
    rest = events[~events["is_mover"]]
    rows = []
    for name in cfg.antecedents:
        if name not in events.columns:
            continue
        a = pd.to_numeric(movers[name], errors="coerce").to_numpy(float)
        b = pd.to_numeric(rest[name], errors="coerce").to_numpy(float)
        if not np.isfinite(a).any() or not np.isfinite(b).any():
            continue
        am, bm = np.nanmean(a), np.nanmean(b)
        t = _welch_t(a, b)
        rows.append(
            {
                "antecedent": name,
                "movers_mean": round(float(am), 5),
                "others_mean": round(float(bm), 5),
                "lift": round(float(am - bm), 5),
                "movers_median": round(float(np.nanmedian(a)), 5),
                "others_median": round(float(np.nanmedian(b)), 5),
                "t": round(t, 2) if math.isfinite(t) else None,
                "signal": "yes" if math.isfinite(t) and abs(t) >= 2 else "noise",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.reindex(out["t"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return out


def bucket_by(
    events: pd.DataFrame,
    field_name: str,
    buckets: int | Sequence[float] = 5,
) -> pd.DataFrame:
    """P(mover) per bucket of one antecedent, against the overall base rate.

    This is the view you act on: it says "when the stock was down 10-20% over
    three months, X% of its gaps ran" — and puts the unconditional rate next to
    it so you can see whether X is actually different.
    """
    if events.empty or field_name not in events.columns:
        return pd.DataFrame()

    values = pd.to_numeric(events[field_name], errors="coerce")
    usable = events[values.notna()].copy()
    if usable.empty:
        return pd.DataFrame()
    values = values[values.notna()]

    try:
        if isinstance(buckets, int):
            usable["_b"] = pd.qcut(values, buckets, duplicates="drop")
        else:
            usable["_b"] = pd.cut(values, list(buckets))
    except ValueError:
        return pd.DataFrame()

    base = float(usable["is_mover"].mean())
    grouped = usable.groupby("_b", observed=True)
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "mover_rate": grouped["is_mover"].mean().round(4),
            "avg_fwd": grouped["fwd_return"].mean().round(4),
            "median_fwd": grouped["fwd_return"].median().round(4),
            "avg_mfe": grouped["fwd_mfe"].mean().round(4),
            "avg_mae": grouped["fwd_mae"].mean().round(4),
        }
    ).reset_index()
    out = out.rename(columns={"_b": field_name})
    out[field_name] = out[field_name].astype(str)
    out["base_rate"] = round(base, 4)
    out["lift"] = (out["mover_rate"] - base).round(4)
    return out


def run(
    frames: dict[str, pd.DataFrame],
    cfg: StudyConfig,
    earnings: dict[str, list[pd.Timestamp]] | None = None,
    start: pd.Timestamp | None = None,
    bucket_fields: Sequence[str] | None = None,
    buckets: int = 5,
) -> StudyResult:
    """Collect events across the universe and compare the cohorts."""
    all_events: list[dict[str, Any]] = []
    for symbol, ann in frames.items():
        dates = (earnings or {}).get(symbol)
        if cfg.event == "earnings" and not dates:
            continue
        all_events.extend(collect_events(ann, symbol, cfg, dates, start))

    events = pd.DataFrame(all_events)
    if events.empty:
        return StudyResult(summary={"events": 0})

    events = events.sort_values(["date", "symbol"]).reset_index(drop=True)
    cohorts = compare_cohorts(events, cfg)

    fields = list(bucket_fields or ("ret_3d", "ret_1w", "ret_1m", "ret_3m"))
    bucket_tables = {f: bucket_by(events, f, buckets) for f in fields if f in events.columns}
    bucket_tables = {k: v for k, v in bucket_tables.items() if not v.empty}

    movers = int(events["is_mover"].sum())
    summary = {
        "events": len(events),
        "symbols": int(events["symbol"].nunique()),
        "movers": movers,
        "base_rate": round(movers / len(events), 4),
        "outcome": f"fwd {cfg.outcome_horizon}d >= {cfg.outcome_threshold:.0%}",
        "event_definition": (
            f"gap >= {cfg.min_gap:.0%}" if cfg.event == "gap"
            else "earnings date" if cfg.event == "earnings" else "every bar"
        ),
        "avg_fwd_movers": round(float(events.loc[events["is_mover"], "fwd_return"].mean()), 4),
        "avg_fwd_others": round(float(events.loc[~events["is_mover"], "fwd_return"].mean()), 4),
        "date_range": [str(events["date"].min().date()), str(events["date"].max().date())],
    }
    return StudyResult(events=events, cohorts=cohorts, buckets=bucket_tables, summary=summary)
