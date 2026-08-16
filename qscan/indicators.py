"""Price/volume indicators used by the breakout detector.

Everything takes a DataFrame indexed by date with columns
open/high/low/close/volume and returns a Series aligned to that index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV = ("open", "high", "low", "close", "volume")


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case the columns, sort by date, drop unusable rows."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out = out.rename(columns={"adj_close": "adj close", "adjclose": "adj close"})

    if "adj close" in out.columns:
        if "close" not in out.columns:
            out = out.rename(columns={"adj close": "close"})
        else:
            # A vendor file carrying both is the classic silent corruption: the
            # raw close still contains every split, so a 2:1 shows up as a 50%
            # cliff the detector reads as a real move. Adjust — and scale the
            # whole bar by the same ratio, because adjusting the close alone
            # would leave high < close and wreck every range and ADR.
            close = pd.to_numeric(out["close"], errors="coerce")
            adj = pd.to_numeric(out["adj close"], errors="coerce")
            ratio = (adj / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            if ratio.notna().any():
                ratio = ratio.fillna(1.0)
                for col in ("open", "high", "low", "close"):
                    if col in out.columns:
                        out[col] = pd.to_numeric(out[col], errors="coerce") * ratio
            out = out.drop(columns=["adj close"])

    missing = [c for c in OHLCV if c not in out.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing} (have {list(out.columns)})")
    out = out[list(OHLCV)]
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.astype(float)
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]

    # Repair impossible bars. Vendor files — especially ones where only some
    # columns were adjusted — can arrive with high < close or low > open. Left
    # alone, that produces a negative daily range, which poisons ADR and every
    # gate built on it. Widening the extremes is the conservative fix: it never
    # invents a move that wasn't there.
    body_high = out[["open", "close"]].max(axis=1)
    body_low = out[["open", "close"]].min(axis=1)
    out["high"] = out[["high"]].join(body_high.rename("b")).max(axis=1)
    out["low"] = out[["low"]].join(body_low.rename("b")).min(axis=1)
    return out


def moving_average(close: pd.Series, length: int, use_ema: bool = False) -> pd.Series:
    if use_ema:
        return close.ewm(span=length, adjust=False, min_periods=length).mean()
    return close.rolling(length, min_periods=length).mean()


def adr_pct(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Average Daily Range as a percent: mean of (high/low - 1) over `length`.

    Qullamaggie's ADR. Using high/low rather than (high-low)/close keeps it
    scale-free and matches the ratio form most charting packages use.
    """
    daily = df["high"] / df["low"] - 1.0
    return daily.rolling(length, min_periods=max(2, length // 2)).mean() * 100.0


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def dollar_volume(df: pd.DataFrame, length: int = 20) -> pd.Series:
    return (df["close"] * df["volume"]).rolling(length, min_periods=max(2, length // 2)).mean()


def pct_change_n(close: pd.Series, n: int) -> pd.Series:
    """Simple return over n bars — the raw material for relative strength."""
    return close / close.shift(n) - 1.0


def slope_pct_per_bar(series: pd.Series, length: int) -> pd.Series:
    """Least-squares slope over a trailing window, expressed as %/bar.

    Normalised by the window mean so it is comparable across price levels.
    """
    x = np.arange(length, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered**2).sum())

    def _fit(window: np.ndarray) -> float:
        mean = window.mean()
        if not np.isfinite(mean) or mean == 0.0:
            return np.nan
        return float((x_centered * window).sum() / denom) / mean * 100.0

    return series.rolling(length, min_periods=length).apply(_fit, raw=True)


def rolling_rank_pct(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank (0-100) of each column, per row."""
    return frame.rank(axis=1, pct=True) * 100.0


def annotate(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Attach every derived column the detector needs."""
    out = normalize_ohlcv(df)
    close = out["close"]

    out["ma_fast"] = moving_average(close, cfg.ma_fast, cfg.use_ema)
    out["ma_mid"] = moving_average(close, cfg.ma_mid, cfg.use_ema)
    out["ma_slow"] = moving_average(close, cfg.ma_slow, cfg.use_ema)

    out["adr20"] = adr_pct(out, 20)
    out["atr14"] = atr(out, 14)
    out["dollar_vol20"] = dollar_volume(out, 20)
    out["vol_avg20"] = out["volume"].rolling(20, min_periods=10).mean()
    out["range_pct"] = out["high"] / out["low"] - 1.0

    # Short horizons matter for the episodic pivot: a stock already up 8% into
    # the print has partly priced the news, which is a different animal from one
    # that was dead going in.
    for n, label in ((3, "3d"), (5, "1w"), (21, "1m"), (63, "3m"), (126, "6m")):
        out[f"ret_{label}"] = pct_change_n(close, n)

    out["ma_mid_slope"] = slope_pct_per_bar(out["ma_mid"], cfg.ma_mid)
    out["ma_slow_slope"] = slope_pct_per_bar(out["ma_slow"], cfg.ma_slow)
    return out
