"""Hand-built OHLCV series so the detector can be tested without market data.

Each builder returns a DataFrame whose last bar is the one under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build(closes, ranges, volumes, start: str = "2021-01-04") -> pd.DataFrame:
    """Turn (close, daily-range-fraction, volume) triples into legal OHLC bars."""
    close = np.asarray(closes, dtype=float)
    rng = np.asarray(ranges, dtype=float)
    vol = np.asarray(volumes, dtype=float)
    assert len(close) == len(rng) == len(vol)

    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]

    top = np.maximum(open_, close)
    bot = np.minimum(open_, close)
    high = top * (1.0 + rng / 2.0)
    low = bot * (1.0 - rng / 2.0)

    idx = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def _ramp(a: float, b: float, n: int) -> np.ndarray:
    return np.linspace(a, b, n)


def textbook_breakout() -> pd.DataFrame:
    """Flat -> +65% impulse -> 26-bar tightening base on the MAs -> breakout."""
    flat_n, imp_n, base_n = 150, 15, 26

    flat_c = 20.0 + np.sin(np.arange(flat_n) / 9.0) * 0.4
    imp_c = _ramp(20.4, 33.0, imp_n)
    # Consolidation: pull back to 29, then grind back up to 32.6 on higher lows.
    base_c = np.concatenate([_ramp(32.4, 29.2, 8), _ramp(29.4, 32.6, base_n - 8)])
    brk_c = np.array([34.3])
    closes = np.concatenate([flat_c, imp_c, base_c, brk_c])

    ranges = np.concatenate([
        np.full(flat_n, 0.025),
        np.full(imp_n, 0.075),
        np.concatenate([_ramp(0.065, 0.030, base_n)]),
        np.array([0.070]),
    ])
    volumes = np.concatenate([
        np.full(flat_n, 1.0e6),
        np.full(imp_n, 3.4e6),
        _ramp(1.6e6, 0.6e6, base_n),
        np.array([4.2e6]),
    ])
    return build(closes, ranges, volumes)


def coiled_setup() -> pd.DataFrame:
    """Same base, but today has not cleared the pivot yet -> state 'setup'."""
    df = textbook_breakout()
    last = df.index[-1]
    df.loc[last, "close"] = 32.5
    df.loc[last, "open"] = 32.2
    df.loc[last, "high"] = 32.9
    df.loc[last, "low"] = 32.0
    df.loc[last, "volume"] = 0.7e6
    return df


def downtrend() -> pd.DataFrame:
    n = 200
    closes = _ramp(60.0, 22.0, n)
    return build(closes, np.full(n, 0.045), np.full(n, 1.5e6))


def deep_sloppy_base() -> pd.DataFrame:
    """Real impulse, but the 'base' gives back 45% - not a Qullamaggie base."""
    flat_n, imp_n, base_n = 150, 15, 30
    flat_c = np.full(flat_n, 20.0)
    imp_c = _ramp(20.0, 33.0, imp_n)
    base_c = np.concatenate([_ramp(32.5, 18.0, 15), _ramp(18.2, 24.0, base_n - 15)])
    closes = np.concatenate([flat_c, imp_c, base_c, [24.5]])
    ranges = np.concatenate([np.full(flat_n, 0.03), np.full(imp_n, 0.07), np.full(base_n, 0.06), [0.06]])
    volumes = np.concatenate([np.full(flat_n, 1e6), np.full(imp_n, 3e6), np.full(base_n, 2e6), [2e6]])
    return build(closes, ranges, volumes)


def below_slow_ma() -> pd.DataFrame:
    """Impulse then a long slide that puts price under the 50MA."""
    flat_n, imp_n, fade_n = 150, 15, 45
    closes = np.concatenate([
        np.full(flat_n, 20.0),
        _ramp(20.0, 34.0, imp_n),
        _ramp(33.5, 26.0, fade_n),
        [25.8],
    ])
    n = len(closes)
    return build(closes, np.full(n, 0.05), np.full(n, 1.4e6))


def illiquid_breakout() -> pd.DataFrame:
    """Perfect chart, $40k a day of volume."""
    df = textbook_breakout()
    df["volume"] = df["volume"] / 1000.0
    return df


def winner_after_signal(base: pd.DataFrame | None = None, run_pct: float = 0.55) -> pd.DataFrame:
    """Append a clean run so trade simulation has something to hold."""
    df = (base if base is not None else textbook_breakout()).copy()
    last_close = float(df["close"].iloc[-1])
    n = 30
    closes = _ramp(last_close * 1.01, last_close * (1 + run_pct), n)
    tail = build(closes, np.full(n, 0.05), np.full(n, 2.5e6))
    tail.index = pd.bdate_range(df.index[-1] + pd.offsets.BDay(1), periods=n)
    return pd.concat([df, tail])


def episodic_pivot(gap: float = 0.22, vol_mult: float = 8.0, dormant_bars: int = 200) -> pd.DataFrame:
    """Months of going nowhere, then a big gap up on enormous volume."""
    flat = 20.0 + np.sin(np.arange(dormant_bars) / 14.0) * 0.9
    closes = np.concatenate([flat, [0.0]])
    ranges = np.concatenate([np.full(dormant_bars, 0.028), [0.09]])
    volumes = np.concatenate([np.full(dormant_bars, 1.0e6), [1.0e6 * vol_mult]])

    df = build(closes[:-1], ranges[:-1], volumes[:-1])
    prev_close = float(df["close"].iloc[-1])
    open_px = prev_close * (1 + gap)
    close_px = open_px * 1.04  # closes above the open, near the highs
    idx = pd.bdate_range(df.index[-1] + pd.offsets.BDay(1), periods=1)
    day = pd.DataFrame(
        {
            "open": [open_px],
            "high": [close_px * 1.012],
            "low": [open_px * 0.985],
            "close": [close_px],
            "volume": [1.0e6 * vol_mult],
        },
        index=idx,
    )
    return pd.concat([df, day])


def ep_after_a_big_run() -> pd.DataFrame:
    """A gap that arrives when the stock has *already* tripled — a blow-off, not an EP."""
    n = 200  # must clear the EP warmup (dormancy_lookback + 25)
    run = _ramp(20.0, 62.0, n)
    df = build(run, np.full(n, 0.05), np.full(n, 2.0e6))
    prev_close = float(df["close"].iloc[-1])
    open_px = prev_close * 1.20
    close_px = open_px * 1.03
    idx = pd.bdate_range(df.index[-1] + pd.offsets.BDay(1), periods=1)
    day = pd.DataFrame(
        {"open": [open_px], "high": [close_px * 1.01], "low": [open_px * 0.98],
         "close": [close_px], "volume": [1.6e7]},
        index=idx,
    )
    return pd.concat([df, day])


def parabolic_top(run_gain: float = 1.9, crack: bool = True) -> pd.DataFrame:
    """A long quiet stretch, a vertical run, then a reversal bar off the high."""
    quiet_n, run_n = 160, 16
    quiet = 20.0 + np.sin(np.arange(quiet_n) / 11.0) * 0.5
    top = 20.0 * (1 + run_gain)
    run = _ramp(21.0, top, run_n)

    closes = np.concatenate([quiet, run])
    ranges = np.concatenate([np.full(quiet_n, 0.03), np.linspace(0.09, 0.16, run_n)])
    volumes = np.concatenate([np.full(quiet_n, 1.0e6), np.linspace(4e6, 1.1e7, run_n)])
    df = build(closes, ranges, volumes)

    last = float(df["close"].iloc[-1])
    idx = pd.bdate_range(df.index[-1] + pd.offsets.BDay(1), periods=1)
    if crack:
        # Opens higher, tags a marginal new high, then closes near the low.
        high = last * 1.10
        day = pd.DataFrame(
            {"open": [last * 1.06], "high": [high], "low": [last * 0.88],
             "close": [last * 0.90], "volume": [1.4e7]},
            index=idx,
        )
    else:
        # Still going straight up — must not signal.
        day = pd.DataFrame(
            {"open": [last * 1.02], "high": [last * 1.13], "low": [last * 1.01],
             "close": [last * 1.12], "volume": [1.3e7]},
            index=idx,
        )
    return pd.concat([df, day])


def ep_cycles(cycles: int = 6, seed: int = 0, start_price: float = 22.0, start: str = "2015-01-02") -> pd.DataFrame:
    """Long dormant stretches punctuated by genuine episodic pivots.

    `multi_cycle` deliberately cannot produce these — its names turn over every
    ~100 bars, so the six-month dormancy window always contains the previous
    run. An EP needs a stock that really has gone nowhere.
    """
    rng = np.random.default_rng(seed)
    closes: list[float] = []
    ranges: list[float] = []
    volumes: list[float] = []
    gaps: list[tuple[int, float]] = []
    price = start_price

    for _ in range(cycles):
        # Dormant: months of drifting sideways in a tight band.
        n = int(rng.integers(150, 200))
        drift = price * (1.0 + np.sin(np.arange(n) / 21.0) * 0.05 + rng.normal(0, 0.006, n).cumsum() * 0.25)
        closes.extend(drift)
        ranges.extend(np.full(n, 0.025))
        volumes.extend(rng.uniform(0.8e6, 1.3e6, n))
        price = float(drift[-1])

        # The pivot: a gap, then a multi-week run as institutions build.
        gaps.append((len(closes), float(rng.uniform(0.14, 0.32))))
        n = int(rng.integers(25, 45))
        run = _ramp(price * 1.05, price * (1 + float(rng.uniform(0.35, 1.10))), n)
        closes.extend(run)
        ranges.extend(np.full(n, 0.055))
        volumes.extend(rng.uniform(2.5e6, 5e6, n))
        price = float(run[-1])

        # Settle into a new, higher range.
        n = int(rng.integers(20, 40))
        settle = _ramp(price, price * float(rng.uniform(0.80, 0.95)), n)
        closes.extend(settle)
        ranges.extend(np.full(n, 0.035))
        volumes.extend(rng.uniform(1.0e6, 1.8e6, n))
        price = float(settle[-1])

    arr = np.asarray(closes, dtype=float)
    noise = np.zeros(len(arr))
    for i in range(1, len(arr)):
        noise[i] = noise[i - 1] * 0.85 + rng.normal(0, 0.010)
    df = build(arr * (1.0 + noise), ranges, volumes, start=start)

    for pos, size in gaps:
        if pos <= 0 or pos >= len(df):
            continue
        prev_close = float(df["close"].iloc[pos - 1])
        open_px = prev_close * (1.0 + size)
        scale = open_px / float(df["open"].iloc[pos])
        idx = df.index[pos:]
        for col in ("open", "high", "low", "close"):
            df.loc[idx, col] = df.loc[idx, col] * scale
        bar = df.index[pos]
        df.loc[bar, "open"] = open_px
        df.loc[bar, "close"] = max(float(df.loc[bar, "close"]), open_px * 1.03)
        df.loc[bar, "high"] = max(float(df.loc[bar, "close"]) * 1.01, float(df.loc[bar, "high"]))
        df.loc[bar, "low"] = min(open_px * 0.99, float(df.loc[bar, "low"]))
        df.loc[bar, "volume"] = float(df.loc[bar, "volume"]) * 9.0
    return df


def multi_cycle(
    cycles: int = 12,
    seed: int = 0,
    start_price: float = 18.0,
    win_rate: float = 0.35,
    start: str = "2015-01-02",
    gap_rate: float = 0.5,
) -> pd.DataFrame:
    """Years of bars containing repeated impulse → base → resolution cycles.

    Enough history to exercise a date cutoff and a portfolio backtest. Winners
    and losers are drawn at `win_rate` so the output is not uniformly one or the
    other; prices are pulled back toward `start_price` between cycles so a long
    run does not end up in the thousands.
    """
    rng = np.random.default_rng(seed)
    closes: list[float] = []
    ranges: list[float] = []
    volumes: list[float] = []
    gap_at: list[int] = []
    gap_size: list[float] = []
    price = start_price

    def push(seq, rng_seq, vol_seq):
        closes.extend(seq)
        ranges.extend(rng_seq)
        volumes.extend(vol_seq)

    for _ in range(cycles):
        # quiet drift
        n = int(rng.integers(30, 70))
        drift = _ramp(price, price * float(rng.uniform(0.92, 1.08)), n)
        push(drift, np.full(n, 0.025), rng.uniform(0.6e6, 1.1e6, n))
        price = float(drift[-1])

        # impulse — sometimes kicked off by an overnight gap (an episodic pivot)
        n = int(rng.integers(10, 20))
        gain = float(rng.uniform(0.45, 0.95))
        if rng.random() < gap_rate:
            gap_at.append(len(closes))
            gap_size.append(float(rng.uniform(0.12, 0.30)))
        imp = _ramp(price, price * (1 + gain), n)
        push(imp, np.full(n, 0.075), rng.uniform(2.8e6, 4.2e6, n))
        price = float(imp[-1])

        # tightening base
        n = int(rng.integers(12, 34))
        depth = float(rng.uniform(0.08, 0.16))
        low = price * (1 - depth)
        half = n // 2
        base = np.concatenate([_ramp(price * 0.99, low, half), _ramp(low * 1.01, price * 0.985, n - half)])
        push(base, _ramp(0.065, 0.030, n), _ramp(1.5e6, 0.55e6, n))
        price = float(base[-1])

        # resolution
        if rng.random() < win_rate:
            n = int(rng.integers(20, 45))
            run = _ramp(price * 1.04, price * (1 + float(rng.uniform(0.30, 0.75))), n)
            push(run, np.full(n, 0.055), rng.uniform(1.8e6, 3.2e6, n))
            price = float(run[-1])
        else:
            n = int(rng.integers(12, 30))
            fail = _ramp(price * 1.02, price * float(rng.uniform(0.72, 0.88)), n)
            push(fail, np.full(n, 0.055), rng.uniform(1.5e6, 2.6e6, n))
            price = float(fail[-1])

        # mean-revert toward the starting level so the series stays plottable
        price = float(price * 0.6 + start_price * 0.4)
        n = 12
        settle = _ramp(closes[-1], price, n)
        push(settle, np.full(n, 0.03), rng.uniform(0.7e6, 1.2e6, n))

    # Smooth ramps almost never take out a stop, which would make any backtest
    # against this fixture look absurdly good. Overlay a mean-reverting noise
    # process so bars wander the way real ones do.
    arr = np.asarray(closes, dtype=float)
    noise = np.zeros(len(arr))
    for i in range(1, len(arr)):
        noise[i] = noise[i - 1] * 0.86 + rng.normal(0, 0.017)
    arr = arr * (1.0 + noise)
    df = build(arr, ranges, volumes, start=start)

    # Re-price the chosen bars as true overnight gaps. build() derives each open
    # from the prior close, so a gap has to be stamped in afterwards; every
    # later bar is shifted with it so the series stays continuous.
    for pos, size in zip(gap_at, gap_size):
        if pos <= 0 or pos >= len(df):
            continue
        prev_close = float(df["close"].iloc[pos - 1])
        open_px = prev_close * (1.0 + size)
        scale = open_px / float(df["open"].iloc[pos])
        idx = df.index[pos:]
        for col in ("open", "high", "low", "close"):
            df.loc[idx, col] = df.loc[idx, col] * scale
        bar = df.index[pos]
        df.loc[bar, "open"] = open_px
        df.loc[bar, "close"] = max(float(df.loc[bar, "close"]), open_px * 1.02)
        df.loc[bar, "high"] = max(float(df.loc[bar, "close"]) * 1.01, float(df.loc[bar, "high"]))
        df.loc[bar, "low"] = min(open_px * 0.99, float(df.loc[bar, "low"]))
        df.loc[bar, "volume"] = float(df.loc[bar, "volume"]) * 6.0
    return df


def loser_after_signal(base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Append an immediate failure that takes out the stop."""
    df = (base if base is not None else textbook_breakout()).copy()
    last_close = float(df["close"].iloc[-1])
    n = 30
    closes = _ramp(last_close * 0.985, last_close * 0.70, n)
    tail = build(closes, np.full(n, 0.05), np.full(n, 3.0e6))
    tail.index = pd.bdate_range(df.index[-1] + pd.offsets.BDay(1), periods=n)
    return pd.concat([df, tail])
