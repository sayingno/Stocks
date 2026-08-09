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


def multi_cycle(
    cycles: int = 12,
    seed: int = 0,
    start_price: float = 18.0,
    win_rate: float = 0.35,
    start: str = "2015-01-02",
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

        # impulse
        n = int(rng.integers(10, 20))
        gain = float(rng.uniform(0.45, 0.95))
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
    return build(arr, ranges, volumes, start=start)


def loser_after_signal(base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Append an immediate failure that takes out the stop."""
    df = (base if base is not None else textbook_breakout()).copy()
    last_close = float(df["close"].iloc[-1])
    n = 30
    closes = _ramp(last_close * 0.985, last_close * 0.70, n)
    tail = build(closes, np.full(n, 0.05), np.full(n, 3.0e6))
    tail.index = pd.bdate_range(df.index[-1] + pd.offsets.BDay(1), periods=n)
    return pd.concat([df, tail])
