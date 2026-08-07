"""Pluggable daily-OHLCV sources with an on-disk cache.

Providers all return a DataFrame indexed by date with open/high/low/close/volume.

  csv       a directory of TICKER.csv files - always works, no network
  yfinance  free, batteries-included, needs `pip install yfinance`
  stooq     free CSV endpoint, no API key, no library
  tiingo    needs TIINGO_API_KEY, best free-tier history quality

The cache is plain CSV under `--cache-dir` so you can inspect or hand-edit it.
Re-running a scan the same day costs no network calls.
"""

from __future__ import annotations

import io
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .indicators import normalize_ohlcv

DEFAULT_CACHE = Path("data/cache")
_STALE_AFTER = timedelta(hours=12)


class DataError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------
def _from_csv_dir(directory: str | Path) -> Callable[[str, str, str], pd.DataFrame]:
    root = Path(directory)

    def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
        for name in (f"{symbol}.csv", f"{symbol.upper()}.csv", f"{symbol.lower()}.csv"):
            path = root / name
            if path.exists():
                df = pd.read_csv(path)
                date_col = next(
                    (c for c in df.columns if str(c).strip().lower() in ("date", "datetime", "timestamp")),
                    df.columns[0],
                )
                df = df.set_index(pd.to_datetime(df[date_col], utc=False, errors="coerce"))
                return normalize_ohlcv(df.drop(columns=[date_col], errors="ignore"))
        raise DataError(f"{symbol}: no CSV in {root}")

    return fetch


def _yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataError("yfinance not installed (pip install yfinance)") from exc

    raw = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise DataError(f"{symbol}: yfinance returned no rows")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return normalize_ohlcv(raw)


def _stooq(symbol: str, start: str, end: str) -> pd.DataFrame:
    import urllib.request

    sym = symbol.lower().replace(".", "-")
    if "." not in sym and not sym.endswith(".us"):
        sym = f"{sym}.us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = resp.read().decode("utf-8", "replace")
    except Exception as exc:
        raise DataError(f"{symbol}: stooq request failed ({exc})") from exc
    if not payload.strip() or payload.lstrip().lower().startswith("<"):
        raise DataError(f"{symbol}: stooq returned no data")
    df = pd.read_csv(io.StringIO(payload))
    if "Date" not in df.columns:
        raise DataError(f"{symbol}: unexpected stooq payload")
    df = df.set_index(pd.to_datetime(df["Date"])).drop(columns=["Date"])
    return normalize_ohlcv(df).loc[start:end]


def _tiingo(symbol: str, start: str, end: str) -> pd.DataFrame:
    import json
    import urllib.request

    token = os.environ.get("TIINGO_API_KEY")
    if not token:
        raise DataError("TIINGO_API_KEY not set")
    url = (
        f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        f"?startDate={start}&endDate={end}&format=json&token={token}"
    )
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            rows = json.loads(resp.read().decode())
    except Exception as exc:
        raise DataError(f"{symbol}: tiingo request failed ({exc})") from exc
    if not rows:
        raise DataError(f"{symbol}: tiingo returned no rows")
    df = pd.DataFrame(rows)
    df = df.set_index(pd.to_datetime(df["date"]).dt.tz_localize(None))
    df = df.rename(
        columns={
            "adjOpen": "open",
            "adjHigh": "high",
            "adjLow": "low",
            "adjClose": "close",
            "adjVolume": "volume",
        }
    )
    return normalize_ohlcv(df)


PROVIDERS: dict[str, Callable[[str, str, str], pd.DataFrame]] = {
    "yfinance": _yfinance,
    "stooq": _stooq,
    "tiingo": _tiingo,
}


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
class PriceStore:
    """Fetches bars through a provider, memoised on disk and in memory."""

    def __init__(
        self,
        provider: str = "yfinance",
        cache_dir: str | Path = DEFAULT_CACHE,
        csv_dir: str | Path | None = None,
        offline: bool = False,
    ):
        if csv_dir is not None:
            self.name = "csv"
            self._fetch = _from_csv_dir(csv_dir)
        elif provider not in PROVIDERS:
            raise DataError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)} or --csv-dir")
        else:
            self.name = provider
            self._fetch = PROVIDERS[provider]
        self.cache_dir = Path(cache_dir) / self.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self._mem: dict[str, pd.DataFrame] = {}

    def _cache_path(self, symbol: str) -> Path:
        safe = symbol.upper().replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe}.csv"

    def _read_cache(self, symbol: str) -> pd.DataFrame | None:
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return normalize_ohlcv(df)
        except Exception:
            return None

    def _cache_fresh(self, symbol: str) -> bool:
        path = self._cache_path(symbol)
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < _STALE_AFTER

    def get(self, symbol: str, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
        if symbol in self._mem and not refresh:
            return self._mem[symbol]

        cached = self._read_cache(symbol)
        use_cache = cached is not None and not refresh and (self.offline or self._cache_fresh(symbol))
        if use_cache:
            df = cached
        else:
            try:
                df = self._fetch(symbol, start, end)
                if self.name != "csv":
                    df.to_csv(self._cache_path(symbol))
            except DataError:
                if cached is None:
                    raise
                df = cached  # network hiccup: fall back to whatever we have

        df = df.loc[str(start) : str(end)]
        self._mem[symbol] = df
        return df

    def get_many(
        self,
        symbols: Iterable[str],
        start: str,
        end: str,
        refresh: bool = False,
        on_error: Callable[[str, Exception], None] | None = None,
        pause: float = 0.0,
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = self.get(sym, start, end, refresh=refresh)
                if len(df) > 0:
                    out[sym] = df
            except Exception as exc:  # keep the scan alive on a bad ticker
                if on_error:
                    on_error(sym, exc)
            if pause:
                time.sleep(pause)
        return out


def default_window(years: int = 3) -> tuple[str, str]:
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=int(365.25 * years) + 200)
    return start.isoformat(), end.isoformat()
