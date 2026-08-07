"""A local daily-bar database that updates incrementally.

The first run downloads full history for every symbol. Every run after that
fetches only the bars since the last update, which is what makes a daily
refresh of a few thousand tickers take a minute instead of an hour.

Layout:

    data/prices/<SYMBOL>.csv    full daily history, one file per symbol
    data/manifest.json          per-symbol last_date / last_fetch / failure state

Two things this handles that a naive "append the new rows" loop does not:

* **Splits and dividends.** With adjusted prices, a split rewrites the entire
  history. Each update re-fetches a short overlap window and compares it against
  what is stored; if the closes have moved, the symbol is re-downloaded in full.
* **Dead symbols.** A ticker that fails repeatedly is marked and skipped for a
  cooldown period instead of burning a request every morning.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .data import PROVIDERS, DataError, _from_csv_dir
from .indicators import normalize_ohlcv

# Bars re-fetched on every update so adjustment changes are visible.
OVERLAP_BARS = 10
# Relative close difference across the overlap that means "prices were adjusted".
ADJUST_TOLERANCE = 0.002
# Consecutive failures before a symbol is benched.
FAILURE_LIMIT = 3
FAILURE_COOLDOWN = timedelta(days=7)


@dataclass
class UpdateStats:
    checked: int = 0
    created: int = 0
    appended: int = 0
    readjusted: int = 0
    unchanged: int = 0
    skipped: int = 0
    failed: int = 0
    new_rows: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "errors"}
        d["error_sample"] = self.errors[:5]
        return d


class PriceRepository:
    """Owns the on-disk price history and knows how to bring it up to date."""

    def __init__(
        self,
        root: str | Path = "data",
        provider: str = "yfinance",
        csv_dir: str | Path | None = None,
        history_years: int = 12,
        retry_attempts: int = 3,
    ):
        self.root = Path(root)
        self.price_dir = self.root / "prices"
        self.price_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.history_years = history_years
        self.retry_attempts = max(1, retry_attempts)

        if csv_dir is not None:
            self.provider_name = "csv"
            self._fetch = _from_csv_dir(csv_dir)
        elif provider in PROVIDERS:
            self.provider_name = provider
            self._fetch = PROVIDERS[provider]
        else:
            raise DataError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)} or pass csv_dir")

        self._lock = threading.Lock()
        self._manifest = self._read_manifest()
        self._mem: dict[str, pd.DataFrame] = {}

    # -- manifest -----------------------------------------------------------
    def _read_manifest(self) -> dict[str, dict]:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def save_manifest(self) -> None:
        with self._lock:
            payload = json.dumps(self._manifest, indent=1, sort_keys=True, default=str)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(payload)
        os.replace(tmp, self.manifest_path)  # atomic; a killed run cannot corrupt it

    def entry(self, symbol: str) -> dict:
        return self._manifest.get(symbol.upper(), {})

    def _record(self, symbol: str, **fields) -> None:
        with self._lock:
            self._manifest.setdefault(symbol.upper(), {}).update(fields)

    # -- storage ------------------------------------------------------------
    def path_for(self, symbol: str) -> Path:
        safe = symbol.upper().replace("/", "_").replace("\\", "_")
        return self.price_dir / f"{safe}.csv"

    def load(self, symbol: str, use_cache: bool = True) -> pd.DataFrame | None:
        key = symbol.upper()
        if use_cache and key in self._mem:
            return self._mem[key]
        path = self.path_for(symbol)
        if not path.exists():
            return None
        try:
            df = normalize_ohlcv(pd.read_csv(path, index_col=0, parse_dates=True))
        except Exception:
            return None
        if use_cache:
            self._mem[key] = df
        return df

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        path = self.path_for(symbol)
        tmp = path.with_suffix(".csv.tmp")
        df.round(6).to_csv(tmp)
        os.replace(tmp, path)
        self._mem[symbol.upper()] = df

    def symbols(self) -> list[str]:
        return sorted(p.stem for p in self.price_dir.glob("*.csv"))

    # -- updating -----------------------------------------------------------
    def _should_skip(self, symbol: str, today: date) -> bool:
        info = self.entry(symbol)
        if info.get("failures", 0) < FAILURE_LIMIT:
            return False
        last = info.get("last_failure")
        if not last:
            return False
        try:
            benched = datetime.fromisoformat(str(last)).date()
        except ValueError:
            return False
        return today - benched < FAILURE_COOLDOWN

    def _fetch_with_retry(self, symbol: str, start: str, end: str, attempts: int | None = None) -> pd.DataFrame:
        attempts = self.retry_attempts if attempts is None else attempts
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._fetch(symbol, start, end)
            except Exception as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    # Jittered backoff so a whole thread pool does not retry in lockstep.
                    time.sleep((2**attempt) + random.uniform(0, 0.75))
        raise DataError(f"{symbol}: {last_exc}")

    def update_symbol(self, symbol: str, today: date | None = None) -> tuple[str, int]:
        """Bring one symbol current. Returns (action, rows_added)."""
        today = today or date.today()
        end = (today + timedelta(days=1)).isoformat()
        existing = self.load(symbol, use_cache=False)

        if existing is None or existing.empty:
            start = (today - timedelta(days=int(365.25 * self.history_years))).isoformat()
            fresh = self._fetch_with_retry(symbol, start, end)
            if fresh.empty:
                raise DataError(f"{symbol}: provider returned no rows")
            self.save(symbol, fresh)
            self._record(
                symbol,
                last_date=str(fresh.index[-1].date()),
                first_date=str(fresh.index[0].date()),
                last_fetch=datetime.now().isoformat(timespec="seconds"),
                rows=len(fresh),
                failures=0,
            )
            return "created", len(fresh)

        last_stored = existing.index[-1].date()
        if last_stored >= today:
            self._record(symbol, last_check=datetime.now().isoformat(timespec="seconds"))
            return "unchanged", 0

        overlap_start = existing.index[max(0, len(existing) - OVERLAP_BARS)].date()
        fresh = self._fetch_with_retry(symbol, overlap_start.isoformat(), end)
        if fresh.empty:
            self._record(symbol, last_check=datetime.now().isoformat(timespec="seconds"))
            return "unchanged", 0

        # Did the provider re-adjust history under us?
        shared = existing.index.intersection(fresh.index)
        readjusted = False
        if len(shared) >= 3:
            old_close = existing.loc[shared, "close"]
            new_close = fresh.loc[shared, "close"]
            drift = ((new_close - old_close).abs() / old_close.replace(0, pd.NA)).median()
            readjusted = bool(pd.notna(drift) and drift > ADJUST_TOLERANCE)

        if readjusted:
            start = (today - timedelta(days=int(365.25 * self.history_years))).isoformat()
            full = self._fetch_with_retry(symbol, start, end)
            if full.empty:
                raise DataError(f"{symbol}: re-adjustment refetch returned no rows")
            self.save(symbol, full)
            self._record(
                symbol,
                last_date=str(full.index[-1].date()),
                first_date=str(full.index[0].date()),
                last_fetch=datetime.now().isoformat(timespec="seconds"),
                rows=len(full),
                failures=0,
                last_readjust=str(today),
            )
            return "readjusted", max(0, len(full) - len(existing))

        merged = pd.concat([existing, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        added = len(merged) - len(existing)
        if added <= 0 and merged.index[-1] == existing.index[-1]:
            self._record(symbol, last_check=datetime.now().isoformat(timespec="seconds"))
            return "unchanged", 0

        self.save(symbol, merged)
        self._record(
            symbol,
            last_date=str(merged.index[-1].date()),
            first_date=str(merged.index[0].date()),
            last_fetch=datetime.now().isoformat(timespec="seconds"),
            rows=len(merged),
            failures=0,
        )
        return "appended", added

    def update(
        self,
        symbols: Iterable[str],
        workers: int = 8,
        pause: float = 0.0,
        force: bool = False,
        today: date | None = None,
        progress: Callable[[int, int, UpdateStats], None] | None = None,
    ) -> UpdateStats:
        """Update many symbols in parallel. Safe to interrupt and re-run."""
        symbols = [s.upper() for s in symbols]
        today = today or date.today()
        stats = UpdateStats()
        total = len(symbols)

        def work(sym: str) -> tuple[str, str, int]:
            if not force and self._should_skip(sym, today):
                return sym, "skipped", 0
            if pause:
                time.sleep(pause)
            action, added = self.update_symbol(sym, today=today)
            return sym, action, added

        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(work, s): s for s in symbols}
            for fut in as_completed(futures):
                sym = futures[fut]
                done += 1
                stats.checked += 1
                try:
                    _, action, added = fut.result()
                    setattr(stats, action, getattr(stats, action) + 1)
                    stats.new_rows += added
                except Exception as exc:
                    stats.failed += 1
                    stats.errors.append((sym, str(exc)[:200]))
                    info = self.entry(sym)
                    self._record(
                        sym,
                        failures=int(info.get("failures", 0)) + 1,
                        last_failure=datetime.now().isoformat(timespec="seconds"),
                        last_error=str(exc)[:200],
                    )
                if progress and done % 100 == 0:
                    progress(done, total, stats)

        self.save_manifest()
        return stats

    def latest_date(self) -> date | None:
        dates = [
            datetime.fromisoformat(str(v["last_date"])).date()
            for v in self._manifest.values()
            if v.get("last_date")
        ]
        return max(dates) if dates else None

    def coverage(self) -> dict:
        """Summary of what the database currently holds."""
        entries = [v for v in self._manifest.values() if v.get("rows")]
        healthy = [v for v in self._manifest.values() if int(v.get("failures", 0)) < FAILURE_LIMIT]
        return {
            "symbols_on_disk": len(self.symbols()),
            "symbols_tracked": len(self._manifest),
            "symbols_healthy": len(healthy),
            "total_rows": sum(int(v.get("rows", 0)) for v in entries),
            "latest_date": str(self.latest_date()) if self.latest_date() else None,
            "provider": self.provider_name,
        }
