"""Load bulk vendor data — zips, folders, or one giant CSV — into the price database.

Vendor dumps arrive in two shapes and this handles both without being told which:

  per-ticker    a zip or folder of AAPL.csv, MSFT.csv, …
  long format   one big file with a symbol column and every ticker stacked

Column naming is guessed from a table of the spellings vendors actually use
(`Adj Close`, `adj_close`, `<TICKER>`, `Trade Date`, `vol`, …). Anything that
cannot be mapped is reported rather than silently dropped, because a column
quietly missing is how a backtest ends up measuring nothing.

Everything routes through `indicators.normalize_ohlcv`, so split adjustment,
de-duplication and ordering happen in exactly one place.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd

from .indicators import normalize_ohlcv

# Vendor spellings -> our canonical name. Compared lower-cased and stripped.
ALIASES: dict[str, str] = {
    "date": "date", "datetime": "date", "timestamp": "date", "time": "date",
    "trade date": "date", "tradedate": "date", "day": "date", "dt": "date",
    "open": "open", "o": "open", "opening price": "open", "open price": "open",
    "high": "high", "h": "high", "high price": "high",
    "low": "low", "l": "low", "low price": "low",
    "close": "close", "c": "close", "closing price": "close", "close price": "close", "last": "close",
    "adj close": "adj close", "adj_close": "adj close", "adjclose": "adj close",
    "adjusted close": "adj close", "adj. close": "adj close",
    "volume": "volume", "v": "volume", "vol": "volume", "total volume": "volume",
    "symbol": "symbol", "ticker": "symbol", "sym": "symbol", "name": "symbol",
    "<ticker>": "symbol", "<date>": "date", "<open>": "open", "<high>": "high",
    "<low>": "low", "<close>": "close", "<vol>": "volume",
}

REQUIRED = ("date", "open", "high", "low", "close", "volume")
TABULAR_SUFFIXES = {".csv", ".txt", ".tsv"}


@dataclass
class IngestReport:
    symbols: int = 0
    rows: int = 0
    files_read: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "symbols": self.symbols,
            "rows": self.rows,
            "files_read": self.files_read,
            "skipped": len(self.skipped),
            "skipped_sample": self.skipped[:8],
        }


def canonical_columns(columns) -> dict[str, str]:
    """Map a frame's columns onto our canonical names."""
    mapping: dict[str, str] = {}
    for col in columns:
        key = str(col).strip().lower().replace("  ", " ")
        if key in ALIASES:
            mapping[col] = ALIASES[key]
    return mapping


def _read_tabular(handle, name: str) -> pd.DataFrame | None:
    """Read a table, keeping any symbol column as text.

    Pandas infers `000001` as the integer 1, which silently destroys every
    zero-padded ticker — the Shenzhen half of the A-share market, most of Hong
    Kong, and any vendor that pads US symbols. The symbol column is read as a
    string; everything else is left to normal inference so a 600 MB file of
    prices does not become 600 MB of Python strings.
    """
    sep = "\t" if name.lower().endswith(".tsv") else None
    try:
        header = pd.read_csv(handle, sep=sep, engine="python", nrows=0)
    except Exception:
        return None

    dtypes = {c: str for c in header.columns
              if str(c).strip().lower() in ALIASES and ALIASES[str(c).strip().lower()] == "symbol"}
    if hasattr(handle, "seek"):
        handle.seek(0)  # a BytesIO from a zip must be rewound; a Path need not be
    try:
        return pd.read_csv(handle, sep=sep, engine="python", dtype=dtypes or None)
    except Exception:
        return None


def _iter_members(source: Path) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield (member name, raw frame) from a zip, a directory, or a single file."""
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.suffix.lower() in TABULAR_SUFFIXES:
                df = _read_tabular(path, path.name)
                if df is not None:
                    yield path.stem, df
        return

    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if Path(info.filename).suffix.lower() not in TABULAR_SUFFIXES:
                    continue
                with zf.open(info) as fh:
                    data = io.BytesIO(fh.read())
                df = _read_tabular(data, info.filename)
                if df is not None:
                    yield Path(info.filename).stem, df
        return

    if source.suffix.lower() in TABULAR_SUFFIXES:
        df = _read_tabular(source, source.name)
        if df is not None:
            yield source.stem, df


def frames_from(source: Path, report: IngestReport) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield (symbol, normalised OHLCV) from one zip / directory / file."""
    for member, raw in _iter_members(source):
        report.files_read += 1
        mapping = canonical_columns(raw.columns)
        renamed = raw.rename(columns=mapping)
        have = set(renamed.columns)

        missing = [c for c in REQUIRED if c not in have]
        if missing:
            report.skipped.append((member, f"missing columns {missing}"))
            continue

        renamed["date"] = pd.to_datetime(renamed["date"], errors="coerce", format="mixed")
        renamed = renamed[renamed["date"].notna()]
        if renamed.empty:
            report.skipped.append((member, "no parseable dates"))
            continue

        if "symbol" in have:
            # Long format: one file, every ticker stacked.
            for symbol, chunk in renamed.groupby("symbol", sort=True):
                sym = str(symbol).strip().upper()
                if not sym:
                    continue
                try:
                    out = normalize_ohlcv(chunk.set_index("date").drop(columns=["symbol"]))
                except Exception as exc:
                    report.skipped.append((f"{member}:{sym}", str(exc)[:120]))
                    continue
                if not out.empty:
                    yield sym, out
        else:
            # Per-ticker file: the symbol is the filename.
            sym = member.strip().upper().split(".")[0]
            try:
                out = normalize_ohlcv(renamed.set_index("date"))
            except Exception as exc:
                report.skipped.append((member, str(exc)[:120]))
                continue
            if not out.empty:
                yield sym, out


def ingest(sources: list[Path], repo, merge: bool = True) -> IngestReport:
    """Load every source into the price repository.

    With `merge`, a symbol appearing in more than one archive is combined rather
    than the later file replacing the earlier — vendors routinely split history
    across yearly zips.
    """
    report = IngestReport()
    seen: set[str] = set()

    for source in sources:
        for symbol, df in frames_from(source, report):
            if merge and symbol in seen:
                existing = repo.load(symbol, use_cache=False)
                if existing is not None and not existing.empty:
                    df = pd.concat([existing, df])
                    df = df[~df.index.duplicated(keep="last")].sort_index()
            elif merge and symbol not in seen:
                existing = repo.load(symbol, use_cache=False)
                if existing is not None and not existing.empty:
                    df = pd.concat([existing, df])
                    df = df[~df.index.duplicated(keep="last")].sort_index()

            repo.save(symbol, df)
            repo._record(
                symbol,
                last_date=str(df.index[-1].date()),
                first_date=str(df.index[0].date()),
                rows=len(df),
                failures=0,
                source="ingest",
            )
            if symbol not in seen:
                seen.add(symbol)
                report.symbols += 1
            report.rows += len(df)

    repo.save_manifest()
    return report
