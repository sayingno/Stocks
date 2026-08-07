"""Where the list of tickers to scan comes from."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

_SPLIT = re.compile(r"[,\s;]+")
BUNDLED = Path(__file__).resolve().parent.parent / "universes"


def _clean(tokens) -> list[str]:
    seen, out = set(), []
    for tok in tokens:
        sym = str(tok).strip().upper()
        if not sym or sym.startswith("#"):
            continue
        if not re.fullmatch(r"[A-Z0-9.\-]{1,10}", sym):
            continue
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def load(spec: str) -> list[str]:
    """Accepts a file path, a bundled universe name, or a comma-separated list."""
    path = Path(spec)
    if not path.exists():
        candidate = BUNDLED / f"{spec}.txt"
        if candidate.exists():
            path = candidate

    if path.exists():
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            col = next(
                (c for c in df.columns if str(c).strip().lower() in ("symbol", "ticker", "act symbol")),
                df.columns[0],
            )
            return _clean(df[col].tolist())
        return _clean(path.read_text().splitlines())

    return _clean(_SPLIT.split(spec))


def download_us_listings(include_etfs: bool = False, timeout: int = 60) -> list[str]:
    """Every US-listed symbol, from Nasdaq Trader's public symbol directory.

    Requires network access to nasdaqtrader.com.
    """
    import urllib.request

    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", "replace")

    df = pd.read_csv(io.StringIO(text), sep="|")
    df = df[df["Symbol"].notna()]
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"] != "Y"]
    if not include_etfs and "ETF" in df.columns:
        df = df[df["ETF"] != "Y"]
    # Drop warrants, units, preferreds and other non-common share classes.
    df = df[~df["Symbol"].astype(str).str.contains(r"[.$]", regex=True)]
    return _clean(df["Symbol"].tolist())
