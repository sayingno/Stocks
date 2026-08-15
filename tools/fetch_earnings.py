"""Build an earnings calendar CSV. Run this where the network works.

Nothing in this repo can reach an exchange or a data vendor from a sandboxed
session, so the calendar is produced here, on your machine, and then handed to
the scanner:

    pip install akshare            # A-shares
    python tools/fetch_earnings.py --market cn --out earnings.csv

    pip install yfinance           # US
    python tools/fetch_earnings.py --market us --symbols universes/us_all.txt --out earnings.csv

    python -m qscan study --earnings earnings.csv --event earnings ...

Output is two columns, `symbol,date`, one row per disclosure. Symbols are
written as text so zero-padded codes survive the round trip.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def fetch_cn(quiet: bool = False) -> pd.DataFrame:
    """Every A-share disclosure date, from cninfo via akshare.

    akshare exposes this a quarter at a time, so this walks the report periods
    and concatenates. Expect a few minutes for a full history.
    """
    try:
        import akshare as ak
    except ImportError:
        sys.exit("pip install akshare")

    frames = []
    periods = [f"{y}{q}" for y in range(2015, pd.Timestamp.today().year + 1)
               for q in ("0331", "0630", "0930", "1231")]
    for period in periods:
        if pd.Timestamp(period) > pd.Timestamp.today():
            break
        for fn in ("stock_report_disclosure", "stock_zh_a_disclosure_report_cninfo"):
            try:
                df = getattr(ak, fn)(period=period)
                if df is not None and len(df):
                    frames.append(df)
                    if not quiet:
                        print(f"  {period}: {len(df)} rows", file=sys.stderr)
                break
            except Exception as exc:
                if fn == "stock_zh_a_disclosure_report_cninfo" and not quiet:
                    print(f"  {period}: {str(exc)[:70]}", file=sys.stderr)

    if not frames:
        sys.exit("no data returned — check the akshare version and your network")

    raw = pd.concat(frames, ignore_index=True)
    lower = {str(c).strip().lower(): c for c in raw.columns}
    sym = next((lower[k] for k in ("代码", "股票代码", "symbol", "code") if k in lower), None)
    dt = next((lower[k] for k in ("实际披露时间", "首次预约时间", "公告日期", "date") if k in lower), None)
    if sym is None or dt is None:
        sys.exit(f"unexpected columns from akshare: {list(raw.columns)}")

    return pd.DataFrame({
        "symbol": raw[sym].astype(str).str.strip().str.zfill(6),
        "date": pd.to_datetime(raw[dt], errors="coerce"),
    })


def fetch_us(symbols: list[str], quiet: bool = False) -> pd.DataFrame:
    """Earnings dates per ticker via yfinance. One request each — slow."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("pip install yfinance")

    rows = []
    for n, sym in enumerate(symbols, 1):
        try:
            df = yf.Ticker(sym).get_earnings_dates(limit=80)
            if df is not None and len(df):
                for when in pd.DatetimeIndex(df.index):
                    rows.append({"symbol": sym, "date": pd.Timestamp(when).tz_localize(None).normalize()})
        except Exception as exc:
            if not quiet:
                print(f"  {sym}: {str(exc)[:60]}", file=sys.stderr)
        if not quiet and n % 100 == 0:
            print(f"  {n}/{len(symbols)}", file=sys.stderr)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", choices=["cn", "us"], required=True)
    ap.add_argument("--symbols", help="file of tickers, one per line (us only)")
    ap.add_argument("--out", default="earnings.csv")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.market == "cn":
        df = fetch_cn(args.quiet)
    else:
        if not args.symbols:
            sys.exit("--symbols is required for --market us")
        syms = [s.strip().upper() for s in Path(args.symbols).read_text().splitlines()
                if s.strip() and not s.startswith("#")]
        df = fetch_us(syms, args.quiet)

    df = df[df["date"].notna()].drop_duplicates().sort_values(["symbol", "date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(args.out, index=False)
    print(f"{len(df):,} disclosures for {df['symbol'].nunique():,} symbols -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
