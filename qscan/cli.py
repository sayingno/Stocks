"""Command line entry point.

    python -m qscan scan     --universe sample                # today's candidates
    python -m qscan history  --universe sample --years 5      # past setups + charts
    python -m qscan explain  --symbol NVDA --date 2023-05-24  # why it did/didn't pass
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from . import universe as universe_mod
from .charts import contact_sheet, plot_signal
from .config import PRESETS, BreakoutConfig
from .data import DataError, PriceStore, default_window
from .indicators import annotate
from .outcomes import TradeRules, forward_returns, simulate_trade, summarise
from .setup_breakout import _Arrays, dedupe_signals, evaluate_bar, scan_symbol

SCAN_COLUMNS = [
    "date", "symbol", "state", "score", "close", "pivot", "dist_from_pivot",
    "entry", "stop", "risk_pct", "risk_in_adr", "shares", "position_value",
    "target_2r", "target_3r", "adr20", "dollar_vol20", "impulse_gain",
    "ret_21d", "ret_63d", "ret_126d", "rs_63d_rank", "base_len", "base_depth",
    "contraction", "vol_dryup", "surf_ma", "surf_dist", "pivot_date",
]

HISTORY_COLUMNS = SCAN_COLUMNS + [
    "outcome", "fill_date", "fill", "r_multiple", "mfe_r", "mae_r",
    "days_held", "exit_date", "exit_price", "fwd_5d", "fwd_10d", "fwd_20d",
    "fwd_60d", "mfe_20d", "mae_20d", "chart",
]


# --------------------------------------------------------------------------
def _build_config(args: argparse.Namespace) -> BreakoutConfig:
    cfg = PRESETS[args.preset]
    overrides: dict[str, Any] = {}
    for field in ("min_price", "min_dollar_volume", "min_adr_pct", "max_base_depth",
                  "min_base_len", "max_base_len", "max_dist_from_pivot",
                  "account_size", "risk_pct", "max_position_pct"):
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    if getattr(args, "set", None):
        for pair in args.set:
            key, _, raw = pair.partition("=")
            if not _:
                raise SystemExit(f"--set expects key=value, got {pair!r}")
            overrides[key.strip()] = json.loads(raw) if raw[:1] in "[{\"-0123456789tfn" else raw
    return cfg.with_overrides(**overrides) if overrides else cfg


def _make_store(args: argparse.Namespace) -> PriceStore:
    return PriceStore(
        provider=args.provider,
        cache_dir=args.cache_dir,
        csv_dir=args.csv_dir,
        offline=args.offline,
    )


def _write(df: pd.DataFrame, path: Path, columns: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in columns if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    df[cols + extra].to_csv(path, index=False)
    return path


def _print_table(df: pd.DataFrame, cols: list[str], limit: int) -> None:
    show = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 50, "display.float_format", "{:,.3f}".format):
        print(df[show].head(limit).to_string(index=False))


# --------------------------------------------------------------------------
def cmd_scan(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    store = _make_store(args)
    symbols = universe_mod.load(args.universe)
    start, end = default_window(years=2)
    if args.asof:
        end = args.asof

    failures: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []
    rs_input: dict[str, float] = {}

    print(f"scanning {len(symbols)} symbols via {store.name} …", file=sys.stderr)
    for n, sym in enumerate(symbols, 1):
        if args.verbose and n % 100 == 0:
            print(f"  {n}/{len(symbols)}", file=sys.stderr)
        try:
            df = store.get(sym, start, end, refresh=args.refresh)
        except Exception as exc:
            failures.append((sym, str(exc)))
            continue
        if len(df) < 140:
            continue
        ann = annotate(df, cfg)
        if "ret_3m" in ann and pd.notna(ann["ret_3m"].iloc[-1]):
            rs_input[sym] = float(ann["ret_3m"].iloc[-1])
        hits = scan_symbol(df, cfg, symbol=sym, keep_rejects=False)
        rows.extend(hits)

    if not rows:
        print("no candidates today.", file=sys.stderr)
        if failures:
            print(f"({len(failures)} symbols failed to load; first: {failures[:3]})", file=sys.stderr)
        return 0

    out = pd.DataFrame(rows)
    if rs_input:
        ranks = pd.Series(rs_input).rank(pct=True) * 100
        out["rs_63d_rank"] = out["symbol"].map(ranks).round(1)
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    if args.state:
        out = out[out["state"] == args.state]
    if args.min_score:
        out = out[out["score"] >= args.min_score]

    path = Path(args.out) / f"scan_{date.today().isoformat()}.csv"
    _write(out, path, SCAN_COLUMNS)
    print(f"\n{len(out)} candidates -> {path}\n", file=sys.stderr)
    _print_table(out, ["date", "symbol", "state", "score", "close", "pivot", "entry", "stop",
                       "risk_pct", "shares", "adr20", "base_len", "base_depth", "impulse_gain"], args.limit)

    if args.charts:
        chart_dir = Path(args.out) / "charts"
        made = []
        for _, row in out.head(args.charts).iterrows():
            sym = row["symbol"]
            ann = annotate(store.get(sym, start, end), cfg)
            made.append(plot_signal(ann, row.to_dict(), chart_dir / f"{sym}_{pd.Timestamp(row['date']).date()}.png"))
        print(f"{len(made)} charts -> {chart_dir}", file=sys.stderr)
    if failures and args.verbose:
        print(f"{len(failures)} load failures", file=sys.stderr)
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    store = _make_store(args)
    symbols = universe_mod.load(args.universe)
    start, end = default_window(years=args.years)
    rules = TradeRules(trail_ma=args.trail_ma, partial_days=args.partial_days)

    all_rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    print(f"sweeping {len(symbols)} symbols over ~{args.years}y via {store.name} …", file=sys.stderr)

    for n, sym in enumerate(symbols, 1):
        if args.verbose and n % 50 == 0:
            print(f"  {n}/{len(symbols)} ({len(all_rows)} signals)", file=sys.stderr)
        try:
            df = store.get(sym, start, end, refresh=args.refresh)
        except Exception:
            continue
        if len(df) < 200:
            continue
        ann = annotate(df, cfg)
        a = _Arrays.from_frame(ann)
        warmup = max(cfg.ma_slow, 126, cfg.max_base_len + 5)
        hits = []
        for i in range(warmup, len(a)):
            res = evaluate_bar(a, i, cfg)
            if res["passed"]:
                res["symbol"] = sym
                res["bar"] = i
                hits.append(res)
        hits = dedupe_signals(hits, cooldown=args.cooldown)
        for hit in hits:
            hit.update(forward_returns(ann, hit["bar"]))
            hit.update(simulate_trade(ann, hit, rules))
        all_rows.extend(hits)
        if hits:
            frames[sym] = ann

    if not all_rows:
        print("no historical signals found with these settings.", file=sys.stderr)
        return 0

    out = pd.DataFrame(all_rows).sort_values(["date", "symbol"]).reset_index(drop=True)
    outdir = Path(args.out)
    stats = summarise(all_rows)

    if args.charts:
        chart_dir = outdir / "charts"
        top = out.sort_values("score", ascending=False).head(args.charts)
        paths = []
        for _, row in top.iterrows():
            sym = row["symbol"]
            name = f"{sym}_{pd.Timestamp(row['date']).date()}.png"
            try:
                p = plot_signal(frames[sym], row.to_dict(), chart_dir / name, outcome=row.to_dict())
                paths.append(p)
                out.loc[out.index == row.name, "chart"] = str(p)
            except Exception as exc:
                print(f"  chart failed for {sym}: {exc}", file=sys.stderr)
        if args.contact_sheet and paths:
            sheet = contact_sheet(paths, outdir / "contact_sheet.png")
            print(f"contact sheet -> {sheet}", file=sys.stderr)
        print(f"{len(paths)} charts -> {chart_dir}", file=sys.stderr)

    path = _write(out, outdir / f"history_{args.years}y.csv", HISTORY_COLUMNS)
    (outdir / "history_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(f"\n{len(out)} historical setups -> {path}", file=sys.stderr)
    print(json.dumps(stats, indent=2, default=str))
    print()
    _print_table(
        out.sort_values("r_multiple", ascending=False),
        ["date", "symbol", "state", "score", "close", "impulse_gain", "base_len",
         "base_depth", "adr20", "outcome", "r_multiple", "mfe_r", "days_held"],
        args.limit,
    )
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    store = _make_store(args)
    start, end = default_window(years=args.years)
    df = store.get(args.symbol, start, end, refresh=args.refresh)
    ann = annotate(df, cfg)
    a = _Arrays.from_frame(ann)

    if args.date:
        target = pd.Timestamp(args.date)
        pos = ann.index.searchsorted(target)
        i = int(min(max(pos, 0), len(ann) - 1))
    else:
        i = len(ann) - 1

    res = evaluate_bar(a, i, cfg)
    res["symbol"] = args.symbol
    res["bar"] = i
    verdict = "PASS" if res["passed"] else f"FAIL at gate: {res['reject']}"
    print(f"\n{args.symbol} @ {ann.index[i].date()} -> {verdict}\n")
    for key, value in res.items():
        if key in ("passed", "reject", "bar"):
            continue
        if isinstance(value, float):
            print(f"  {key:<22} {value:,.4f}")
        else:
            print(f"  {key:<22} {value}")

    if args.chart:
        out = Path(args.out) / f"{args.symbol}_{ann.index[i].date()}_explain.png"
        res.setdefault("state", "n/a")
        plot_signal(ann, res, out, title_extra=verdict)
        print(f"\nchart -> {out}")
    return 0


# --------------------------------------------------------------------------
def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", choices=sorted(PRESETS), default="default")
    p.add_argument("--provider", default="yfinance", help="yfinance | stooq | tiingo")
    p.add_argument("--csv-dir", default=None, help="use a local directory of TICKER.csv files instead")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--offline", action="store_true", help="never hit the network; cache only")
    p.add_argument("--refresh", action="store_true", help="ignore the cache and refetch")
    p.add_argument("--out", default="out")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any config field")
    for field, kind in (("min_price", float), ("min_dollar_volume", float), ("min_adr_pct", float),
                        ("max_base_depth", float), ("min_base_len", int), ("max_base_len", int),
                        ("max_dist_from_pivot", float), ("account_size", float),
                        ("risk_pct", float), ("max_position_pct", float)):
        p.add_argument(f"--{field.replace('_', '-')}", dest=field, type=kind, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qscan", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="today's breakout candidates")
    _common(s)
    s.add_argument("--universe", default="sample")
    s.add_argument("--asof", default=None, help="pretend today is this date (YYYY-MM-DD)")
    s.add_argument("--state", choices=["setup", "breakout"], default=None)
    s.add_argument("--min-score", type=float, default=0.0)
    s.add_argument("--charts", type=int, default=0, metavar="N", help="render charts for the top N")
    s.set_defaults(func=cmd_scan)

    h = sub.add_parser("history", help="find past setups and what happened next")
    _common(h)
    h.add_argument("--universe", default="sample")
    h.add_argument("--years", type=int, default=5)
    h.add_argument("--cooldown", type=int, default=10, help="bars before the same name can re-signal")
    h.add_argument("--charts", type=int, default=20, metavar="N")
    h.add_argument("--contact-sheet", action="store_true")
    h.add_argument("--trail-ma", type=int, default=20, choices=[10, 20])
    h.add_argument("--partial-days", type=int, default=4)
    h.set_defaults(func=cmd_history)

    e = sub.add_parser("explain", help="show every measured value for one symbol/date")
    _common(e)
    e.add_argument("--symbol", required=True)
    e.add_argument("--date", default=None)
    e.add_argument("--years", type=int, default=3)
    e.add_argument("--chart", action="store_true")
    e.set_defaults(func=cmd_explain)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DataError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
