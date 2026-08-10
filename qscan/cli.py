"""Command line entry point.

    python -m qscan update   --universe us_all                # refresh the price database
    python -m qscan daily    --universe us_all                # update + scan + charts + report
    python -m qscan scan     --universe sample                # today's candidates
    python -m qscan history  --universe sample --years 5      # past setups + charts
    python -m qscan explain  --symbol NVDA --date 2023-05-24  # why it did/didn't pass

`daily` is the one to schedule. Everything else is for exploring by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import universe as universe_mod
from .charts import contact_sheet, plot_signal
from .config import PRESETS, BreakoutConfig
from .data import DataError, PriceStore, default_window
from .indicators import annotate
from .outcomes import TradeRules, forward_returns, simulate_trade, summarise
from . import ingest as ingest_mod
from . import portfolio, setups, strength
from .portfolio import PortfolioConfig
from .report import build_report
from .repository import PriceRepository
from .setup_breakout import (
    _Arrays,
    dedupe_signals,
    evaluate_bar,
    rs_array as rs_arr,
    scan_frame,
    scan_symbol,
)

SCAN_COLUMNS = [
    "date", "symbol", "state", "score", "close", "pivot", "dist_from_pivot",
    "entry", "stop", "risk_pct", "risk_in_adr", "shares", "position_value",
    "target_2r", "target_3r", "adr20", "dollar_vol20", "impulse_gain",
    "ret_21d", "ret_63d", "ret_126d", "rs_rank", "base_len", "base_depth",
    "contraction", "vol_dryup", "surf_ma", "surf_dist", "pivot_date",
]

HISTORY_COLUMNS = SCAN_COLUMNS + [
    "outcome", "fill_date", "fill", "r_multiple", "mfe_r", "mae_r",
    "days_held", "exit_date", "exit_price", "fwd_5d", "fwd_10d", "fwd_20d",
    "fwd_60d", "mfe_20d", "mae_20d", "chart",
]


# --------------------------------------------------------------------------
def _setup(args: argparse.Namespace):
    return setups.get(getattr(args, "setup", "breakout"))


def _build_config(args: argparse.Namespace):
    """Resolve the preset for the chosen setup, then apply CLI overrides.

    Overrides are filtered to fields the chosen setup actually has, so
    --min-adr-pct works everywhere while --max-base-depth only binds on the
    breakout rather than erroring on the others.
    """
    setup = _setup(args)
    cfg = setup.config(args.preset)
    valid = set(cfg.to_dict())
    overrides: dict[str, Any] = {}
    for field in ("min_price", "min_dollar_volume", "min_adr_pct", "max_base_depth",
                  "min_base_len", "max_base_len", "max_dist_from_pivot",
                  "account_size", "risk_pct", "max_position_pct", "min_rs_rank"):
        value = getattr(args, field, None)
        if value is not None and field in valid:
            overrides[field] = value
    if getattr(args, "set", None):
        for pair in args.set:
            key, _, raw = pair.partition("=")
            if not _:
                raise SystemExit(f"--set expects key=value, got {pair!r}")
            overrides[key.strip()] = json.loads(raw) if raw[:1] in "[{\"-0123456789tfn" else raw
    return cfg.with_overrides(**overrides) if overrides else cfg


class _Source:
    """Uniform read interface over either the price database or the ad-hoc cache.

    `--repo DIR` reads the incrementally-maintained database written by
    `qscan update`, which is what a scheduled run should use: no network, no
    per-symbol staleness checks. Without it, symbols are fetched on demand.
    """

    def __init__(self, args: argparse.Namespace):
        repo_dir = getattr(args, "repo", None)
        if repo_dir:
            self.repo: PriceRepository | None = PriceRepository(
                root=repo_dir, provider=args.provider, csv_dir=args.csv_dir
            )
            self.name = f"db:{repo_dir}"
        else:
            self.repo = None
            self.store = PriceStore(
                provider=args.provider,
                cache_dir=args.cache_dir,
                csv_dir=args.csv_dir,
                offline=args.offline,
            )
            self.name = self.store.name

    def get(self, symbol: str, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
        if self.repo is None:
            return self.store.get(symbol, start, end, refresh=refresh)
        df = self.repo.load(symbol)
        if df is None or df.empty:
            raise DataError(f"{symbol}: not in the price database (run `qscan update`)")
        return df.loc[str(start) : str(end)]


def _make_store(args: argparse.Namespace) -> _Source:
    return _Source(args)


def _history_window(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve --start/--end, falling back to --years back from today."""
    end = args.end or date.today().isoformat()
    if args.start:
        return args.start, end
    start, _ = default_window(years=args.years)
    return pd.Timestamp(end).date().replace(year=pd.Timestamp(end).year - args.years).isoformat(), end


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
    setup = _setup(args)
    cfg = _build_config(args)
    store = _make_store(args)
    symbols = universe_mod.load(args.universe)
    start, end = default_window(years=2)
    if args.asof:
        end = args.asof

    failures: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    latest_returns: dict[str, tuple[float, float, float]] = {}

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
        frames[sym] = ann
        row = ann.iloc[-1]
        latest_returns[sym] = tuple(
            float(row[c]) if pd.notna(row[c]) else float("nan") for c in ("ret_1m", "ret_3m", "ret_6m")
        )

    ranks = strength.latest_ranks(latest_returns, cfg.rs_weights)
    if cfg.min_rs_rank is not None and not ranks:
        print("(universe too small to rank; RS gate disabled)", file=sys.stderr)
        cfg = cfg.with_overrides(min_rs_rank=None)

    for sym, ann in frames.items():
        rows.extend(setups.scan_frame(ann, setup, cfg, symbol=sym, rs_value=ranks.get(sym)))

    if not rows:
        print("no candidates today.", file=sys.stderr)
        if failures:
            print(f"({len(failures)} symbols failed to load; first: {failures[:3]})", file=sys.stderr)
        return 0

    out = pd.DataFrame(rows)
    if ranks:
        out["rs_rank"] = out["symbol"].map(ranks).round(1)
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


def _sweep_history(args: argparse.Namespace, cfg: BreakoutConfig):
    """Find every historical setup in the universe. Shared by history and backtest.

    Returns (signals, annotated_frames, window).
    """
    setup = _setup(args)
    store = _make_store(args)
    symbols = universe_mod.load(args.universe)
    start, end = _history_window(args)
    rules = TradeRules(trail_ma=args.trail_ma, partial_days=args.partial_days)
    # Indicators need roughly six months of run-up before the first testable
    # bar, so load earlier than the cutoff and only emit signals on or after it.
    load_start = (pd.Timestamp(start) - pd.Timedelta(days=400)).date().isoformat()

    print(f"sweeping {len(symbols)} symbols for {setup.label} {start} → {end} via {store.name} …", file=sys.stderr)

    # ---- pass 1: load and annotate -----------------------------------------
    frames: dict[str, pd.DataFrame] = {}
    for n, sym in enumerate(symbols, 1):
        if args.verbose and n % 200 == 0:
            print(f"  loading {n}/{len(symbols)}", file=sys.stderr)
        try:
            df = store.get(sym, load_start, end, refresh=args.refresh)
        except Exception:
            continue
        if len(df) < 200:
            continue
        frames[sym] = annotate(df, cfg)

    if not frames:
        return [], {}, (start, end)

    # ---- pass 2: cross-sectional relative strength --------------------------
    rs_panel = pd.DataFrame()
    if cfg.min_rs_rank is not None:
        rs_panel = strength.build_panel(frames, cfg.rs_weights)
        if rs_panel.empty:
            print("  (universe too small to rank; RS gate disabled)", file=sys.stderr)
            cfg = cfg.with_overrides(min_rs_rank=None)
        elif args.verbose:
            print(f"  ranked {rs_panel.shape[1]} symbols over {rs_panel.shape[0]} dates", file=sys.stderr)

    # ---- pass 3: detect ------------------------------------------------------
    cutoff = pd.Timestamp(start)
    warmup = max(cfg.ma_slow, 126, getattr(cfg, "max_base_len", 0) + 5,
                 getattr(cfg, "dormancy_lookback", 0) + 25,
                 getattr(cfg, "run_lookback", 0) + 25)
    all_rows: list[dict[str, Any]] = []
    kept: dict[str, pd.DataFrame] = {}

    for n, (sym, ann) in enumerate(frames.items(), 1):
        if args.verbose and n % 200 == 0:
            print(f"  scanning {n}/{len(frames)} ({len(all_rows)} signals)", file=sys.stderr)
        rs_series = strength.series_for(rs_panel, sym) if not rs_panel.empty else None
        first = max(warmup, int(ann.index.searchsorted(cutoff)))
        hits = setups.scan_frame(
            ann, setup, cfg, symbol=sym, positions=range(first, len(ann)), rs_series=rs_series
        )
        cooldown = args.cooldown if args.cooldown is not None else setup.cooldown
        hits = dedupe_signals(hits, cooldown=cooldown)
        for hit in hits:
            hit.update(forward_returns(ann, hit["bar"], direction=setup.direction))
            hit.update(simulate_trade(ann, hit, rules))
        all_rows.extend(hits)
        if hits:
            kept[sym] = ann

    return all_rows, kept, (start, end)


def cmd_history(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    all_rows, frames, (start, end) = _sweep_history(args, cfg)

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

    path = _write(out, outdir / f"history_{start}_{end}.csv", HISTORY_COLUMNS)
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


def cmd_ingest(args: argparse.Namespace) -> int:
    """Load bulk vendor archives into the price database."""
    from .repository import PriceRepository

    sources = [Path(p) for p in args.sources]
    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        raise DataError(f"no such path: {missing}")

    repo = PriceRepository(root=args.repo or "data", provider=args.provider, csv_dir=args.csv_dir)
    print(f"ingesting {len(sources)} source(s) into {repo.price_dir} …", file=sys.stderr)
    report = ingest_mod.ingest(sources, repo, merge=not args.no_merge)

    print(json.dumps({"ingest": report.as_dict(), "coverage": repo.coverage()}, indent=2, default=str))
    if report.skipped:
        print(f"\n{len(report.skipped)} file(s) skipped — first few:", file=sys.stderr)
        for name, why in report.skipped[:8]:
            print(f"  {name}: {why}", file=sys.stderr)
    if report.symbols:
        uni = Path(args.repo or "data") / "universe.txt"
        uni.write_text("\n".join(sorted(repo.symbols())) + "\n")
        print(f"\nuniverse written to {uni} ({len(repo.symbols())} symbols)", file=sys.stderr)
    return 0 if report.symbols else 1


def cmd_backtest(args: argparse.Namespace) -> int:
    """Replay every historical setup as one account, with real constraints."""
    cfg = _build_config(args)
    all_rows, frames, (start, end) = _sweep_history(args, cfg)
    outdir = Path(args.out)

    if not all_rows:
        print("no signals in this window — nothing to backtest.", file=sys.stderr)
        return 0

    pcfg = PortfolioConfig(
        starting_equity=args.starting_equity,
        risk_pct=args.risk_pct if args.risk_pct is not None else cfg.risk_pct,
        max_positions=args.max_positions,
        max_position_pct=cfg.max_position_pct,
        max_exposure_pct=args.max_exposure_pct,
    )
    print(
        f"backtesting {len(all_rows)} signals · ${pcfg.starting_equity:,.0f} · "
        f"{pcfg.risk_pct:.2%} risk · max {pcfg.max_positions} positions",
        file=sys.stderr,
    )
    result = portfolio.run(all_rows, pcfg)

    if result.trades.empty:
        print("no signal ever triggered — nothing to report.", file=sys.stderr)
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    signals_df = pd.DataFrame(all_rows).sort_values(["date", "symbol"]).reset_index(drop=True)
    _write(signals_df, outdir / f"backtest_signals_{start}_{end}.csv", HISTORY_COLUMNS)
    result.trades.to_csv(outdir / f"backtest_trades_{start}_{end}.csv", index=False)
    result.equity.to_csv(outdir / f"backtest_equity_{start}_{end}.csv", index=False)
    (outdir / "backtest_stats.json").write_text(json.dumps(result.stats, indent=2, default=str))

    curve = portfolio.plot_equity(result, outdir / f"equity_{start}_{end}.png", title=f"Breakout setup · {start} → {end}")
    if curve:
        print(f"equity curve -> {curve}", file=sys.stderr)

    if args.charts:
        chart_dir = outdir / "charts"
        taken = set(zip(result.trades["symbol"], pd.to_datetime(result.trades["fill_date"])))
        rendered = 0
        for _, row in signals_df.sort_values("score", ascending=False).iterrows():
            if rendered >= args.charts:
                break
            key = (row["symbol"], pd.Timestamp(row.get("fill_date")))
            if key not in taken:
                continue  # only chart trades the portfolio actually took
            try:
                plot_signal(
                    frames[row["symbol"]],
                    row.to_dict(),
                    chart_dir / f"{row['symbol']}_{pd.Timestamp(row['date']).date()}.png",
                    outcome=row.to_dict(),
                )
                rendered += 1
            except Exception as exc:
                print(f"  chart failed for {row['symbol']}: {exc}", file=sys.stderr)
        print(f"{rendered} charts -> {chart_dir}", file=sys.stderr)

    print(json.dumps(result.stats, indent=2, default=str))
    print()
    best = result.trades.sort_values("pnl", ascending=False)
    _print_table(
        pd.concat([best.head(args.limit // 2), best.tail(max(1, args.limit // 2))]),
        ["symbol", "fill_date", "exit_date", "days_held", "shares", "fill", "r_multiple", "pnl", "equity_after"],
        args.limit,
    )
    return 0


def _progress(done: int, total: int, stats) -> None:
    print(
        f"  {done}/{total}  +{stats.new_rows:,} bars  "
        f"({stats.created} new, {stats.appended} appended, {stats.failed} failed)",
        file=sys.stderr,
    )


def cmd_update(args: argparse.Namespace) -> int:
    """Bring the local price database up to date. Safe to interrupt and re-run."""
    repo = PriceRepository(
        root=args.repo or "data",
        provider=args.provider,
        csv_dir=args.csv_dir,
        history_years=args.history_years,
    )
    symbols = universe_mod.load(args.universe)
    print(f"updating {len(symbols)} symbols from {repo.provider_name} …", file=sys.stderr)

    stats = repo.update(
        symbols,
        workers=args.workers,
        pause=args.pause,
        force=args.force,
        progress=_progress if args.verbose else None,
    )
    print(json.dumps({"update": stats.as_dict(), "coverage": repo.coverage()}, indent=2, default=str))
    return 0


def cmd_daily(args: argparse.Namespace) -> int:
    """The scheduled job: refresh data, apply the filters, render charts, write a report."""
    started = datetime.now()
    setup = _setup(args)
    cfg = _build_config(args)
    repo_root = args.repo or "data"
    repo = PriceRepository(
        root=repo_root, provider=args.provider, csv_dir=args.csv_dir, history_years=args.history_years
    )
    symbols = universe_mod.load(args.universe)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 1. refresh ---------------------------------------------------------
    update_stats: dict[str, Any] = {}
    problems: list[str] = []
    if args.skip_update:
        print("skipping data refresh (--skip-update)", file=sys.stderr)
    else:
        print(f"[1/4] refreshing {len(symbols)} symbols from {repo.provider_name} …", file=sys.stderr)
        stats = repo.update(
            symbols,
            workers=args.workers,
            pause=args.pause,
            force=args.force,
            progress=_progress if args.verbose else None,
        )
        update_stats = stats.as_dict()
        print(
            f"      +{stats.new_rows:,} bars · {stats.created} new · {stats.appended} appended · "
            f"{stats.readjusted} re-adjusted · {stats.failed} failed",
            file=sys.stderr,
        )
        # A scheduled job that silently reports "0 candidates" because the feed
        # broke is worse than one that fails loudly.
        if stats.checked:
            fail_ratio = stats.failed / stats.checked
            if fail_ratio > args.max_failure_ratio:
                problems.append(
                    f"{stats.failed}/{stats.checked} fetches failed "
                    f"({fail_ratio:.0%} > {args.max_failure_ratio:.0%} allowed)"
                )

    coverage = repo.coverage()
    latest = coverage.get("latest_date")
    if latest:
        stale_days = (date.today() - date.fromisoformat(str(latest))).days
        if stale_days > args.max_stale_days:
            problems.append(f"newest bar is {latest} ({stale_days} days old)")
    elif not args.skip_update:
        problems.append("price database is empty")

    # ---- 2. filter ----------------------------------------------------------
    print(f"[2/4] applying filters to {len(symbols)} symbols …", file=sys.stderr)
    rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    missing = 0

    # Relative strength needs the whole universe ranked before any symbol can be
    # judged, so collect the latest returns first.
    latest_returns: dict[str, tuple[float, float, float]] = {}
    for sym in symbols:
        df = repo.load(sym)
        if df is None or len(df) < 140:
            missing += 1
            continue
        ann = annotate(df, cfg)
        frames[sym] = ann
        row = ann.iloc[-1]
        latest_returns[sym] = (
            float(row["ret_1m"]) if pd.notna(row["ret_1m"]) else float("nan"),
            float(row["ret_3m"]) if pd.notna(row["ret_3m"]) else float("nan"),
            float(row["ret_6m"]) if pd.notna(row["ret_6m"]) else float("nan"),
        )

    ranks = strength.latest_ranks(latest_returns, cfg.rs_weights)
    if cfg.min_rs_rank is not None and not ranks:
        print("      (universe too small to rank; RS gate disabled)", file=sys.stderr)
        cfg = cfg.with_overrides(min_rs_rank=None)

    for sym, ann in frames.items():
        hits = setups.scan_frame(ann, setup, cfg, symbol=sym, rs_value=ranks.get(sym))
        if hits:
            rows.extend(hits)

    frames = {s: f for s, f in frames.items() if any(r["symbol"] == s for r in rows)}

    if rows:
        out = pd.DataFrame(rows)
        if ranks:
            out["rs_rank"] = out["symbol"].map(ranks).round(1)
        out = out[out["score"] >= args.min_score]
        if args.state:
            out = out[out["state"] == args.state]
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=["symbol", "state", "score"])
    print(f"      {len(out)} candidates ({missing} symbols short on history)", file=sys.stderr)

    # ---- 3. charts ----------------------------------------------------------
    stamp = date.today().isoformat()
    chart_dir = outdir / "charts" / stamp
    chart_paths: dict[str, str] = {}
    if len(out) and args.charts:
        print(f"[3/4] rendering top {min(args.charts, len(out))} charts …", file=sys.stderr)
        for _, row in out.head(args.charts).iterrows():
            sym = row["symbol"]
            try:
                p = plot_signal(frames[sym], row.to_dict(), chart_dir / f"{sym}.png")
                chart_paths[sym] = str(p)
            except Exception as exc:
                print(f"      chart failed for {sym}: {exc}", file=sys.stderr)
    else:
        print("[3/4] no charts to render", file=sys.stderr)

    # ---- 4. report ----------------------------------------------------------
    print("[4/4] writing report …", file=sys.stderr)
    csv_path = outdir / f"scan_{stamp}.csv"
    _write(out, csv_path, SCAN_COLUMNS)
    report_path = build_report(
        out,
        outdir / f"report_{stamp}.html",
        charts=chart_paths,
        update_stats=update_stats,
        coverage=coverage,
        universe_size=len(symbols),
        config_note=f"preset {args.preset}",
    )
    latest = outdir / "latest.html"
    latest.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    elapsed = (datetime.now() - started).total_seconds()
    summary = {
        "date": stamp,
        "universe": len(symbols),
        "candidates": len(out),
        "breakouts": int((out.get("state") == "breakout").sum()) if len(out) else 0,
        "setups": int((out.get("state") == "setup").sum()) if len(out) else 0,
        "charts": len(chart_paths),
        "report": str(report_path),
        "latest": str(latest),
        "csv": str(csv_path),
        "data_as_of": coverage.get("latest_date"),
        "elapsed_sec": round(elapsed, 1),
        "problems": problems,
    }
    print(json.dumps(summary, indent=2, default=str))
    if len(out):
        print()
        _print_table(out, ["symbol", "state", "score", "close", "pivot", "entry", "stop",
                           "risk_pct", "shares", "adr20", "impulse_gain", "base_len"], args.limit)

    if problems:
        # The report is still written — an operator wants to see how far it got.
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        return 1
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    setup = _setup(args)
    cfg = _build_config(args)
    store = _make_store(args)
    start, end = default_window(years=args.years)
    df = store.get(args.symbol, start, end, refresh=args.refresh)

    if df.empty:
        # Almost always a window/data mismatch rather than a missing symbol, so
        # say which it is instead of dying on an index error further down.
        full = store.get(args.symbol, "1900-01-01", "2100-01-01")
        if full.empty:
            raise DataError(f"{args.symbol}: no data at all from {store.name}")
        raise DataError(
            f"{args.symbol}: no bars between {start} and {end}; "
            f"available range is {full.index[0].date()} to {full.index[-1].date()} "
            f"(widen with --years, or pass --date inside that range)"
        )

    ann = annotate(df, cfg)
    if len(ann) == 0:
        raise DataError(f"{args.symbol}: {len(df)} raw bars but none usable after cleaning")
    a = _Arrays.from_frame(ann)

    if args.date:
        target = pd.Timestamp(args.date)
        pos = ann.index.searchsorted(target)
        i = int(min(max(pos, 0), len(ann) - 1))
    else:
        i = len(ann) - 1

    res = setup.evaluate(a, i, cfg, rs=None)
    res["symbol"] = args.symbol
    res["bar"] = i
    verdict = "PASS" if res["passed"] else f"FAIL at gate: {res['reject']}"
    print(f"\n{args.symbol} @ {ann.index[i].date()} [{setup.label}] -> {verdict}\n")
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
    # argparse runs help through %-formatting, so any literal % must be doubled.
    p.add_argument("--setup", choices=list(setups.NAMES), default="breakout",
                   help="; ".join(
                       f"{n}: {setups.REGISTRY[n].blurb}".replace("%", "%%") for n in setups.NAMES
                   ))
    p.add_argument("--preset", choices=["default", "relaxed", "strict"], default="default")
    p.add_argument("--provider", default="yfinance", help="yfinance | stooq | tiingo")
    p.add_argument("--csv-dir", default=None, help="use a local directory of TICKER.csv files instead")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--repo", default=None, metavar="DIR",
                   help="read/write the incremental price database at DIR (e.g. 'data')")
    p.add_argument("--offline", action="store_true", help="never hit the network; cache only")
    p.add_argument("--refresh", action="store_true", help="ignore the cache and refetch")
    p.add_argument("--out", default="out")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any config field")
    p.add_argument("--min-rs-rank", dest="min_rs_rank", type=float, default=None,
                   help="require this cross-sectional 1m/3m/6m strength percentile (0-100)")
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

    def _sweep_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--universe", default="sample")
        p.add_argument("--start", default=None, metavar="YYYY-MM-DD",
                       help="only emit signals on or after this date (e.g. 2018-01-01)")
        p.add_argument("--end", default=None, metavar="YYYY-MM-DD")
        p.add_argument("--years", type=int, default=5, help="used only when --start is omitted")
        p.add_argument("--cooldown", type=int, default=None,
                       help="bars before the same name can re-signal (default: per-setup)")
        p.add_argument("--trail-ma", type=int, default=20, choices=[10, 20])
        p.add_argument("--partial-days", type=int, default=4)

    h = sub.add_parser("history", help="find past setups and what happened next")
    _common(h)
    _sweep_opts(h)
    h.add_argument("--charts", type=int, default=20, metavar="N")
    h.add_argument("--contact-sheet", action="store_true")
    h.set_defaults(func=cmd_history)

    b = sub.add_parser("backtest", help="replay the setups as one account: equity curve, drawdown, CAGR")
    _common(b)
    _sweep_opts(b)
    b.add_argument("--starting-equity", type=float, default=100_000.0)
    b.add_argument("--max-positions", type=int, default=10, help="concurrent open positions")
    b.add_argument("--max-exposure-pct", type=float, default=1.0, help="cap on total capital deployed")
    b.add_argument("--charts", type=int, default=0, metavar="N", help="chart the top N trades actually taken")
    b.set_defaults(func=cmd_backtest)

    def _fetch_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--workers", type=int, default=8, help="parallel fetches; lower it if rate-limited")
        p.add_argument("--pause", type=float, default=0.0, help="seconds to wait before each fetch")
        p.add_argument("--history-years", type=int, default=12, help="how far back to seed new symbols")
        p.add_argument("--force", action="store_true", help="also retry symbols benched for repeated failures")

    u = sub.add_parser("update", help="refresh the local price database (incremental)")
    _common(u)
    _fetch_opts(u)
    u.add_argument("--universe", default="sample")
    u.set_defaults(func=cmd_update)

    d = sub.add_parser("daily", help="update + filter + charts + HTML report — schedule this one")
    _common(d)
    _fetch_opts(d)
    d.add_argument("--universe", default="sample")
    d.add_argument("--charts", type=int, default=25, metavar="N")
    d.add_argument("--min-score", type=float, default=0.0)
    d.add_argument("--state", choices=["setup", "breakout"], default=None)
    d.add_argument("--skip-update", action="store_true", help="filter against the database as it stands")
    d.add_argument("--max-failure-ratio", type=float, default=0.5,
                   help="exit non-zero if more than this fraction of fetches fail (default 0.5)")
    d.add_argument("--max-stale-days", type=int, default=5,
                   help="exit non-zero if the newest bar is older than this (default 5)")
    d.set_defaults(func=cmd_daily)

    g = sub.add_parser("ingest", help="load vendor zips / folders / a long CSV into the price database")
    _common(g)
    g.add_argument("sources", nargs="+", metavar="PATH",
                   help="zip files, directories of CSVs, or one long-format CSV")
    g.add_argument("--no-merge", action="store_true",
                   help="let a later archive replace an earlier one instead of combining")
    g.set_defaults(func=cmd_ingest)

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
