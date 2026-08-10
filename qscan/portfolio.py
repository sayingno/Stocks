"""Portfolio-level backtest.

`outcomes.simulate_trade` answers "what would this one trade have returned, in
R?". That is the right unit for judging the setup, but it is not a backtest: it
assumes infinite capital, takes every signal, and never asks what happens when
forty names trigger on the same morning.

This module walks the calendar with a fixed account and real constraints:

* **Concurrency.** At most `max_positions` open at once. When more signals fire
  than there are slots, the highest-scoring ones win and the rest are recorded
  as skipped — that skip count is itself a useful diagnostic.
* **Compounding.** Risk is a percentage of *current* equity, so the position
  size grows and shrinks with the account.
* **Exposure.** A cap on total capital deployed, because a book of twenty names
  each sized at 20% of equity is not a thing you can actually hold.

Each signal's R multiple, fill date and exit date come from the per-trade
simulation, which is equity-independent; this layer decides which trades get
taken and how many dollars ride on each.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    starting_equity: float = 100_000.0
    risk_pct: float = 0.005  # fraction of equity risked per trade
    max_positions: int = 10
    max_position_pct: float = 0.20  # cap on any single name
    max_exposure_pct: float = 1.00  # cap on total capital deployed
    allow_fractional_shares: bool = False


@dataclass
class _Open:
    symbol: str
    fill_date: pd.Timestamp
    exit_date: pd.Timestamp
    fill: float
    shares: float
    risk_dollars: float
    r_multiple: float
    score: float
    direction: int = 1
    setup: str = ""

    @property
    def cost(self) -> float:
        # A short still consumes buying power, so exposure is the absolute notional.
        return self.shares * self.fill

    @property
    def pnl(self) -> float:
        return self.risk_dollars * self.r_multiple


@dataclass
class BacktestResult:
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: dict[str, Any] = field(default_factory=dict)


def _to_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return None if pd.isna(ts) else ts


def run(signals: list[dict[str, Any]], cfg: PortfolioConfig = PortfolioConfig()) -> BacktestResult:
    """Replay the signals as a single account.

    Each signal needs: symbol, fill_date, exit_date, fill, r_multiple, score.
    Signals without a fill (never triggered) are ignored.
    """
    tradable = []
    for s in signals:
        fill_date = _to_ts(s.get("fill_date"))
        exit_date = _to_ts(s.get("exit_date"))
        r = s.get("r_multiple")
        fill = s.get("fill")
        if fill_date is None or exit_date is None or fill is None:
            continue
        if r is None or (isinstance(r, float) and not math.isfinite(r)):
            continue
        if exit_date < fill_date:
            exit_date = fill_date
        tradable.append(
            {
                "symbol": s.get("symbol", ""),
                "signal_date": _to_ts(s.get("date")),
                "fill_date": fill_date,
                "exit_date": exit_date,
                "fill": float(fill),
                "r_multiple": float(r),
                "score": float(s.get("score", 0.0)),
                "stop": float(s.get("stop", 0.0)),
                "direction": int(s.get("direction", 1)),
                "setup": s.get("setup", ""),
                "outcome": s.get("outcome", ""),
            }
        )

    if not tradable:
        return BacktestResult(stats={"trades": 0, "signals": len(signals)})

    by_fill: dict[pd.Timestamp, list[dict]] = {}
    for t in tradable:
        by_fill.setdefault(t["fill_date"], []).append(t)

    calendar = sorted({d for t in tradable for d in (t["fill_date"], t["exit_date"])})

    equity = cfg.starting_equity
    peak = equity
    open_positions: list[_Open] = []
    closed: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    skipped_capacity = 0
    skipped_capital = 0

    for today in calendar:
        # ---- 1. close anything whose exit lands today ------------------------
        still_open: list[_Open] = []
        for pos in open_positions:
            if pos.exit_date <= today:
                equity += pos.pnl
                closed.append(
                    {
                        "symbol": pos.symbol,
                        "setup": pos.setup,
                        "direction": pos.direction,
                        "fill_date": pos.fill_date,
                        "exit_date": pos.exit_date,
                        "days_held": int(np.busday_count(pos.fill_date.date(), pos.exit_date.date())),
                        "shares": pos.shares,
                        "fill": pos.fill,
                        "risk_dollars": round(pos.risk_dollars, 2),
                        "r_multiple": pos.r_multiple,
                        "pnl": round(pos.pnl, 2),
                        "equity_after": round(equity, 2),
                        "score": pos.score,
                    }
                )
            else:
                still_open.append(pos)
        open_positions = still_open

        # ---- 2. open new positions, best score first -------------------------
        candidates = sorted(by_fill.get(today, []), key=lambda t: -t["score"])
        for cand in candidates:
            if len(open_positions) >= cfg.max_positions:
                skipped_capacity += 1
                continue

            deployed = sum(p.cost for p in open_positions)
            room = cfg.max_exposure_pct * equity - deployed
            if room <= 0:
                skipped_capital += 1
                continue

            risk_dollars = equity * cfg.risk_pct
            # Risk per share is entry-to-stop, whichever side the stop sits on.
            risk_ps = abs(cand["fill"] - cand["stop"])
            if risk_ps <= 0:
                # Fall back to an ADR-ish distance if the stored stop is unusable.
                risk_ps = cand["fill"] * 0.05

            shares = risk_dollars / risk_ps
            shares = min(shares, (cfg.max_position_pct * equity) / cand["fill"], room / cand["fill"])
            if not cfg.allow_fractional_shares:
                shares = float(int(shares))
            if shares <= 0:
                skipped_capital += 1
                continue

            open_positions.append(
                _Open(
                    symbol=cand["symbol"],
                    fill_date=today,
                    exit_date=cand["exit_date"],
                    fill=cand["fill"],
                    shares=shares,
                    risk_dollars=shares * risk_ps,
                    r_multiple=cand["r_multiple"],
                    score=cand["score"],
                    direction=cand["direction"],
                    setup=cand["setup"],
                )
            )

        peak = max(peak, equity)
        curve.append(
            {
                "date": today,
                "equity": round(equity, 2),
                "open_positions": len(open_positions),
                "exposure": round(sum(p.cost for p in open_positions), 2),
                "drawdown": round(equity / peak - 1.0, 5) if peak > 0 else 0.0,
            }
        )

    # Anything still open at the end is marked to its last known R.
    for pos in open_positions:
        equity += pos.pnl
        closed.append(
            {
                "symbol": pos.symbol,
                "setup": pos.setup,
                "direction": pos.direction,
                "fill_date": pos.fill_date,
                "exit_date": pos.exit_date,
                "days_held": int(np.busday_count(pos.fill_date.date(), pos.exit_date.date())),
                "shares": pos.shares,
                "fill": pos.fill,
                "risk_dollars": round(pos.risk_dollars, 2),
                "r_multiple": pos.r_multiple,
                "pnl": round(pos.pnl, 2),
                "equity_after": round(equity, 2),
                "score": pos.score,
            }
        )

    equity_df = pd.DataFrame(curve)
    trades_df = pd.DataFrame(closed).sort_values("fill_date").reset_index(drop=True) if closed else pd.DataFrame()

    return BacktestResult(
        equity=equity_df,
        trades=trades_df,
        stats=_stats(equity_df, trades_df, cfg, len(tradable), skipped_capacity, skipped_capital, len(signals)),
    )


def _stats(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    cfg: PortfolioConfig,
    tradable: int,
    skipped_capacity: int,
    skipped_capital: int,
    total_signals: int,
) -> dict[str, Any]:
    if equity.empty or trades.empty:
        return {"signals": total_signals, "trades": 0}

    final = float(trades["equity_after"].iloc[-1])
    total_return = final / cfg.starting_equity - 1.0
    start, end = equity["date"].iloc[0], equity["date"].iloc[-1]
    years = max((end - start).days / 365.25, 1e-9)
    cagr = (final / cfg.starting_equity) ** (1 / years) - 1.0 if final > 0 else -1.0

    r = trades["r_multiple"]
    wins, losses = r[r > 0], r[r <= 0]
    pnl_wins = trades.loc[r > 0, "pnl"].sum()
    pnl_losses = trades.loc[r <= 0, "pnl"].sum()

    return {
        "signals": total_signals,
        "tradable_signals": tradable,
        "trades_taken": int(len(trades)),
        "skipped_no_slot": skipped_capacity,
        "skipped_no_capital": skipped_capital,
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "years": round(years, 2),
        "starting_equity": cfg.starting_equity,
        "final_equity": round(final, 2),
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(float(equity["drawdown"].min()), 4),
        "win_rate": round(float(len(wins) / len(r)), 3),
        "avg_r": round(float(r.mean()), 3),
        "avg_win_r": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_r": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "profit_factor": round(float(pnl_wins / abs(pnl_losses)), 2) if pnl_losses < 0 else None,
        "best_trade": round(float(trades["pnl"].max()), 2),
        "worst_trade": round(float(trades["pnl"].min()), 2),
        "avg_days_held": round(float(trades["days_held"].mean()), 1),
        "avg_open_positions": round(float(equity["open_positions"].mean()), 2),
        "max_open_positions": int(equity["open_positions"].max()),
    }


def plot_equity(result: BacktestResult, out_path, title: str = "Portfolio backtest"):
    """Equity curve over a drawdown panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    if result.equity.empty:
        return None

    eq = result.equity
    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )
    ax.plot(eq["date"], eq["equity"], color="#0b6b5f", linewidth=1.6)
    ax.fill_between(eq["date"], eq["equity"].min(), eq["equity"], color="#0b6b5f", alpha=0.07)
    ax.set_ylabel("equity ($)", fontsize=9)
    ax.grid(color="#e6e6e6", linewidth=0.6)

    s = result.stats
    ax.set_title(
        f"{title}\n"
        f"{s.get('trades_taken', 0)} trades · {s.get('total_return', 0):.0%} total · "
        f"{s.get('cagr', 0):.1%} CAGR · {s.get('max_drawdown', 0):.1%} max DD · "
        f"win rate {s.get('win_rate', 0):.0%} · avg {s.get('avg_r', 0)}R",
        fontsize=10,
        loc="left",
    )

    axd.fill_between(eq["date"], eq["drawdown"] * 100, 0, color="#b3402f", alpha=0.35)
    axd.set_ylabel("drawdown (%)", fontsize=9)
    axd.grid(color="#e6e6e6", linewidth=0.6)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
