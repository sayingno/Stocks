"""Annotated candlestick charts for a detected setup.

One PNG per signal: candles, the 10/20/50 MAs, the consolidation shaded in, the
pivot line, entry and stop levels, and - for historical signals - what the
stock did next.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

UP, DOWN = "#26a69a", "#ef5350"
BASE_FILL = "#ffd54f"
GRID = "#e6e6e6"


def _candles(ax, df: pd.DataFrame) -> None:
    x = mdates.date2num(df.index.to_pydatetime())
    width = 0.6 if len(x) < 2 else max(0.4, float(np.median(np.diff(x))) * 0.65)
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()  # noqa: E741
    c = df["close"].to_numpy()
    up = c >= o

    ax.vlines(x[up], l[up], h[up], color=UP, linewidth=0.8)
    ax.vlines(x[~up], l[~up], h[~up], color=DOWN, linewidth=0.8)
    for mask, color in ((up, UP), (~up, DOWN)):
        lows = np.minimum(o[mask], c[mask])
        heights = np.maximum(np.abs(c[mask] - o[mask]), 1e-9)
        for xi, lo, ht in zip(x[mask], lows, heights):
            ax.add_patch(Rectangle((xi - width / 2, lo), width, ht, facecolor=color, edgecolor=color, linewidth=0.4))


def plot_signal(
    ann: pd.DataFrame,
    signal: dict[str, Any],
    out_path: str | Path,
    lookback: int = 90,
    lookforward: int = 45,
    outcome: dict[str, Any] | None = None,
    title_extra: str = "",
) -> Path:
    """Render one signal. `ann` must be the annotated frame from indicators.annotate."""
    bar = int(signal["bar"])
    start = max(0, bar - lookback)
    stop_i = min(len(ann), bar + lookforward + 1)
    view = ann.iloc[start:stop_i]
    if view.empty:
        raise ValueError("empty plotting window")

    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.06}
    )

    _candles(ax, view)
    for col, color, label in (
        ("ma_fast", "#1e88e5", "10 MA"),
        ("ma_mid", "#8e24aa", "20 MA"),
        ("ma_slow", "#455a64", "50 MA"),
    ):
        if col in view:
            ax.plot(view.index, view[col], color=color, linewidth=1.2, label=label, alpha=0.9)

    # --- the consolidation ---------------------------------------------------
    pivot_date = signal.get("pivot_date")
    pivot = signal.get("pivot")
    sig_date = ann.index[bar]
    if pivot_date is not None and pivot is not None and pd.notna(pivot):
        lo = signal.get("base_low", view["low"].min())
        ax.add_patch(
            Rectangle(
                (mdates.date2num(pd.Timestamp(pivot_date).to_pydatetime()), lo),
                mdates.date2num(sig_date.to_pydatetime()) - mdates.date2num(pd.Timestamp(pivot_date).to_pydatetime()),
                pivot - lo,
                facecolor=BASE_FILL,
                alpha=0.20,
                edgecolor="#f9a825",
                linewidth=1.0,
                zorder=0,
            )
        )
        ax.axhline(pivot, color="#f57f17", linestyle="--", linewidth=1.1, label=f"pivot {pivot:,.2f}")

    # --- the trade -----------------------------------------------------------
    entry, stop_px = signal.get("entry"), signal.get("stop")
    if entry is not None and pd.notna(entry):
        ax.axhline(entry, color="#2e7d32", linewidth=1.0, alpha=0.85, label=f"entry {entry:,.2f}")
    if stop_px is not None and pd.notna(stop_px):
        ax.axhline(stop_px, color="#c62828", linewidth=1.0, alpha=0.85, label=f"stop {stop_px:,.2f}")
    ax.axvline(sig_date, color="#212121", linewidth=1.0, alpha=0.55)

    if outcome and outcome.get("fill_date") is not None and pd.notna(outcome.get("fill_date")):
        ax.scatter([outcome["fill_date"]], [outcome["fill"]], marker="^", s=90, color="#2e7d32", zorder=5, label="fill")
    if outcome and outcome.get("exit_date") is not None and pd.notna(outcome.get("exit_date")):
        ax.scatter(
            [outcome["exit_date"]], [outcome["exit_price"]], marker="v", s=90, color="#c62828", zorder=5, label="exit"
        )

    # --- volume --------------------------------------------------------------
    vol_colors = np.where(view["close"].to_numpy() >= view["open"].to_numpy(), UP, DOWN)
    axv.bar(view.index, view["volume"], color=vol_colors, width=0.7, alpha=0.65)
    if "vol_avg20" in view:
        axv.plot(view.index, view["vol_avg20"], color="#37474f", linewidth=1.0, label="20d avg vol")
    axv.set_ylabel("volume", fontsize=9)
    axv.grid(color=GRID, linewidth=0.6)
    axv.legend(loc="upper left", fontsize=7, frameon=False)

    bits = [
        f"{signal.get('symbol', '')} — {pd.Timestamp(sig_date).date()}",
        str(signal.get("state", "")).upper(),
        f"score {signal.get('score', float('nan')):.0f}",
    ]
    sub = (
        f"prior move {signal.get('impulse_gain', float('nan')):.0%} · "
        f"base {signal.get('base_len', 0)}d deep {signal.get('base_depth', float('nan')):.1%} · "
        f"ADR {signal.get('adr20', float('nan')):.1f}% · "
        f"surf {signal.get('surf_ma', '')} {signal.get('surf_dist', float('nan')):.1%} · "
        f"risk {signal.get('risk_pct', float('nan')):.1%}"
    )
    if outcome:
        sub += f"\noutcome: {outcome.get('outcome')} · {outcome.get('r_multiple', float('nan'))}R · held {outcome.get('days_held', 0)}d · MFE {outcome.get('mfe_r', float('nan'))}R"
    if title_extra:
        sub += f"\n{title_extra}"

    ax.set_title("  |  ".join(bits) + "\n" + sub, fontsize=10, loc="left")
    ax.grid(color=GRID, linewidth=0.6)
    ax.legend(loc="upper left", fontsize=7, ncol=3, frameon=False)
    ax.set_ylabel("price", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.autofmt_xdate(rotation=0, ha="center")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def contact_sheet(images: list[Path], out_path: str | Path, cols: int = 2) -> Path | None:
    """Stack rendered charts into one browsable PNG."""
    if not images:
        return None
    import matplotlib.image as mpimg

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 9, rows * 5.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, path in zip(axes, images):
        ax.imshow(mpimg.imread(path))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
