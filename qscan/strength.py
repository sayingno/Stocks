"""Cross-sectional relative strength.

The prior-move gate in the detector is absolute: "+30% in a month". In a tape
where everything is up 30%, that filter stops discriminating. Relative strength
asks the other question — *did it move more than everything else on that day* —
which is what actually isolates leaders.

The composite is IBD-style: rank the universe on each of the 1-, 3- and 6-month
returns separately, weight the three ranks, then re-rank the blend so the output
is a clean 0-100 percentile per (date, symbol).

Ranking is done per date across whatever symbols have data that day, so a
symbol that IPO'd mid-sample simply joins the ranking when it appears — no
lookahead, and no penalty for a short history beyond the return windows
themselves being NaN.
"""

from __future__ import annotations

import pandas as pd

RS_HORIZONS = (("ret_1m", 21), ("ret_3m", 63), ("ret_6m", 126))


def composite_rank(
    returns_by_horizon: dict[str, pd.DataFrame],
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> pd.DataFrame:
    """Blend per-horizon return panels into one 0-100 percentile panel.

    Each input is a wide frame indexed by date with one column per symbol.
    """
    if not returns_by_horizon:
        return pd.DataFrame()

    weighted_sum: pd.DataFrame | None = None
    weight_total: pd.DataFrame | None = None

    for (name, _bars), weight in zip(RS_HORIZONS, weights):
        panel = returns_by_horizon.get(name)
        if panel is None or panel.empty:
            continue
        # Percentile within each date, ignoring symbols with no value yet.
        ranks = panel.rank(axis=1, pct=True) * 100.0
        contribution = ranks * weight
        present = ranks.notna() * weight
        weighted_sum = contribution if weighted_sum is None else weighted_sum.add(contribution, fill_value=0.0)
        weight_total = present if weight_total is None else weight_total.add(present, fill_value=0.0)

    if weighted_sum is None or weight_total is None:
        return pd.DataFrame()

    # Renormalise so a symbol missing one horizon is scored on the others
    # rather than silently penalised.
    blended = weighted_sum / weight_total.replace(0.0, pd.NA)
    return blended.rank(axis=1, pct=True) * 100.0


def build_panel(frames: dict[str, pd.DataFrame], weights=(0.4, 0.3, 0.3)) -> pd.DataFrame:
    """Build the RS percentile panel from annotated per-symbol frames.

    `frames` maps symbol -> the frame produced by indicators.annotate.
    Returns a date x symbol frame of percentiles, or an empty frame if the
    universe is too small for a ranking to mean anything.
    """
    if len(frames) < 2:
        return pd.DataFrame()

    panels: dict[str, pd.DataFrame] = {}
    for name, _bars in RS_HORIZONS:
        series = {sym: df[name] for sym, df in frames.items() if name in df}
        if series:
            panels[name] = pd.DataFrame(series)
    return composite_rank(panels, weights)


def latest_ranks(
    returns: dict[str, tuple[float, float, float]],
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> dict[str, float]:
    """Rank the universe on a single date.

    The daily scan only needs today's percentile, so this takes each symbol's
    (1m, 3m, 6m) returns on the latest bar rather than building a whole panel —
    far cheaper than `build_panel` over thousands of symbols.
    """
    if len(returns) < 2:
        return {}
    frame = pd.DataFrame.from_dict(returns, orient="index", columns=["ret_1m", "ret_3m", "ret_6m"])
    panels = {name: frame[[name]].T for name in ("ret_1m", "ret_3m", "ret_6m")}
    blended = composite_rank(panels, weights)
    if blended.empty:
        return {}
    return {sym: float(v) for sym, v in blended.iloc[0].items() if pd.notna(v)}


def series_for(panel: pd.DataFrame, symbol: str) -> pd.Series | None:
    """Pull one symbol's percentile series out of the panel."""
    if panel.empty or symbol not in panel.columns:
        return None
    return panel[symbol]
