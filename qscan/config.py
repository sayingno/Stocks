"""Tunable thresholds for the breakout scan.

Every number Qullamaggie states as a rule of thumb lives here so you can loosen
or tighten the scan without touching detection logic. The defaults are the
"textbook" reading of the setup; `RELAXED` and `STRICT` are starting points for
seeing more or fewer names.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any


@dataclass(frozen=True)
class BreakoutConfig:
    # ---- liquidity / tradability -------------------------------------------
    min_price: float = 5.0
    min_dollar_volume: float = 3_000_000.0  # 20d average close*volume
    min_adr_pct: float = 3.5  # 20d average daily range, in percent
    max_adr_pct: float = 25.0  # above this the thing is a lottery ticket

    # ---- the prior move ("big move higher in the past 1-3 months") ---------
    # A name qualifies if ANY of these lookbacks cleared its threshold.
    move_lookbacks: tuple[tuple[int, float], ...] = (
        (21, 0.30),  # +30% in ~1 month
        (63, 0.50),  # +50% in ~3 months
        (126, 1.00),  # +100% in ~6 months
    )
    # The impulse must be reasonably recent: the leg's high has to sit within
    # this many bars of today, otherwise the move is stale.
    max_bars_since_impulse: int = 60

    # ---- relative strength --------------------------------------------------
    # The absolute thresholds above ask "did it move?"; this asks "did it move
    # more than everything else?", which is what picks leaders out of a bull
    # tape where everything is up. Percentile 0-100 against the universe on the
    # same date; None disables the gate.
    min_rs_rank: float | None = None
    # Weights on the 1m / 3m / 6m percentile ranks that make the composite.
    rs_weights: tuple[float, float, float] = (0.4, 0.3, 0.3)

    # ---- the consolidation --------------------------------------------------
    min_base_len: int = 3  # 3 days (tight flag) ...
    max_base_len: int = 60  # ... up to ~3 months
    max_base_depth: float = 0.35  # base high -> base low drawdown
    ideal_base_depth: float = 0.15  # scores full marks at or below this
    # Second half of the base must be tighter than the first half.
    max_contraction_ratio: float = 0.95
    # Volume in the back half of the base vs volume during the impulse leg.
    max_volume_dryup_ratio: float = 1.00
    # How close to the pivot price must be to count as "coiled and ready".
    max_dist_from_pivot: float = 0.10

    # ---- moving average structure ------------------------------------------
    ma_fast: int = 10
    ma_mid: int = 20
    ma_slow: int = 50
    use_ema: bool = False
    require_above_slow_ma: bool = True  # "never buy below the 50MA"
    require_ma_stack: bool = True  # fast > mid > slow
    require_rising_mid_ma: bool = True
    # Mean |close - MA| / close across the base, for the MA it hugged closest.
    max_surf_distance: float = 0.12

    # ---- breakout trigger ---------------------------------------------------
    breakout_volume_mult: float = 1.0  # today's vol vs 20d avg
    # Stop must not be wider than this many ADRs (Qullamaggie: 1x).
    max_stop_adr_mult: float = 1.0

    # ---- risk ---------------------------------------------------------------
    account_size: float = 100_000.0
    risk_pct: float = 0.005  # 0.5% of equity per trade
    max_position_pct: float = 0.20  # cap any single name at 20% of equity

    def with_overrides(self, **kwargs: Any) -> "BreakoutConfig":
        unknown = set(kwargs) - set(asdict(self))
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT = BreakoutConfig()

RELAXED = BreakoutConfig(
    min_dollar_volume=1_000_000.0,
    min_adr_pct=2.5,
    min_rs_rank=None,
    move_lookbacks=((21, 0.20), (63, 0.35), (126, 0.60)),
    max_base_depth=0.45,
    max_contraction_ratio=1.15,
    max_volume_dryup_ratio=1.30,
    max_dist_from_pivot=0.18,
    require_ma_stack=False,
    max_surf_distance=0.20,
)

STRICT = BreakoutConfig(
    min_dollar_volume=20_000_000.0,
    min_adr_pct=5.0,
    min_rs_rank=90.0,  # top decile of the universe
    move_lookbacks=((21, 0.40), (63, 0.70), (126, 1.20)),
    max_base_len=40,
    max_base_depth=0.25,
    max_contraction_ratio=0.80,
    max_volume_dryup_ratio=0.75,
    max_dist_from_pivot=0.06,
    max_surf_distance=0.07,
)

PRESETS = {"default": DEFAULT, "relaxed": RELAXED, "strict": STRICT}


# ---------------------------------------------------------------------------
# Setup #2 — Episodic Pivot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EPConfig:
    # tradability (dollar volume is measured on the 20 days *before* the gap)
    min_price: float = 5.0
    min_dollar_volume: float = 1_000_000.0
    min_adr_pct: float = 3.0
    max_adr_pct: float = 30.0

    # the gap
    min_gap: float = 0.10  # open vs prior close
    min_volume_mult: float = 2.0  # vs the prior 20-day average

    # dormancy — the part people skip, and the part that matters
    dormancy_lookback: int = 126  # ~6 months before the gap
    max_prior_gain: float = 0.50  # it must not already have run
    max_dormant_band: float = 0.60  # high-to-low range of that stretch

    # Two-sided bounds on the run-up into the event, per horizon. (min, max),
    # either side None to leave it open; None disables the horizon entirely.
    # Each asks a different question: 3d is "was it already moving into the
    # print?", 3m is "was it genuinely dormant?".
    perf_3d: tuple[float | None, float | None] | None = None
    perf_1w: tuple[float | None, float | None] | None = None
    perf_1m: tuple[float | None, float | None] | None = None
    perf_3m: tuple[float | None, float | None] | None = None

    # holding the gap
    min_close_position: float = 0.50  # close in the upper half of the day
    require_close_above_open: bool = True

    ma_fast: int = 10
    ma_mid: int = 20
    ma_slow: int = 50
    use_ema: bool = False

    min_rs_rank: float | None = None
    rs_weights: tuple[float, float, float] = (0.4, 0.3, 0.3)

    max_stop_adr_mult: float = 1.0
    account_size: float = 100_000.0
    risk_pct: float = 0.005
    max_position_pct: float = 0.20

    def with_overrides(self, **kwargs: Any) -> "EPConfig":
        unknown = set(kwargs) - set(asdict(self))
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EP_DEFAULT = EPConfig()
EP_RELAXED = EPConfig(
    min_dollar_volume=300_000.0, min_adr_pct=2.5, min_gap=0.06, min_volume_mult=1.5,
    max_prior_gain=1.00, max_dormant_band=0.80, min_close_position=0.35,
    require_close_above_open=False,
)
EP_STRICT = EPConfig(
    min_dollar_volume=10_000_000.0, min_adr_pct=4.0, min_gap=0.15, min_volume_mult=4.0,
    max_prior_gain=0.25, max_dormant_band=0.45, min_close_position=0.65,
)

EP_PRESETS = {"default": EP_DEFAULT, "relaxed": EP_RELAXED, "strict": EP_STRICT}


# ---------------------------------------------------------------------------
# Setup #3 — Parabolic short
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParabolicConfig:
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0
    min_adr_pct: float = 5.0  # it has to move, or the short cannot pay
    max_adr_pct: float = 60.0

    # the run
    run_lookback: int = 25
    min_run_gain: float = 0.80  # low to high inside the lookback
    max_run_bars: int = 25  # vertical, not a grind

    min_extension: float = 0.30  # close vs the 20MA
    min_volume_mult: float = 1.5  # run volume vs the prior 20-day average

    # the crack
    max_close_position: float = 0.40  # a reversal bar closes near its low
    min_green_streak: int = 3  # for the red-after-green trigger
    max_days_off_high: int = 3  # short the break, not the third day down

    ma_fast: int = 10
    ma_mid: int = 20
    ma_slow: int = 50
    use_ema: bool = False

    min_rs_rank: float | None = None
    rs_weights: tuple[float, float, float] = (0.4, 0.3, 0.3)

    max_stop_adr_mult: float = 1.0
    account_size: float = 100_000.0
    risk_pct: float = 0.005
    max_position_pct: float = 0.10  # half the long cap: losses are unbounded

    def with_overrides(self, **kwargs: Any) -> "ParabolicConfig":
        unknown = set(kwargs) - set(asdict(self))
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PARA_DEFAULT = ParabolicConfig()
PARA_RELAXED = ParabolicConfig(
    min_dollar_volume=2_000_000.0, min_adr_pct=4.0, min_run_gain=0.50,
    min_extension=0.20, max_close_position=0.55, max_days_off_high=5,
)
PARA_STRICT = ParabolicConfig(
    min_dollar_volume=20_000_000.0, min_adr_pct=8.0, min_run_gain=1.50,
    min_extension=0.50, max_close_position=0.30, max_days_off_high=2,
)

PARA_PRESETS = {"default": PARA_DEFAULT, "relaxed": PARA_RELAXED, "strict": PARA_STRICT}
