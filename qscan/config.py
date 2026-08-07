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
    move_lookbacks=((21, 0.40), (63, 0.70), (126, 1.20)),
    max_base_len=40,
    max_base_depth=0.25,
    max_contraction_ratio=0.80,
    max_volume_dryup_ratio=0.75,
    max_dist_from_pivot=0.06,
    max_surf_distance=0.07,
)

PRESETS = {"default": DEFAULT, "relaxed": RELAXED, "strict": STRICT}
