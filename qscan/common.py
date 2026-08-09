"""Pieces shared by all three setups.

The breakout, the episodic pivot and the parabolic short disagree about almost
everything — one wants a prior move, one wants dormancy; two want volume to dry
up, one wants it to explode; two are long and one is short — but they agree on
what makes a stock tradable at all, and on the risk arithmetic.

Anything both long and short setups touch is written in terms of `direction`
(+1 long, -1 short) so there is one implementation rather than a mirrored copy.
"""

from __future__ import annotations

import math
from typing import Any

LONG, SHORT = 1, -1


def clamp01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def reject(reason: str, **fields: Any) -> dict[str, Any]:
    return {"passed": False, "reject": reason, **fields}


def tradable(close: float, dollar_vol: float, adr: float, cfg) -> str | None:
    """The liquidity floor every setup shares. Returns a reject reason or None."""
    if not math.isfinite(close) or not math.isfinite(adr):
        return "nan_inputs"
    if close < cfg.min_price:
        return "price"
    if not (dollar_vol >= cfg.min_dollar_volume):
        return "liquidity"
    if not (cfg.min_adr_pct <= adr <= cfg.max_adr_pct):
        return "adr"
    return None


def trade_plan(
    entry: float,
    raw_stop: float,
    adr_pct: float,
    direction: int,
    cfg,
) -> dict[str, Any]:
    """Entry, ADR-capped stop, size and targets — for either direction.

    Qullamaggie's rule is that the stop is the extreme of the day but never
    wider than 1x ADR. On a long that means the stop is pulled *up* to the ADR
    floor; on a short it is pulled *down* to the ADR ceiling. Same rule, mirrored.
    """
    adr = (adr_pct / 100.0) * cfg.max_stop_adr_mult

    if direction == LONG:
        stop = max(raw_stop, entry * (1.0 - adr))
        risk_ps = entry - stop
    else:
        stop = min(raw_stop, entry * (1.0 + adr))
        risk_ps = stop - entry

    if risk_ps <= 0 or not math.isfinite(risk_ps):
        return {"entry": entry, "stop": stop, "risk_pct": float("nan"), "shares": 0, "direction": direction}

    shares = int(
        min(
            (cfg.account_size * cfg.risk_pct) / risk_ps,
            (cfg.account_size * cfg.max_position_pct) / entry,
        )
    )
    return {
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "raw_stop": raw_stop,
        "stop_capped_by_adr": stop != raw_stop,
        "risk_pct": risk_ps / entry,
        "risk_in_adr": (risk_ps / entry) / (adr_pct / 100.0) if adr_pct > 0 else float("nan"),
        "shares": shares,
        "position_value": shares * entry,
        "target_2r": entry + direction * 2 * risk_ps,
        "target_3r": entry + direction * 3 * risk_ps,
    }


def band_score(value: float, best: float, worst: float) -> float:
    """1.0 at `best`, 0.0 at `worst`, linear between. Handles either ordering."""
    if not math.isfinite(value) or best == worst:
        return 0.0
    return clamp01((value - worst) / (best - worst))


def weighted(parts: dict[str, float], weights: dict[str, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return round(100.0 * sum(parts.get(k, 0.0) * w for k, w in weights.items()) / total, 1)
