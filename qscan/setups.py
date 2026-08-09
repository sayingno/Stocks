"""Registry that lets one CLI drive all three setups.

Each entry supplies: the presets, the per-bar evaluator, the trade direction,
and how long a signal stays "the same signal" for dedupe purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import setup_breakout, setup_ep, setup_parabolic
from .common import LONG, SHORT
from .config import EP_PRESETS, PARA_PRESETS, PRESETS
from .setup_breakout import _Arrays


@dataclass(frozen=True)
class Setup:
    name: str
    label: str
    presets: dict[str, Any]
    evaluate: Callable[..., dict[str, Any]]
    direction: int
    cooldown: int
    blurb: str

    def config(self, preset: str = "default"):
        if preset not in self.presets:
            raise ValueError(f"{self.name}: unknown preset {preset!r}; choose from {sorted(self.presets)}")
        return self.presets[preset]


REGISTRY: dict[str, Setup] = {
    "breakout": Setup(
        name="breakout",
        label="Breakout",
        presets=PRESETS,
        evaluate=setup_breakout.evaluate_bar,
        direction=LONG,
        cooldown=10,
        blurb="big prior move, orderly tightening base on the MAs, pressing the pivot",
    ),
    "ep": Setup(
        name="ep",
        label="Episodic pivot",
        presets=EP_PRESETS,
        evaluate=setup_ep.evaluate_bar,
        direction=LONG,
        cooldown=15,
        blurb="dead for months, then gaps 10%+ out of the base on huge volume",
    ),
    "parabolic": Setup(
        name="parabolic",
        label="Parabolic short",
        presets=PARA_PRESETS,
        evaluate=setup_parabolic.evaluate_bar,
        direction=SHORT,
        cooldown=10,
        blurb="vertical run, miles above the 20MA, first real crack — short side",
    ),
}

NAMES = tuple(REGISTRY)


def get(name: str) -> Setup:
    if name not in REGISTRY:
        raise ValueError(f"unknown setup {name!r}; choose from {list(NAMES)}")
    return REGISTRY[name]


def scan_frame(
    ann: pd.DataFrame,
    setup: Setup,
    cfg,
    symbol: str = "",
    positions=None,
    rs_series: pd.Series | None = None,
    rs_value: float | None = None,
    keep_rejects: bool = False,
) -> list[dict[str, Any]]:
    """Run any setup's detector over an already-annotated frame."""
    if len(ann) == 0:
        return []
    a = _Arrays.from_frame(ann)

    if rs_series is not None:
        rs = rs_series.reindex(ann.index).to_numpy(dtype=float)
    elif rs_value is not None:
        rs = np.full(len(ann), float(rs_value))
    else:
        rs = None

    idx = list(positions) if positions is not None else [len(a) - 1]
    hits: list[dict[str, Any]] = []
    for i in idx:
        if i < 0 or i >= len(a):
            continue
        res = setup.evaluate(a, i, cfg, rs=rs)
        if res["passed"] or keep_rejects:
            res["symbol"] = symbol
            res.setdefault("setup", setup.name)
            res.setdefault("date", pd.Timestamp(a.date[i]))
            res.setdefault("direction", setup.direction)
            res["bar"] = i
            hits.append(res)
    return hits
