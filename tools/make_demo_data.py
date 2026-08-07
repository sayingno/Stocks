"""Generate a synthetic CSV dataset so the CLI can be exercised without network.

    python tools/make_demo_data.py --out data/demo
    python -m qscan scan    --csv-dir data/demo --universe data/demo/universe.txt
    python -m qscan history --csv-dir data/demo --universe data/demo/universe.txt --charts 4

The names are fictional. This is for smoke-testing the plumbing, not for
drawing conclusions about the strategy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import synthetic  # noqa: E402

BUILDERS = {
    "GOODBRK": lambda: synthetic.winner_after_signal(synthetic.textbook_breakout(), run_pct=0.55),
    "COILED": synthetic.coiled_setup,
    "FAILBRK": lambda: synthetic.loser_after_signal(synthetic.textbook_breakout()),
    "DEEPBASE": synthetic.deep_sloppy_base,
    "DOWNTRND": synthetic.downtrend,
    "UNDER50": synthetic.below_slow_ma,
    "THINVOL": synthetic.illiquid_breakout,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, builder in BUILDERS.items():
        df = builder().copy()
        # Anchor the last bar to today so the default scan window covers it.
        df.index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(df))
        # A touch of noise so every file is not pathologically smooth.
        jitter = 1.0 + rng.normal(0, 0.0015, len(df))
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * jitter
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        df.index.name = "date"
        df.round(4).to_csv(out / f"{name}.csv")

    (out / "universe.txt").write_text("\n".join(BUILDERS) + "\n")
    print(f"wrote {len(BUILDERS)} symbols + universe.txt to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
