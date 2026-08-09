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
    ap.add_argument("--long-names", type=int, default=0, metavar="N",
                    help="also emit N multi-year symbols (LONG00…) for cutoff and backtest testing")
    ap.add_argument("--ep-names", type=int, default=0, metavar="N",
                    help="also emit N dormant-then-gap symbols (EPIC00…) for the episodic pivot")
    ap.add_argument("--long-start", default="2015-01-02")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    builders = dict(BUILDERS)
    for i in range(args.long_names):
        builders[f"LONG{i:02d}"] = (
            lambda i=i: synthetic.multi_cycle(cycles=14, seed=args.seed + i, start=args.long_start,
                                              start_price=float(12 + 6 * (i % 5)))
        )
    # A few names that go dormant for months at a time, so the episodic-pivot
    # detector has something to find. multi_cycle turns over too fast to qualify.
    for i in range(args.ep_names):
        builders[f"EPIC{i:02d}"] = (
            lambda i=i: synthetic.ep_cycles(cycles=7, seed=args.seed + 100 + i, start=args.long_start,
                                            start_price=float(15 + 7 * (i % 4)))
        )

    for name, builder in builders.items():
        df = builder().copy()
        if not name.startswith(("LONG", "EPIC")):
            # Anchor short fixtures to today so the default scan window covers them.
            df.index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(df))
        # A touch of noise so every file is not pathologically smooth.
        jitter = 1.0 + rng.normal(0, 0.0015, len(df))
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * jitter
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        df.index.name = "date"
        df.round(4).to_csv(out / f"{name}.csv")

    (out / "universe.txt").write_text("\n".join(builders) + "\n")
    print(f"wrote {len(builders)} symbols + universe.txt to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
