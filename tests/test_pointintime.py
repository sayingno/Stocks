"""Point-in-time correctness: a signal on date T must use only data up to T.

This is the test that decides whether the scanner can be trusted on a historical
cutoff. Everything else is plumbing; a lookahead bug silently invents an edge
that would not have existed in real time.

The method is simple and hard to fool: scan a bar with the *full* history
loaded, then truncate the data at that bar and scan it again. If any indicator,
gate or trade plan peeks forward, the two runs disagree.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qscan import setups, strength  # noqa: E402
from qscan.indicators import annotate  # noqa: E402
from tests import synthetic  # noqa: E402

# Fields produced by forward-looking evaluation, not by the detector.
OUTCOME_FIELDS = {
    "outcome", "fill", "fill_date", "exit_date", "exit_price", "r_multiple",
    "mfe_r", "mae_r", "days_held", "sim_stop", "chart",
    "fwd_5d", "fwd_10d", "fwd_20d", "fwd_60d", "mfe_20d", "mae_20d",
}


def signal_fields(hit: dict) -> dict:
    """Strip evaluation-only fields and round floats so comparison is exact."""
    out = {}
    for k, v in hit.items():
        if k in OUTCOME_FIELDS or k == "bar":
            continue
        if isinstance(v, float):
            out[k] = None if not np.isfinite(v) else round(v, 9)
        else:
            out[k] = v
    return out


class TestNoLookahead(unittest.TestCase):
    """Truncating the future must not change the signal."""

    def _compare(self, df: pd.DataFrame, setup_name: str, preset: str = "default", stride: int = 7):
        setup = setups.get(setup_name)
        cfg = setup.config(preset)
        ann_full = annotate(df, cfg)

        warmup = max(cfg.ma_slow, 126, getattr(cfg, "max_base_len", 0) + 5,
                     getattr(cfg, "dormancy_lookback", 0) + 25,
                     getattr(cfg, "run_lookback", 0) + 25)

        compared = 0
        for i in range(warmup, len(ann_full), stride):
            date = ann_full.index[i]

            full_hits = setups.scan_frame(ann_full, setup, cfg, symbol="T", positions=[i], keep_rejects=True)
            # Rebuild from raw OHLCV truncated at this bar — as if today were T.
            truncated = df.loc[:date]
            ann_pit = annotate(truncated, cfg)
            j = len(ann_pit) - 1
            pit_hits = setups.scan_frame(ann_pit, setup, cfg, symbol="T", positions=[j], keep_rejects=True)

            self.assertEqual(len(full_hits), len(pit_hits))
            if not full_hits:
                continue
            a, b = signal_fields(full_hits[0]), signal_fields(pit_hits[0])
            self.assertEqual(
                a, b,
                f"{setup_name} disagrees at {date.date()}: full-history scan differs "
                f"from the point-in-time scan — something reads forward",
            )
            compared += 1
        self.assertGreater(compared, 20, "not enough bars compared to be meaningful")

    def test_breakout_is_point_in_time(self):
        self._compare(synthetic.multi_cycle(cycles=6, seed=3), "breakout")

    def test_ep_is_point_in_time(self):
        self._compare(synthetic.ep_cycles(cycles=4, seed=4), "ep")

    def test_parabolic_is_point_in_time(self):
        self._compare(synthetic.multi_cycle(cycles=6, seed=5), "parabolic", preset="relaxed")

    def test_passing_signals_are_reproduced_exactly(self):
        """Same check, but restricted to bars that actually produced a signal."""
        setup = setups.get("breakout")
        cfg = setup.config("relaxed")
        df = synthetic.multi_cycle(cycles=8, seed=11)
        ann = annotate(df, cfg)

        hits = setups.scan_frame(ann, setup, cfg, symbol="T", positions=range(160, len(ann)))
        self.assertGreater(len(hits), 5, "fixture produced too few signals to test")

        for hit in hits[:25]:
            date = hit["date"]
            ann_pit = annotate(df.loc[:date], cfg)
            pit = setups.scan_frame(ann_pit, setup, cfg, symbol="T", positions=[len(ann_pit) - 1])
            self.assertEqual(len(pit), 1, f"signal at {date.date()} vanishes without future data")
            self.assertEqual(signal_fields(hit), signal_fields(pit[0]))


class TestRelativeStrengthIsPointInTime(unittest.TestCase):
    """The RS panel is built across the whole universe — check it ranks per date."""

    def setUp(self):
        self.frames = {
            f"S{i}": annotate(synthetic.multi_cycle(cycles=5, seed=20 + i, start_price=10 + 3 * i),
                              setups.get("breakout").config("default"))
            for i in range(6)
        }

    def test_rank_on_a_date_ignores_later_bars(self):
        full = strength.build_panel(self.frames)
        cut = full.index[len(full) // 2]

        truncated = {s: f.loc[:cut] for s, f in self.frames.items()}
        pit = strength.build_panel(truncated)

        for sym in self.frames:
            a = full.loc[cut, sym]
            b = pit.loc[cut, sym]
            if pd.isna(a) and pd.isna(b):
                continue
            self.assertAlmostEqual(a, b, places=9, msg=f"{sym} RS on {cut.date()} used future data")

    def test_a_symbol_that_delists_midway_does_not_distort_earlier_ranks(self):
        full = strength.build_panel(self.frames)
        cut = full.index[len(full) // 2]

        shortened = dict(self.frames)
        shortened["S0"] = shortened["S0"].loc[:cut]
        after = strength.build_panel(shortened)

        early = full.index[len(full) // 4]
        for sym in ("S1", "S2"):
            self.assertAlmostEqual(
                full.loc[early, sym], after.loc[early, sym], places=9,
                msg="truncating one symbol changed another symbol's earlier rank",
            )


class TestCutoffDiscipline(unittest.TestCase):
    """No signal may be dated before the requested cutoff."""

    def test_no_signals_before_the_cutoff(self):
        setup = setups.get("breakout")
        cfg = setup.config("relaxed")
        df = synthetic.multi_cycle(cycles=10, seed=7, start="2015-01-02")
        ann = annotate(df, cfg)

        cutoff = pd.Timestamp("2018-01-01")
        first = max(126, int(ann.index.searchsorted(cutoff)))
        hits = setups.scan_frame(ann, setup, cfg, symbol="T", positions=range(first, len(ann)))

        self.assertGreater(len(hits), 0, "cutoff test needs at least one signal")
        for h in hits:
            self.assertGreaterEqual(pd.Timestamp(h["date"]), cutoff)

    def test_indicators_still_warm_from_before_the_cutoff(self):
        """Bars before the cutoff must still feed the moving averages."""
        setup = setups.get("breakout")
        cfg = setup.config("relaxed")
        df = synthetic.multi_cycle(cycles=10, seed=7, start="2015-01-02")
        cutoff = pd.Timestamp("2018-01-01")

        warm = annotate(df, cfg)  # full history loaded
        cold = annotate(df.loc[cutoff:], cfg)  # loaded only from the cutoff

        i = int(warm.index.searchsorted(cutoff))
        self.assertTrue(np.isfinite(warm["ma_slow"].iloc[i]), "50MA should be warm at the cutoff")
        self.assertTrue(np.isnan(cold["ma_slow"].iloc[0]), "cold start should have no 50MA yet")


if __name__ == "__main__":
    unittest.main()
