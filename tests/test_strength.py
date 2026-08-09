import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qscan.config import DEFAULT  # noqa: E402
from qscan.indicators import annotate  # noqa: E402
from qscan.setup_breakout import _Arrays, evaluate_bar, rs_array  # noqa: E402
from qscan.strength import build_panel, composite_rank, series_for  # noqa: E402
from tests import synthetic  # noqa: E402

DATES = pd.bdate_range("2024-01-01", periods=4)


def panel(values: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(values, index=DATES)


class TestCompositeRank(unittest.TestCase):
    def test_leader_ranks_top_laggard_bottom(self):
        p = {
            "ret_1m": panel({"LEAD": [0.5] * 4, "MID": [0.1] * 4, "LAG": [-0.2] * 4}),
            "ret_3m": panel({"LEAD": [0.9] * 4, "MID": [0.2] * 4, "LAG": [-0.3] * 4}),
            "ret_6m": panel({"LEAD": [1.5] * 4, "MID": [0.3] * 4, "LAG": [-0.1] * 4}),
        }
        out = composite_rank(p)
        self.assertAlmostEqual(out["LEAD"].iloc[-1], 100.0)
        self.assertAlmostEqual(out["LAG"].iloc[-1], 100 / 3, places=4)
        self.assertGreater(out["MID"].iloc[-1], out["LAG"].iloc[-1])

    def test_ranking_is_per_date_not_global(self):
        # A and B swap leadership halfway through.
        p = {
            "ret_1m": panel({"A": [0.9, 0.9, -0.1, -0.1], "B": [-0.1, -0.1, 0.9, 0.9]}),
        }
        out = composite_rank(p)
        self.assertGreater(out["A"].iloc[0], out["B"].iloc[0])
        self.assertLess(out["A"].iloc[-1], out["B"].iloc[-1])

    def test_weights_shift_the_blend(self):
        p = {
            "ret_1m": panel({"FAST": [0.9] * 4, "SLOW": [-0.1] * 4}),
            "ret_6m": panel({"FAST": [-0.1] * 4, "SLOW": [0.9] * 4}),
        }
        short_biased = composite_rank(p, weights=(0.9, 0.0, 0.1))
        long_biased = composite_rank(p, weights=(0.1, 0.0, 0.9))
        self.assertGreater(short_biased["FAST"].iloc[-1], short_biased["SLOW"].iloc[-1])
        self.assertLess(long_biased["FAST"].iloc[-1], long_biased["SLOW"].iloc[-1])

    def test_symbol_missing_a_horizon_is_not_penalised(self):
        # BOTH and ONLY1M have identical 1m strength; ONLY1M has no 6m data.
        p = {
            "ret_1m": panel({"BOTH": [0.5] * 4, "ONLY1M": [0.5] * 4, "WEAK": [-0.5] * 4}),
            "ret_6m": panel({"BOTH": [0.5] * 4, "ONLY1M": [np.nan] * 4, "WEAK": [-0.5] * 4}),
        }
        out = composite_rank(p)
        self.assertGreater(out["ONLY1M"].iloc[-1], out["WEAK"].iloc[-1])

    def test_empty_input_is_safe(self):
        self.assertTrue(composite_rank({}).empty)


class TestBuildPanel(unittest.TestCase):
    def test_single_symbol_universe_cannot_rank(self):
        frames = {"ONLY": annotate(synthetic.textbook_breakout(), DEFAULT)}
        self.assertTrue(build_panel(frames).empty)

    def test_panel_covers_every_symbol(self):
        frames = {
            "A": annotate(synthetic.textbook_breakout(), DEFAULT),
            "B": annotate(synthetic.downtrend(), DEFAULT),
            "C": annotate(synthetic.below_slow_ma(), DEFAULT),
        }
        p = build_panel(frames)
        self.assertEqual(set(p.columns), {"A", "B", "C"})
        self.assertTrue((p.max(axis=1).dropna() <= 100.0 + 1e-9).all())

    def test_uptrend_outranks_downtrend(self):
        frames = {
            "UP": annotate(synthetic.textbook_breakout(), DEFAULT),
            "DOWN": annotate(synthetic.downtrend(), DEFAULT),
        }
        p = build_panel(frames)
        # The fixtures end on different dates; compare on the last day both trade.
        shared = frames["UP"].index.intersection(frames["DOWN"].index)[-1]
        self.assertGreater(p.loc[shared, "UP"], p.loc[shared, "DOWN"])

    def test_symbol_absent_on_a_date_ranks_nan_rather_than_zero(self):
        """No data must not read as 'weakest' — the RS gate rejects NaN separately."""
        frames = {
            "SHORT": annotate(synthetic.textbook_breakout(), DEFAULT),
            "LONG": annotate(synthetic.downtrend(), DEFAULT),
        }
        p = build_panel(frames)
        beyond = frames["LONG"].index[-1]
        self.assertTrue(pd.isna(p.loc[beyond, "SHORT"]))

    def test_series_for_missing_symbol_is_none(self):
        self.assertIsNone(series_for(pd.DataFrame(), "X"))


class TestRsGate(unittest.TestCase):
    def setUp(self):
        self.ann = annotate(synthetic.textbook_breakout(), DEFAULT)
        self.arrays = _Arrays.from_frame(self.ann)
        self.last = len(self.ann) - 1

    def _rs(self, value: float):
        return rs_array(self.ann, pd.Series(value, index=self.ann.index))

    def test_strong_name_passes_the_gate(self):
        cfg = DEFAULT.with_overrides(min_rs_rank=80.0)
        res = evaluate_bar(self.arrays, self.last, cfg, rs=self._rs(95.0))
        self.assertTrue(res["passed"], res.get("reject"))
        self.assertEqual(res["rs_rank"], 95.0)

    def test_weak_name_is_rejected(self):
        cfg = DEFAULT.with_overrides(min_rs_rank=80.0)
        res = evaluate_bar(self.arrays, self.last, cfg, rs=self._rs(40.0))
        self.assertFalse(res["passed"])
        self.assertEqual(res["reject"], "weak_rs")

    def test_gate_off_by_default_ignores_rs(self):
        res = evaluate_bar(self.arrays, self.last, DEFAULT, rs=self._rs(1.0))
        self.assertTrue(res["passed"], res.get("reject"))

    def test_gate_on_without_data_rejects_rather_than_passing_silently(self):
        cfg = DEFAULT.with_overrides(min_rs_rank=80.0)
        res = evaluate_bar(self.arrays, self.last, cfg, rs=None)
        self.assertFalse(res["passed"])
        self.assertEqual(res["reject"], "no_rs_data")

    def test_nan_rs_is_treated_as_failing(self):
        cfg = DEFAULT.with_overrides(min_rs_rank=80.0)
        res = evaluate_bar(self.arrays, self.last, cfg, rs=self._rs(float("nan")))
        self.assertFalse(res["passed"])
        self.assertEqual(res["reject"], "weak_rs")

    def test_strict_preset_ships_with_the_gate_on(self):
        from qscan.config import STRICT

        self.assertIsNotNone(STRICT.min_rs_rank)


if __name__ == "__main__":
    unittest.main()
