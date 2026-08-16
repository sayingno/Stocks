"""Tests for setup #2 (episodic pivot) and setup #3 (parabolic short)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qscan import setups  # noqa: E402
from qscan.common import LONG, SHORT, trade_plan  # noqa: E402
from qscan.config import EP_DEFAULT, EP_STRICT, PARA_DEFAULT, PARA_STRICT  # noqa: E402
from qscan.indicators import annotate  # noqa: E402
from qscan.setup_breakout import _Arrays  # noqa: E402
from qscan.setup_ep import evaluate_bar as ep_eval  # noqa: E402
from qscan.setup_parabolic import evaluate_bar as para_eval  # noqa: E402
from tests import synthetic  # noqa: E402


def last(df, evaluator, cfg):
    ann = annotate(df, cfg)
    a = _Arrays.from_frame(ann)
    return evaluator(a, len(ann) - 1, cfg)


class TestEpisodicPivot(unittest.TestCase):
    def test_textbook_ep_passes(self):
        res = last(synthetic.episodic_pivot(), ep_eval, EP_DEFAULT)
        self.assertTrue(res["passed"], f"rejected at {res.get('reject')}")
        self.assertEqual(res["state"], "ep")
        self.assertEqual(res["setup"], "ep")

    def test_measured_features_are_sane(self):
        res = last(synthetic.episodic_pivot(gap=0.22, vol_mult=8.0), ep_eval, EP_DEFAULT)
        self.assertGreater(res["gap"], 0.20)
        self.assertGreater(res["vol_mult"], 5.0)
        self.assertLess(res["prior_gain"], 0.10)  # it was dead beforehand
        self.assertGreater(res["score"], 50)

    def test_small_gap_rejected(self):
        res = last(synthetic.episodic_pivot(gap=0.03), ep_eval, EP_DEFAULT)
        self.assertFalse(res["passed"])
        self.assertEqual(res["reject"], "gap_too_small")

    def test_gap_without_volume_rejected(self):
        res = last(synthetic.episodic_pivot(gap=0.20, vol_mult=1.1), ep_eval, EP_DEFAULT)
        self.assertFalse(res["passed"])
        self.assertEqual(res["reject"], "no_volume_surge")

    def test_gap_after_a_big_run_is_not_an_ep(self):
        """The dormancy rule is the whole point — a gap on an extended name is a blow-off."""
        res = last(synthetic.ep_after_a_big_run(), ep_eval, EP_STRICT)
        self.assertFalse(res["passed"])
        self.assertIn(res["reject"], {"already_extended", "not_dormant"})

    def test_breakout_fixture_is_not_an_ep(self):
        res = last(synthetic.textbook_breakout(), ep_eval, EP_DEFAULT)
        self.assertFalse(res["passed"])

    def test_liquidity_uses_pre_gap_volume(self):
        """A one-day volume spike must not sneak an untradable name through."""
        df = synthetic.episodic_pivot(vol_mult=400.0)
        df.loc[df.index[:-1], "volume"] = 2_000.0  # ~$40k/day before the gap
        res = last(df, ep_eval, EP_DEFAULT)
        self.assertFalse(res["passed"])
        self.assertEqual(res["reject"], "liquidity")

    def test_weak_close_rejected(self):
        df = synthetic.episodic_pivot()
        i = df.index[-1]
        df.loc[i, "close"] = df.loc[i, "low"] * 1.001  # gave the whole gap back
        res = last(df, ep_eval, EP_DEFAULT)
        self.assertFalse(res["passed"])
        self.assertIn(res["reject"], {"closed_weak", "closed_red", "still_inside_base"})

    def test_stop_within_one_adr(self):
        res = last(synthetic.episodic_pivot(), ep_eval, EP_DEFAULT)
        self.assertTrue(res["passed"])
        self.assertLessEqual(res["risk_in_adr"], 1.0 + 1e-9)
        self.assertLess(res["stop"], res["entry"])

    def test_stop_uses_the_gap_days_range_not_the_dormant_adr(self):
        """Regression: capping at the dormant ADR stops you out on noise next bar.

        The stock spent months with a 2.8% range and then gapped with a much
        wider one. The stop must sit at the low of the day, not be squeezed to
        1x the stale ADR.
        """
        df = synthetic.episodic_pivot(gap=0.22)
        res = last(df, ep_eval, EP_DEFAULT)
        self.assertTrue(res["passed"], res.get("reject"))

        day_low = float(df["low"].iloc[-1])
        self.assertAlmostEqual(res["stop"], day_low, places=6)
        self.assertGreater(res["stop_adr_ref"], res["adr20"])
        self.assertFalse(res["stop_capped_by_adr"])


class TestParabolicShort(unittest.TestCase):
    def test_textbook_parabolic_passes(self):
        res = last(synthetic.parabolic_top(), para_eval, PARA_DEFAULT)
        self.assertTrue(res["passed"], f"rejected at {res.get('reject')}")
        self.assertEqual(res["state"], "short")
        self.assertEqual(res["direction"], SHORT)

    def test_still_going_up_does_not_signal(self):
        """Shorting strength is how accounts die — it must wait for the crack."""
        res = last(synthetic.parabolic_top(crack=False), para_eval, PARA_DEFAULT)
        self.assertFalse(res["passed"])
        self.assertEqual(res["reject"], "no_weakness_yet")

    def test_modest_run_is_not_parabolic(self):
        res = last(synthetic.parabolic_top(run_gain=0.15), para_eval, PARA_DEFAULT)
        self.assertFalse(res["passed"])
        self.assertIn(res["reject"], {"not_parabolic", "not_extended"})

    def test_measured_features_are_sane(self):
        res = last(synthetic.parabolic_top(run_gain=1.9), para_eval, PARA_DEFAULT)
        self.assertGreater(res["run_gain"], 1.0)
        self.assertGreater(res["extension"], 0.30)
        self.assertTrue(res["reversal_bar"] or res["broke_prior_low"] or res["red_after_green"])

    def test_short_stop_sits_above_entry(self):
        res = last(synthetic.parabolic_top(), para_eval, PARA_DEFAULT)
        self.assertTrue(res["passed"])
        self.assertGreater(res["stop"], res["entry"], "a short's stop must be above the entry")
        self.assertLess(res["target_2r"], res["entry"], "a short's target must be below the entry")
        self.assertLessEqual(res["risk_in_adr"], 1.0 + 1e-9)

    def test_position_cap_is_tighter_than_the_long_setups(self):
        self.assertLess(PARA_DEFAULT.max_position_pct, EP_DEFAULT.max_position_pct)

    def test_quiet_stock_never_signals(self):
        res = last(synthetic.downtrend(), para_eval, PARA_STRICT)
        self.assertFalse(res["passed"])


class TestTradePlanDirection(unittest.TestCase):
    def test_long_plan(self):
        plan = trade_plan(entry=100.0, raw_stop=97.0, adr_pct=5.0, direction=LONG, cfg=EP_DEFAULT)
        self.assertEqual(plan["stop"], 97.0)  # inside 1 ADR, so untouched
        self.assertAlmostEqual(plan["target_2r"], 106.0)

    def test_long_stop_pulled_up_to_the_adr_floor(self):
        plan = trade_plan(entry=100.0, raw_stop=80.0, adr_pct=5.0, direction=LONG, cfg=EP_DEFAULT)
        self.assertAlmostEqual(plan["stop"], 95.0)
        self.assertTrue(plan["stop_capped_by_adr"])

    def test_short_stop_pulled_down_to_the_adr_ceiling(self):
        plan = trade_plan(entry=100.0, raw_stop=130.0, adr_pct=6.0, direction=SHORT, cfg=PARA_DEFAULT)
        self.assertAlmostEqual(plan["stop"], 106.0)
        self.assertAlmostEqual(plan["target_2r"], 88.0)
        self.assertTrue(plan["stop_capped_by_adr"])

    def test_short_risk_is_positive(self):
        plan = trade_plan(entry=100.0, raw_stop=104.0, adr_pct=8.0, direction=SHORT, cfg=PARA_DEFAULT)
        self.assertGreater(plan["risk_pct"], 0)
        self.assertGreater(plan["shares"], 0)


class TestRegistry(unittest.TestCase):
    def test_all_three_registered(self):
        self.assertEqual(set(setups.NAMES), {"breakout", "ep", "parabolic"})

    def test_directions(self):
        self.assertEqual(setups.get("breakout").direction, LONG)
        self.assertEqual(setups.get("ep").direction, LONG)
        self.assertEqual(setups.get("parabolic").direction, SHORT)

    def test_unknown_setup_raises(self):
        with self.assertRaises(ValueError):
            setups.get("nope")

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            setups.get("ep").config("banana")

    def test_scan_frame_dispatches(self):
        setup = setups.get("ep")
        ann = annotate(synthetic.episodic_pivot(), EP_DEFAULT)
        hits = setups.scan_frame(ann, setup, EP_DEFAULT, symbol="TEST")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["symbol"], "TEST")
        self.assertEqual(hits[0]["setup"], "ep")
        self.assertEqual(hits[0]["direction"], LONG)

    def test_each_setup_rejects_the_others_fixture(self):
        """The three detectors must not be interchangeable."""
        cases = [
            ("breakout", synthetic.episodic_pivot()),
            ("ep", synthetic.parabolic_top()),
            ("parabolic", synthetic.episodic_pivot()),
        ]
        for name, df in cases:
            setup = setups.get(name)
            cfg = setup.config("default")
            ann = annotate(df, cfg)
            hits = setups.scan_frame(ann, setup, cfg, symbol="X")
            self.assertEqual(hits, [], f"{name} should not fire on the other setup's fixture")


if __name__ == "__main__":
    unittest.main()


class TestEpPerfWindows(unittest.TestCase):
    """Two-sided run-up bounds, one per horizon."""

    def setUp(self):
        self.df = synthetic.episodic_pivot()

    def run_cfg(self, **over):
        return last(self.df, ep_eval, EP_DEFAULT.with_overrides(**over))

    def test_unbounded_by_default(self):
        res = self.run_cfg()
        self.assertTrue(res["passed"], res.get("reject"))
        for h in ("3d", "1w", "1m", "3m"):
            self.assertIn(f"pre_ret_{h}", res)

    def test_upper_bound_rejects_a_name_that_already_ran(self):
        res = self.run_cfg(perf_3m=(None, -0.50))
        self.assertFalse(res["passed"])
        self.assertEqual(res["reject"], "perf_3m_too_high")

    def test_lower_bound_rejects_a_name_that_did_nothing(self):
        res = self.run_cfg(perf_3m=(0.50, None))
        self.assertFalse(res["passed"])
        self.assertEqual(res["reject"], "perf_3m_too_low")

    def test_range_that_contains_the_value_passes(self):
        self.assertTrue(self.run_cfg(perf_3m=(-0.20, 0.30))["passed"])

    def test_each_horizon_is_independent(self):
        res = self.run_cfg(perf_3d=(None, 0.50), perf_1w=(None, -0.90))
        self.assertEqual(res["reject"], "perf_1w_too_high")

    def test_run_up_excludes_the_event_bar(self):
        """The gap itself must never count as part of the run-up into it."""
        res = self.run_cfg()
        self.assertLess(abs(res["pre_ret_3d"]), 0.05,
                        "a 22% gap leaked into the 3-day run-up")
