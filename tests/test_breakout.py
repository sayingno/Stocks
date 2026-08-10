import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qscan.config import DEFAULT, RELAXED, STRICT  # noqa: E402
from qscan.indicators import annotate  # noqa: E402
from qscan.setup_breakout import _Arrays, dedupe_signals, evaluate_bar, scan_symbol  # noqa: E402
from tests import synthetic  # noqa: E402


def last_result(df, cfg=DEFAULT):
    ann = annotate(df, cfg)
    a = _Arrays.from_frame(ann)
    return evaluate_bar(a, len(a) - 1, cfg)


class TestPositiveCases(unittest.TestCase):
    def test_textbook_breakout_passes(self):
        res = last_result(synthetic.textbook_breakout())
        self.assertTrue(res["passed"], f"rejected at {res.get('reject')}")
        self.assertEqual(res["state"], "breakout")

    def test_coiled_base_is_a_setup_not_a_breakout(self):
        res = last_result(synthetic.coiled_setup())
        self.assertTrue(res["passed"], f"rejected at {res.get('reject')}")
        self.assertEqual(res["state"], "setup")

    def test_measured_features_are_sane(self):
        res = last_result(synthetic.textbook_breakout())
        self.assertGreater(res["impulse_gain"], 0.50)
        self.assertGreaterEqual(res["base_len"], 20)
        self.assertLessEqual(res["base_len"], 35)
        self.assertLess(res["base_depth"], 0.20)
        self.assertLess(res["contraction"], 1.0)  # range tightened
        self.assertLess(res["vol_dryup"], 1.0)  # volume dried up
        self.assertGreater(res["score"], 50)

    def test_scan_symbol_wraps_the_same_result(self):
        hits = scan_symbol(synthetic.textbook_breakout(), DEFAULT, symbol="TEST")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["symbol"], "TEST")


class TestRejections(unittest.TestCase):
    def test_downtrend_rejected(self):
        res = last_result(synthetic.downtrend())
        self.assertFalse(res["passed"])
        self.assertIn(res["reject"], {"no_prior_move", "below_slow_ma", "ma_not_stacked"})

    def test_deep_base_rejected(self):
        res = last_result(synthetic.deep_sloppy_base())
        self.assertFalse(res["passed"])

    def test_below_50ma_rejected(self):
        res = last_result(synthetic.below_slow_ma())
        self.assertFalse(res["passed"])
        self.assertIn(res["reject"], {"below_slow_ma", "ma_not_stacked", "mid_ma_not_rising", "no_prior_move"})

    def test_illiquid_rejected_on_liquidity(self):
        res = last_result(synthetic.illiquid_breakout())
        self.assertFalse(res["passed"])
        self.assertEqual(res["reject"], "liquidity")

    def test_warmup_bars_rejected(self):
        ann = annotate(synthetic.textbook_breakout(), DEFAULT)
        a = _Arrays.from_frame(ann)
        self.assertEqual(evaluate_bar(a, 10, DEFAULT)["reject"], "warmup")


class TestTradePlan(unittest.TestCase):
    def test_stop_is_never_wider_than_one_adr(self):
        for builder in (synthetic.textbook_breakout, synthetic.coiled_setup):
            res = last_result(builder())
            self.assertTrue(res["passed"])
            self.assertLessEqual(res["risk_in_adr"], 1.0 + 1e-9, builder.__name__)
            self.assertGreater(res["risk_pct"], 0.0)

    def test_stop_below_entry_and_targets_above(self):
        res = last_result(synthetic.textbook_breakout())
        self.assertLess(res["stop"], res["entry"])
        self.assertGreater(res["target_2r"], res["entry"])
        self.assertAlmostEqual(res["target_3r"] - res["entry"], 3 * (res["entry"] - res["stop"]), places=6)

    def test_position_respects_risk_budget_and_cap(self):
        cfg = DEFAULT.with_overrides(account_size=100_000.0, risk_pct=0.005, max_position_pct=0.20)
        res = last_result(synthetic.textbook_breakout(), cfg)
        risk_dollars = res["shares"] * (res["entry"] - res["stop"])
        self.assertLessEqual(risk_dollars, 100_000 * 0.005 + res["entry"])
        self.assertLessEqual(res["position_value"], 100_000 * 0.20 + res["entry"])

    def test_bigger_account_buys_proportionally_more(self):
        small = last_result(synthetic.textbook_breakout(), DEFAULT.with_overrides(account_size=50_000.0))
        big = last_result(synthetic.textbook_breakout(), DEFAULT.with_overrides(account_size=500_000.0))
        self.assertGreater(big["shares"], small["shares"])


class TestPresets(unittest.TestCase):
    def test_strict_is_a_subset_of_relaxed(self):
        df = synthetic.textbook_breakout()
        relaxed = last_result(df, RELAXED)
        self.assertTrue(relaxed["passed"], f"relaxed rejected at {relaxed.get('reject')}")
        # STRICT demands ADR >= 5% and $20M; the synthetic name is deliberately
        # below both, so it must be filtered out.
        strict = last_result(df, STRICT)
        self.assertFalse(strict["passed"])

    def test_config_rejects_unknown_field(self):
        with self.assertRaises(ValueError):
            DEFAULT.with_overrides(not_a_real_field=1)


class TestDedupe(unittest.TestCase):
    def test_clusters_collapse_to_first_bar(self):
        hits = [{"symbol": "A", "bar": b, "date": b} for b in (100, 101, 102, 130, 131)]
        out = dedupe_signals(hits, cooldown=10)
        self.assertEqual([h["bar"] for h in out], [100, 130])

    def test_symbols_are_independent(self):
        hits = [
            {"symbol": "A", "bar": 100, "date": 100},
            {"symbol": "B", "bar": 101, "date": 101},
        ]
        self.assertEqual(len(dedupe_signals(hits, cooldown=10)), 2)


class TestHistoricalSweep(unittest.TestCase):
    def test_sweep_finds_the_breakout_cluster(self):
        df = synthetic.winner_after_signal()
        ann = annotate(df, DEFAULT)
        a = _Arrays.from_frame(ann)
        hits = [
            dict(evaluate_bar(a, i, DEFAULT), symbol="TEST", bar=i)
            for i in range(130, len(a))
            if evaluate_bar(a, i, DEFAULT)["passed"]
        ]
        self.assertGreater(len(hits), 0)
        deduped = dedupe_signals(hits, cooldown=10)
        self.assertLessEqual(len(deduped), len(hits))


if __name__ == "__main__":
    unittest.main()
