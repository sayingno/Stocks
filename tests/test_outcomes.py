import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qscan.config import DEFAULT  # noqa: E402
from qscan.indicators import annotate  # noqa: E402
from qscan.outcomes import TradeRules, forward_returns, simulate_trade, summarise  # noqa: E402
from qscan.setup_breakout import _Arrays, evaluate_bar  # noqa: E402
from tests import synthetic  # noqa: E402


def signal_on(df, bar=None, cfg=DEFAULT):
    ann = annotate(df, cfg)
    a = _Arrays.from_frame(ann)
    i = len(a) - 1 if bar is None else bar
    res = evaluate_bar(a, i, cfg)
    res["bar"] = i
    res["symbol"] = "TEST"
    return ann, res


class TestForwardReturns(unittest.TestCase):
    def test_winner_has_positive_forward_returns(self):
        base = synthetic.textbook_breakout()
        signal_bar = len(base) - 1
        df = synthetic.winner_after_signal(base)
        ann, _ = signal_on(df, bar=signal_bar)
        fwd = forward_returns(ann, signal_bar)
        self.assertGreater(fwd["fwd_5d"], 0)
        self.assertGreater(fwd["fwd_20d"], fwd["fwd_5d"])
        self.assertGreater(fwd["mfe_20d"], 0)

    def test_horizons_past_the_end_are_nan(self):
        df = synthetic.textbook_breakout()
        ann, _ = signal_on(df)
        fwd = forward_returns(ann, len(ann) - 1)
        self.assertTrue(np.isnan(fwd["fwd_5d"]))


class TestSimulateTrade(unittest.TestCase):
    def test_winner_returns_positive_r(self):
        base = synthetic.textbook_breakout()
        signal_bar = len(base) - 1
        ann, sig = signal_on(synthetic.winner_after_signal(base), bar=signal_bar)
        self.assertTrue(sig["passed"], sig.get("reject"))
        out = simulate_trade(ann, sig)
        self.assertGreater(out["r_multiple"], 1.0)
        self.assertGreater(out["mfe_r"], out["r_multiple"] * 0.5)
        self.assertGreater(out["days_held"], 0)

    def test_loser_is_capped_near_minus_one_r(self):
        base = synthetic.textbook_breakout()
        signal_bar = len(base) - 1
        ann, sig = signal_on(synthetic.loser_after_signal(base), bar=signal_bar)
        out = simulate_trade(ann, sig)
        self.assertEqual(out["outcome"], "stopped")
        # A gap through the stop can do slightly worse than -1R, but not wildly.
        self.assertLess(out["r_multiple"], 0.0)
        self.assertGreater(out["r_multiple"], -2.0)

    def test_untriggered_setup_reports_no_trigger(self):
        ann, sig = signal_on(synthetic.coiled_setup())
        sig["entry"] = float(ann["close"].iloc[-1]) * 5.0  # unreachable
        sig["state"] = "setup"
        out = simulate_trade(ann, sig, TradeRules(trigger_window=5))
        self.assertEqual(out["outcome"], "no_trigger")

    def test_partial_moves_stop_to_breakeven(self):
        """After the partial, a slow bleed should not lose a full R."""
        base = synthetic.textbook_breakout()
        signal_bar = len(base) - 1
        ann, sig = signal_on(synthetic.winner_after_signal(base, run_pct=0.20), bar=signal_bar)
        out = simulate_trade(ann, sig, TradeRules(partial_days=3, partial_fraction=0.5))
        self.assertGreater(out["r_multiple"], -0.6)


class TestSummarise(unittest.TestCase):
    def test_aggregates_the_expected_fields(self):
        rows = [
            {"outcome": "ma_trail", "r_multiple": 4.0},
            {"outcome": "stopped", "r_multiple": -1.0},
            {"outcome": "stopped", "r_multiple": -1.0},
            {"outcome": "no_trigger", "r_multiple": float("nan")},
        ]
        stats = summarise(rows)
        self.assertEqual(stats["signals"], 4)
        self.assertEqual(stats["trades"], 3)
        self.assertEqual(stats["no_trigger"], 1)
        self.assertAlmostEqual(stats["win_rate"], 1 / 3, places=3)
        self.assertAlmostEqual(stats["total_r"], 2.0, places=6)
        self.assertAlmostEqual(stats["avg_loss_r"], -1.0, places=6)

    def test_empty_input_is_safe(self):
        self.assertEqual(summarise([{"outcome": "no_trigger", "r_multiple": float("nan")}])["trades"], 0)


if __name__ == "__main__":
    unittest.main()
