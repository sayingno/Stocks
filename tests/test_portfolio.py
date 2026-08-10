import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qscan.portfolio import PortfolioConfig, run  # noqa: E402


def signal(sym, fill_day, exit_day, r, score=70.0, fill=100.0, stop=95.0):
    return {
        "symbol": sym,
        "date": pd.Timestamp(f"2024-01-{fill_day:02d}"),
        "fill_date": pd.Timestamp(f"2024-01-{fill_day:02d}"),
        "exit_date": pd.Timestamp(f"2024-01-{exit_day:02d}"),
        "fill": fill,
        "stop": stop,
        "r_multiple": r,
        "score": score,
        "outcome": "ma_trail" if r > 0 else "stopped",
    }


BASE = PortfolioConfig(starting_equity=100_000.0, risk_pct=0.01, max_positions=10, max_position_pct=1.0)


class TestBasics(unittest.TestCase):
    def test_no_signals_reports_zero_trades(self):
        self.assertEqual(run([], BASE).stats["trades"], 0)

    def test_untriggered_signals_are_ignored(self):
        s = {"symbol": "A", "fill_date": None, "exit_date": None, "fill": None, "r_multiple": float("nan")}
        self.assertEqual(run([s], BASE).stats["trades"], 0)

    def test_single_winner_credits_the_expected_dollars(self):
        # $100k, 1% risk = $1,000 risked; $5 stop distance -> 200 shares; +2R = +$2,000.
        res = run([signal("A", 2, 9, 2.0)], BASE)
        trade = res.trades.iloc[0]
        self.assertEqual(trade["shares"], 200)
        self.assertAlmostEqual(trade["pnl"], 2000.0, places=2)
        self.assertAlmostEqual(res.stats["final_equity"], 102_000.0, places=2)

    def test_loss_debits_one_r(self):
        res = run([signal("A", 2, 9, -1.0)], BASE)
        self.assertAlmostEqual(res.stats["final_equity"], 99_000.0, places=2)


class TestConcurrency(unittest.TestCase):
    def test_position_cap_skips_the_lower_scores(self):
        sigs = [signal(f"S{i}", 2, 20, 1.0, score=50 + i) for i in range(6)]
        res = run(sigs, PortfolioConfig(starting_equity=100_000, risk_pct=0.01, max_positions=2, max_position_pct=1.0))
        self.assertEqual(res.stats["trades_taken"], 2)
        self.assertEqual(res.stats["skipped_no_slot"], 4)

    def test_highest_score_wins_the_slot(self):
        sigs = [signal("LOW", 2, 20, 1.0, score=10), signal("HIGH", 2, 20, 1.0, score=99)]
        res = run(sigs, PortfolioConfig(starting_equity=100_000, risk_pct=0.01, max_positions=1, max_position_pct=1.0))
        self.assertEqual(list(res.trades["symbol"]), ["HIGH"])

    def test_slot_frees_up_after_an_exit(self):
        sigs = [signal("FIRST", 2, 5, 1.0), signal("SECOND", 8, 12, 1.0)]
        res = run(sigs, PortfolioConfig(starting_equity=100_000, risk_pct=0.01, max_positions=1, max_position_pct=1.0))
        self.assertEqual(res.stats["trades_taken"], 2)
        self.assertEqual(res.stats["skipped_no_slot"], 0)

    def test_max_open_positions_is_tracked(self):
        sigs = [signal(f"S{i}", 2, 20, 1.0) for i in range(4)]
        res = run(sigs, BASE)
        self.assertEqual(res.stats["max_open_positions"], 4)


class TestSizing(unittest.TestCase):
    def test_risk_compounds_with_equity(self):
        first = signal("A", 2, 5, 4.0)
        second = signal("B", 8, 12, 1.0)
        res = run([first, second], BASE)
        # After +4R the account is $104k, so the second trade risks 1% of that.
        self.assertGreater(res.trades.iloc[1]["risk_dollars"], res.trades.iloc[0]["risk_dollars"])

    def test_position_cap_limits_share_count(self):
        cfg = PortfolioConfig(starting_equity=100_000, risk_pct=0.05, max_positions=5, max_position_pct=0.10)
        res = run([signal("A", 2, 9, 1.0)], cfg)
        trade = res.trades.iloc[0]
        self.assertLessEqual(trade["shares"] * trade["fill"], 100_000 * 0.10 + trade["fill"])

    def test_exposure_cap_blocks_extra_names(self):
        sigs = [signal(f"S{i}", 2, 20, 1.0) for i in range(8)]
        cfg = PortfolioConfig(
            starting_equity=100_000, risk_pct=0.02, max_positions=20,
            max_position_pct=0.30, max_exposure_pct=0.50,
        )
        res = run(sigs, cfg)
        peak_exposure = res.equity["exposure"].max()
        self.assertLessEqual(peak_exposure, 100_000 * 0.50 + 200)
        self.assertGreater(res.stats["skipped_no_capital"], 0)

    def test_shares_are_whole_by_default(self):
        res = run([signal("A", 2, 9, 1.0, fill=333.33, stop=320.0)], BASE)
        self.assertEqual(res.trades.iloc[0]["shares"] % 1, 0)


class TestStats(unittest.TestCase):
    def setUp(self):
        self.res = run(
            [signal("W1", 2, 6, 3.0), signal("L1", 3, 4, -1.0),
             signal("W2", 8, 15, 2.0), signal("L2", 9, 10, -1.0)],
            BASE,
        )

    def test_win_rate_and_averages(self):
        s = self.res.stats
        self.assertEqual(s["trades_taken"], 4)
        self.assertAlmostEqual(s["win_rate"], 0.5)
        self.assertAlmostEqual(s["avg_win_r"], 2.5)
        self.assertAlmostEqual(s["avg_loss_r"], -1.0)

    def test_profit_factor_is_positive_for_a_profitable_run(self):
        self.assertGreater(self.res.stats["profit_factor"], 1.0)

    def test_equity_curve_has_a_row_per_event_date(self):
        self.assertFalse(self.res.equity.empty)
        self.assertIn("drawdown", self.res.equity.columns)
        self.assertLessEqual(self.res.equity["drawdown"].max(), 0.0)

    def test_drawdown_is_negative_when_equity_dips(self):
        res = run([signal("L", 2, 5, -1.0), signal("W", 8, 12, 5.0)], BASE)
        self.assertLess(res.stats["max_drawdown"], 0.0)


if __name__ == "__main__":
    unittest.main()
