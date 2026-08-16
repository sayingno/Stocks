"""Tests for the dataset inventory."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qscan import summary as summary_mod  # noqa: E402
from tests import synthetic  # noqa: E402


def universe():
    return {
        "LONGHIST": synthetic.multi_cycle(cycles=6, seed=1),
        "ALSOLONG": synthetic.multi_cycle(cycles=6, seed=2, start_price=40.0),
        "SHORTHIST": synthetic.textbook_breakout().head(60),
    }


class TestDescribeSymbol(unittest.TestCase):
    def test_reports_span_and_bar_count(self):
        df = synthetic.textbook_breakout()
        row = summary_mod.describe_symbol("T", df)
        self.assertEqual(row["bars"], len(df))
        self.assertEqual(row["first"], df.index[0].date())
        self.assertEqual(row["last"], df.index[-1].date())

    def test_empty_frame_is_safe(self):
        self.assertEqual(summary_mod.describe_symbol("T", pd.DataFrame())["bars"], 0)

    def test_counts_zero_volume_days(self):
        df = synthetic.textbook_breakout().copy()
        df.iloc[3, df.columns.get_loc("volume")] = 0.0
        self.assertEqual(summary_mod.describe_symbol("T", df)["zero_volume_days"], 1)

    def test_flags_an_unadjusted_two_for_one_split(self):
        """The most common split ratio lands on exactly a 50% drop.

        Regression: the threshold used to sit at 60%, so the canonical case
        walked straight through the check that exists to catch it.
        """
        df = synthetic.textbook_breakout().copy()
        half = len(df) // 2
        for col in ("open", "high", "low", "close"):
            df.iloc[:half, df.columns.get_loc(col)] *= 2.0
        self.assertGreater(summary_mod.describe_symbol("T", df)["suspect_jumps"], 0)

    def test_flags_a_ten_for_one_split(self):
        df = synthetic.textbook_breakout().copy()
        half = len(df) // 2
        for col in ("open", "high", "low", "close"):
            df.iloc[:half, df.columns.get_loc(col)] *= 10.0
        self.assertGreater(summary_mod.describe_symbol("T", df)["suspect_jumps"], 0)

    def test_clean_series_has_no_suspect_jumps(self):
        self.assertEqual(summary_mod.describe_symbol("T", synthetic.textbook_breakout())["suspect_jumps"], 0)


class TestSummarise(unittest.TestCase):
    def setUp(self):
        self.res = summary_mod.summarise(universe(), min_bars=200)

    def test_counts_symbols_and_usable_symbols(self):
        self.assertEqual(self.res.overview["symbols"], 3)
        self.assertEqual(self.res.overview["usable_symbols"], 2)

    def test_reports_the_overall_date_range(self):
        lo, hi = self.res.overview["date_range"]
        self.assertLess(pd.Timestamp(lo), pd.Timestamp(hi))

    def test_warns_about_symbols_too_short_to_scan(self):
        self.assertTrue(any("under 200 bars" in w for w in self.res.warnings))

    def test_distributions_are_reported_as_quantiles(self):
        for key in ("last_close", "median_dollar_vol", "median_adr_pct"):
            self.assertIn("p50", self.res.overview[key])

    def test_empty_dataset_is_safe(self):
        self.assertEqual(summary_mod.summarise({}).overview["symbols"], 0)

    def test_per_symbol_table_has_a_row_each(self):
        self.assertEqual(len(self.res.per_symbol), 3)
        self.assertEqual(set(self.res.per_symbol["symbol"]), set(universe()))


class TestSuggestedThresholds(unittest.TestCase):
    def test_derived_from_the_dataset_not_from_us_defaults(self):
        """A CNY or KRW dataset must not inherit a $5 floor."""
        frames = {f"S{i}": synthetic.multi_cycle(cycles=5, seed=i, start_price=800.0 + 50 * i)
                  for i in range(5)}
        out = summary_mod.suggest_thresholds(summary_mod.summarise(frames))
        self.assertGreater(out["min_price"], 100, "suggestion ignored the dataset's price scale")

    def test_empty_dataset_suggests_nothing(self):
        self.assertEqual(summary_mod.suggest_thresholds(summary_mod.summarise({})), {})
