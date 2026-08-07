import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qscan.config import DEFAULT  # noqa: E402
from qscan.indicators import (  # noqa: E402
    adr_pct,
    annotate,
    atr,
    dollar_volume,
    moving_average,
    normalize_ohlcv,
    slope_pct_per_bar,
)
from tests.synthetic import build, textbook_breakout  # noqa: E402


class TestNormalize(unittest.TestCase):
    def test_lowercases_sorts_and_dedupes(self):
        idx = pd.to_datetime(["2021-01-05", "2021-01-04", "2021-01-05"])
        raw = pd.DataFrame(
            {"Open": [1, 2, 3], "High": [2, 3, 4], "Low": [0.5, 1, 2], "Close": [1.5, 2.5, 3.5], "Volume": [10, 20, 30]},
            index=idx,
        )
        out = normalize_ohlcv(raw)
        self.assertEqual(list(out.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(out), 2)
        self.assertTrue(out.index.is_monotonic_increasing)
        self.assertEqual(out["close"].iloc[-1], 3.5)  # last duplicate wins

    def test_missing_column_raises(self):
        df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0]}, index=pd.to_datetime(["2021-01-04"]))
        with self.assertRaises(ValueError):
            normalize_ohlcv(df)


class TestIndicators(unittest.TestCase):
    def test_moving_average_matches_manual_mean(self):
        s = pd.Series([1.0, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(moving_average(s, 3).iloc[-1], 5.0)
        self.assertTrue(np.isnan(moving_average(s, 3).iloc[1]))

    def test_adr_pct_on_constant_range(self):
        # Every bar has high/low exactly 5% apart -> ADR must be 5.0.
        n = 30
        df = build(np.full(n, 100.0), np.full(n, 0.0), np.full(n, 1e6))
        df["high"] = 105.0
        df["low"] = 100.0
        self.assertAlmostEqual(adr_pct(df, 20).iloc[-1], 5.0, places=6)

    def test_atr_positive_and_finite(self):
        df = textbook_breakout()
        value = atr(df, 14).iloc[-1]
        self.assertTrue(np.isfinite(value) and value > 0)

    def test_dollar_volume(self):
        n = 25
        df = build(np.full(n, 10.0), np.full(n, 0.01), np.full(n, 2e5))
        self.assertAlmostEqual(dollar_volume(df, 20).iloc[-1], 2e6, places=3)

    def test_slope_sign(self):
        rising = pd.Series(np.linspace(10, 20, 40))
        falling = pd.Series(np.linspace(20, 10, 40))
        self.assertGreater(slope_pct_per_bar(rising, 20).iloc[-1], 0)
        self.assertLess(slope_pct_per_bar(falling, 20).iloc[-1], 0)

    def test_annotate_adds_expected_columns(self):
        ann = annotate(textbook_breakout(), DEFAULT)
        for col in ("ma_fast", "ma_mid", "ma_slow", "adr20", "dollar_vol20", "ret_1m", "ret_3m", "ret_6m"):
            self.assertIn(col, ann.columns)
        self.assertTrue(np.isfinite(ann["adr20"].iloc[-1]))


if __name__ == "__main__":
    unittest.main()
