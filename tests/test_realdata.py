"""Robustness against the shapes real vendor data actually arrives in.

Hand-built fixtures are clean. Exported CSVs are not: duplicated dates, rows out
of order, holiday gaps, blank cells, zero-volume halts, symbols that delist
mid-sample, and — the one that quietly ruins a backtest — a file carrying both
`Close` and `Adj Close`.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qscan import setups  # noqa: E402
from qscan.config import DEFAULT  # noqa: E402
from qscan.data import PriceStore  # noqa: E402
from qscan.indicators import annotate, normalize_ohlcv  # noqa: E402
from tests import synthetic  # noqa: E402


class TestNormalizeMessyInput(unittest.TestCase):
    def setUp(self):
        self.clean = synthetic.textbook_breakout()

    def test_unsorted_rows_are_ordered(self):
        shuffled = self.clean.sample(frac=1.0, random_state=0)
        out = normalize_ohlcv(shuffled)
        self.assertTrue(out.index.is_monotonic_increasing)
        self.assertEqual(len(out), len(self.clean))

    def test_duplicate_dates_keep_the_last_row(self):
        dup = pd.concat([self.clean, self.clean.tail(3)])
        out = normalize_ohlcv(dup)
        self.assertFalse(out.index.has_duplicates)
        self.assertEqual(len(out), len(self.clean))

    def test_blank_cells_are_dropped_not_propagated(self):
        holed = self.clean.copy()
        holed.iloc[5:8, holed.columns.get_loc("close")] = np.nan
        out = normalize_ohlcv(holed)
        self.assertEqual(len(out), len(self.clean) - 3)
        self.assertFalse(out.isna().any().any())

    def test_zero_and_negative_prices_are_dropped(self):
        bad = self.clean.copy()
        bad.iloc[10, bad.columns.get_loc("low")] = 0.0
        bad.iloc[11, bad.columns.get_loc("open")] = -1.0
        out = normalize_ohlcv(bad)
        self.assertEqual(len(out), len(self.clean) - 2)

    def test_zero_volume_bars_survive(self):
        """A halted day is real data — dropping it would fabricate continuity."""
        halted = self.clean.copy()
        halted.iloc[20, halted.columns.get_loc("volume")] = 0.0
        out = normalize_ohlcv(halted)
        self.assertEqual(len(out), len(self.clean))

    def test_mixed_case_and_spaced_headers(self):
        odd = self.clean.copy()
        odd.columns = [" Open ", "HIGH", "Low", "Close", " Volume"]
        out = normalize_ohlcv(odd)
        self.assertEqual(list(out.columns), ["open", "high", "low", "close", "volume"])

    def test_missing_column_is_a_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_ohlcv(self.clean.drop(columns=["volume"]))
        self.assertIn("volume", str(ctx.exception))

    def test_extra_columns_are_ignored(self):
        wide = self.clean.copy()
        wide["dividends"] = 0.0
        wide["stock splits"] = 0.0
        out = normalize_ohlcv(wide)
        self.assertEqual(list(out.columns), ["open", "high", "low", "close", "volume"])


class TestAdjustedClose(unittest.TestCase):
    """A file with both Close and Adj Close must be adjusted consistently."""

    def _yahoo_style(self) -> pd.DataFrame:
        raw = synthetic.textbook_breakout().copy()
        # Simulate a 2:1 split 40 bars from the end: raw prices double before it,
        # adjusted prices are continuous.
        split_at = len(raw) - 40
        factor = np.ones(len(raw))
        factor[:split_at] = 2.0
        out = pd.DataFrame(
            {
                "Open": raw["open"] * factor,
                "High": raw["high"] * factor,
                "Low": raw["low"] * factor,
                "Close": raw["close"] * factor,
                "Adj Close": raw["close"],
                "Volume": raw["volume"],
            },
            index=raw.index,
        )
        return out

    def test_adjusted_close_is_preferred_over_raw_close(self):
        out = normalize_ohlcv(self._yahoo_style())
        expected = synthetic.textbook_breakout()["close"]
        pd.testing.assert_series_equal(out["close"], expected, check_names=False, rtol=1e-9)

    def test_whole_bar_is_scaled_not_just_the_close(self):
        """Adjusting close alone would leave high < close and corrupt every range."""
        out = normalize_ohlcv(self._yahoo_style())
        self.assertTrue((out["high"] >= out["close"] - 1e-9).all())
        self.assertTrue((out["low"] <= out["close"] + 1e-9).all())
        self.assertTrue((out["high"] >= out["low"]).all())

    def test_adjustment_removes_the_split_cliff(self):
        adjusted = normalize_ohlcv(self._yahoo_style())
        step = (adjusted["close"] / adjusted["close"].shift(1)).dropna()
        self.assertLess(step.min(), 1.5)
        self.assertGreater(step.min(), 0.5, "a 50% cliff survived the adjustment")

    def test_partially_adjusted_file_is_repaired_not_left_corrupt(self):
        """Some vendors adjust Close but not OHL, leaving high < close.

        A negative daily range poisons ADR and every gate built on it, so the
        extremes are widened to contain the body rather than passed through.
        """
        raw = synthetic.textbook_breakout().copy()
        frame = pd.DataFrame(
            {
                "Open": raw["open"], "High": raw["high"], "Low": raw["low"],
                "Close": raw["close"] * 2.0,  # only the close was scaled
                "Adj Close": raw["close"],
                "Volume": raw["volume"],
            },
            index=raw.index,
        )
        out = normalize_ohlcv(frame)
        self.assertTrue((out["high"] >= out["close"] - 1e-9).all())
        self.assertTrue((out["high"] >= out["open"] - 1e-9).all())
        self.assertTrue((out["low"] <= out["close"] + 1e-9).all())
        self.assertTrue(((out["high"] / out["low"] - 1.0) >= 0).all(), "negative range survived")

    def test_repair_does_not_touch_already_valid_bars(self):
        clean = synthetic.textbook_breakout()
        out = normalize_ohlcv(clean)
        pd.testing.assert_frame_equal(out, clean, check_names=False, rtol=1e-12)

    def test_adj_close_alone_still_works(self):
        raw = synthetic.textbook_breakout().copy()
        frame = raw.rename(columns={"close": "Adj Close"})
        frame.columns = [c.title() if c != "Adj Close" else c for c in frame.columns]
        out = normalize_ohlcv(frame)
        self.assertIn("close", out.columns)


class TestUniverseScale(unittest.TestCase):
    """A universe of CSVs on disk, including junk symbols, must scan cleanly."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.good = [f"GOOD{i:02d}" for i in range(12)]
        for i, sym in enumerate(cls.good):
            df = synthetic.multi_cycle(cycles=5, seed=i, start_price=12 + 2 * i)
            df.index.name = "date"
            df.to_csv(cls.tmp / f"{sym}.csv")

        # A short listing, a delisted name, an empty file and a malformed one.
        short = synthetic.textbook_breakout().head(40)
        short.index.name = "date"
        short.to_csv(cls.tmp / "TOOSHORT.csv")

        delisted = synthetic.multi_cycle(cycles=5, seed=99)
        delisted = delisted.iloc[: len(delisted) // 2]
        delisted.index.name = "date"
        delisted.to_csv(cls.tmp / "DELISTED.csv")

        (cls.tmp / "EMPTY.csv").write_text("date,open,high,low,close,volume\n")
        (cls.tmp / "BROKEN.csv").write_text("date,foo,bar\n2020-01-01,1,2\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_junk_symbols_do_not_abort_the_scan(self):
        store = PriceStore(csv_dir=self.tmp, cache_dir=self.tmp / "cache")
        setup = setups.get("breakout")
        cfg = setup.config("relaxed")

        loaded, failed, signals = 0, 0, 0
        for sym in self.good + ["TOOSHORT", "DELISTED", "EMPTY", "BROKEN", "NOSUCHTICKER"]:
            try:
                df = store.get(sym, "1900-01-01", "2100-01-01")
            except Exception:
                failed += 1
                continue
            if len(df) < 200:
                continue
            loaded += 1
            signals += len(
                setups.scan_frame(annotate(df, cfg), setup, cfg, symbol=sym,
                                  positions=range(200, len(df)))
            )

        # The good names plus DELISTED, which has real history and should load.
        self.assertEqual(loaded, len(self.good) + 1)
        self.assertGreaterEqual(failed, 2, "malformed files should surface as failures")
        self.assertGreater(signals, 0, "no signals found across a healthy universe")

    def test_delisted_symbol_signals_only_within_its_own_history(self):
        store = PriceStore(csv_dir=self.tmp, cache_dir=self.tmp / "cache2")
        setup = setups.get("breakout")
        cfg = setup.config("relaxed")
        df = store.get("DELISTED", "1900-01-01", "2100-01-01")
        ann = annotate(df, cfg)
        hits = setups.scan_frame(ann, setup, cfg, symbol="DELISTED", positions=range(200, len(ann)))
        for h in hits:
            self.assertLessEqual(pd.Timestamp(h["date"]), ann.index[-1])


if __name__ == "__main__":
    unittest.main()


class TestZeroPaddedTickers(unittest.TestCase):
    """Regression: pandas reads 000001 as the integer 1.

    That silently unmatches the entire Shenzhen half of the A-share market,
    most of Hong Kong, and any vendor that zero-pads. The failure is invisible —
    the ingest reports success and the symbols simply never join.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_long_format_preserves_leading_zeros(self):
        from qscan.ingest import _read_tabular

        rows = []
        for sym in ("000001", "000858", "600519"):
            d = synthetic.textbook_breakout().head(30)
            rows.append(pd.DataFrame({
                "symbol": sym, "date": d.index.date, "open": d.open,
                "high": d.high, "low": d.low, "close": d.close, "volume": d.volume,
            }))
        path = self.tmp / "long.csv"
        pd.concat(rows).to_csv(path, index=False)

        df = _read_tabular(path, "long.csv")
        self.assertEqual(set(df["symbol"].unique()), {"000001", "000858", "600519"})

    def test_prices_are_still_numeric_after_the_symbol_fix(self):
        """Reading the symbol as text must not turn the price columns into strings."""
        from qscan.ingest import _read_tabular

        d = synthetic.textbook_breakout().head(20)
        path = self.tmp / "l2.csv"
        pd.DataFrame({
            "symbol": "000001", "date": d.index.date, "open": d.open, "high": d.high,
            "low": d.low, "close": d.close, "volume": d.volume,
        }).to_csv(path, index=False)

        df = _read_tabular(path, "l2.csv")
        for col in ("open", "high", "low", "close", "volume"):
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]), f"{col} became text")
