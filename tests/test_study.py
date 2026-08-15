"""Tests for the event study and the earnings calendar.

The load-bearing tests here are the two statistical ones: the study must
*recover a relationship that was deliberately planted*, and must *report noise
when none exists*. A tool that only ever finds signal is worse than useless —
it manufactures confidence.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qscan import earnings as earnings_mod  # noqa: E402
from qscan import study as study_mod  # noqa: E402
from qscan.config import DEFAULT  # noqa: E402
from qscan.indicators import annotate  # noqa: E402
from tests import synthetic  # noqa: E402


def planted_universe(n_symbols=14, seed=0, link=True):
    """Symbols whose gaps run only when the stock was dormant beforehand.

    With link=False the outcome is decided by a coin flip instead, so the
    antecedent carries no information and the study must say so.
    """
    rng = np.random.default_rng(seed)
    frames = {}
    for i in range(n_symbols):
        closes, ranges, vols, gaps = [], [], [], []
        price = 20.0 + i
        for cycle in range(9):
            dormant = bool(rng.random() < 0.5)
            # Antecedent: a long quiet stretch, or a steady prior climb.
            n = int(rng.integers(150, 190))
            if dormant:
                seg = price * (1 + np.sin(np.arange(n) / 23.0) * 0.04)
            else:
                seg = np.linspace(price, price * 1.45, n)
            closes.extend(seg); ranges.extend([0.03] * n); vols.extend(rng.uniform(1e6, 1.4e6, n))
            price = float(seg[-1])

            gaps.append((len(closes), 0.11))
            runs = dormant if link else bool(rng.random() < 0.5)
            n = 30
            # Steep enough that a runner clears the 20% mover bar inside the
            # 10-bar outcome window; a dud must not.
            end = price * (2.10 if runs else 0.95)
            seg = np.linspace(price * 1.02, end, n)
            closes.extend(seg); ranges.extend([0.05] * n); vols.extend(rng.uniform(2e6, 3e6, n))
            price = float(seg[-1]) * 0.7 + (20.0 + i) * 0.3
            n = 15
            seg = np.linspace(float(closes[-1]), price, n)
            closes.extend(seg); ranges.extend([0.03] * n); vols.extend(rng.uniform(1e6, 1.3e6, n))

        df = synthetic.build(closes, ranges, vols, start="2015-01-02")
        for pos, size in gaps:
            if pos <= 0 or pos >= len(df):
                continue
            prev = float(df["close"].iloc[pos - 1])
            op = prev * (1 + size)
            scale = op / float(df["open"].iloc[pos])
            idx = df.index[pos:]
            for col in ("open", "high", "low", "close"):
                df.loc[idx, col] = df.loc[idx, col] * scale
            bar = df.index[pos]
            df.loc[bar, "open"] = op
            df.loc[bar, "close"] = max(float(df.loc[bar, "close"]), op * 1.02)
            df.loc[bar, "high"] = max(float(df.loc[bar, "close"]) * 1.01, float(df.loc[bar, "high"]))
            df.loc[bar, "low"] = min(op * 0.99, float(df.loc[bar, "low"]))
            df.loc[bar, "volume"] = float(df.loc[bar, "volume"]) * 5
        frames[f"S{i:02d}"] = annotate(df, DEFAULT)
    return frames


CFG = study_mod.StudyConfig(event="gap", min_gap=0.08, outcome_horizon=10, outcome_threshold=0.20)


class TestEventCollection(unittest.TestCase):
    def setUp(self):
        self.ann = annotate(synthetic.ep_cycles(cycles=5, seed=1), DEFAULT)

    def test_finds_the_gaps(self):
        ev = study_mod.collect_events(self.ann, "T", CFG)
        self.assertGreater(len(ev), 2)
        self.assertTrue(all(e["gap"] >= CFG.min_gap for e in ev))

    def test_antecedents_are_read_before_the_event(self):
        """The 'what did it look like going in' fields must exclude the event bar."""
        ev = study_mod.collect_events(self.ann, "T", CFG)
        e = ev[0]
        prev = e["bar"] - 1
        self.assertAlmostEqual(e["ret_3m"], float(self.ann["ret_3m"].iloc[prev]), places=9)
        self.assertAlmostEqual(e["adr20"], float(self.ann["adr20"].iloc[prev]), places=9)

    def test_outcome_is_measured_forward_from_the_event_close(self):
        ev = study_mod.collect_events(self.ann, "T", CFG)
        e = ev[0]
        c = self.ann["close"].to_numpy(float)
        expected = c[e["bar"] + CFG.outcome_horizon] / c[e["bar"]] - 1.0
        self.assertAlmostEqual(e["fwd_return"], expected, places=9)

    def test_events_near_the_end_are_dropped(self):
        """No event may be kept without a full forward window."""
        ev = study_mod.collect_events(self.ann, "T", CFG)
        for e in ev:
            self.assertLess(e["bar"] + CFG.outcome_horizon, len(self.ann))

    def test_cooldown_prevents_double_counting(self):
        cfg = study_mod.StudyConfig(event="all", cooldown=20, outcome_horizon=5)
        ev = study_mod.collect_events(self.ann, "T", cfg)
        bars = [e["bar"] for e in ev]
        self.assertTrue(all(b - a > 20 for a, b in zip(bars, bars[1:])))

    def test_start_cutoff_respected(self):
        cutoff = pd.Timestamp("2017-01-01")
        ev = study_mod.collect_events(self.ann, "T", CFG, start=cutoff)
        self.assertTrue(all(e["date"] >= cutoff for e in ev))


class TestRecoversPlantedSignal(unittest.TestCase):
    """The whole point: find what is there, and only what is there."""

    def test_finds_the_relationship_that_was_planted(self):
        res = study_mod.run(planted_universe(link=True), CFG)
        self.assertGreater(res.summary["events"], 40)
        row = res.cohorts.set_index("antecedent").loc["ret_3m"]
        self.assertEqual(row["signal"], "yes", f"planted ret_3m link not detected (t={row['t']})")
        self.assertLess(row["lift"], 0, "movers should have been the *dormant* ones")

    def test_reports_noise_when_there_is_none(self):
        """With the outcome randomised, ret_3m must stop looking predictive."""
        res = study_mod.run(planted_universe(link=False, seed=5), CFG)
        row = res.cohorts.set_index("antecedent").loc["ret_3m"]
        self.assertEqual(
            row["signal"], "noise",
            f"claimed signal (t={row['t']}) where the outcome was a coin flip",
        )

    def test_bucket_lift_is_monotonic_for_the_planted_field(self):
        res = study_mod.run(planted_universe(link=True), CFG, bucket_fields=["ret_3m"])
        tbl = res.buckets["ret_3m"]
        self.assertGreater(tbl["mover_rate"].iloc[0], tbl["mover_rate"].iloc[-1])

    def test_base_rate_is_reported_alongside_every_bucket(self):
        res = study_mod.run(planted_universe(link=True), CFG, bucket_fields=["ret_3m"])
        tbl = res.buckets["ret_3m"]
        self.assertIn("base_rate", tbl.columns)
        self.assertIn("lift", tbl.columns)
        np.testing.assert_allclose(
            tbl["lift"], tbl["mover_rate"] - tbl["base_rate"], atol=1e-9
        )


class TestCohortMechanics(unittest.TestCase):
    def test_empty_input_is_safe(self):
        res = study_mod.run({}, CFG)
        self.assertEqual(res.summary["events"], 0)
        self.assertTrue(res.cohorts.empty)

    def test_welch_t_sign_follows_the_difference(self):
        hi = np.random.default_rng(0).normal(1.0, 0.2, 200)
        lo = np.random.default_rng(1).normal(0.0, 0.2, 200)
        self.assertGreater(study_mod._welch_t(hi, lo), 2)
        self.assertLess(study_mod._welch_t(lo, hi), -2)

    def test_welch_t_on_identical_samples_is_near_zero(self):
        x = np.random.default_rng(2).normal(0, 1, 300)
        y = np.random.default_rng(3).normal(0, 1, 300)
        self.assertLess(abs(study_mod._welch_t(x, y)), 2)

    def test_bucket_by_unknown_field_is_empty_not_an_error(self):
        res = study_mod.run(planted_universe(n_symbols=6), CFG)
        self.assertTrue(study_mod.bucket_by(res.events, "nope").empty)


class TestEarningsCalendar(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_loads_symbol_and_date(self):
        p = self.tmp / "cal.csv"
        p.write_text("symbol,date\n600519,2024-04-27\n600519,2024-08-09\n000001,2024-04-25\n")
        cal = earnings_mod.load(p)
        self.assertEqual(set(cal), {"600519", "000001"})
        self.assertEqual(len(cal["600519"]), 2)

    def test_accepts_vendor_column_spellings(self):
        p = self.tmp / "cal2.csv"
        p.write_text("ticker,report_date\nAAPL,2024-05-02\n")
        self.assertIn("AAPL", earnings_mod.load(p))

    def test_missing_columns_raise_clearly(self):
        p = self.tmp / "bad.csv"
        p.write_text("foo,bar\n1,2\n")
        with self.assertRaises(ValueError):
            earnings_mod.load(p)

    def test_duplicates_collapse_and_dates_sort(self):
        p = self.tmp / "dup.csv"
        p.write_text("symbol,date\nX,2024-05-02\nX,2024-01-02\nX,2024-05-02\n")
        dates = earnings_mod.load(p)["X"]
        self.assertEqual(len(dates), 2)
        self.assertLess(dates[0], dates[1])

    def test_distances_count_bars_either_side(self):
        idx = pd.bdate_range("2024-01-01", periods=40)
        d = earnings_mod.distances(idx, [idx[10], idx[30]])
        self.assertEqual(d["days_since_earnings"].iloc[10], 0.0)
        self.assertEqual(d["days_since_earnings"].iloc[13], 3.0)
        self.assertEqual(d["days_to_next_earnings"].iloc[27], 3.0)

    def test_no_report_on_a_side_is_nan_not_zero(self):
        """'No next report known' must not read as 'the report is today'."""
        idx = pd.bdate_range("2024-01-01", periods=20)
        d = earnings_mod.distances(idx, [idx[10]])
        self.assertTrue(pd.isna(d["days_since_earnings"].iloc[0]))
        self.assertTrue(pd.isna(d["days_to_next_earnings"].iloc[15]))

    def test_inferred_days_are_spaced_like_reports(self):
        ann = annotate(synthetic.ep_cycles(cycles=5, seed=3), DEFAULT)
        picks = earnings_mod.infer_from_prices(ann, min_spacing=40)
        self.assertGreater(len(picks), 1)
        gaps = [(b - a).days for a, b in zip(picks, picks[1:])]
        self.assertTrue(all(g >= 40 for g in gaps))

    def test_attach_adds_both_columns(self):
        ann = annotate(synthetic.textbook_breakout(), DEFAULT)
        out = earnings_mod.attach(ann, [ann.index[50]])
        self.assertIn("days_since_earnings", out.columns)
        self.assertEqual(out["days_since_earnings"].iloc[55], 5.0)


if __name__ == "__main__":
    unittest.main()
