import shutil
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qscan.data import DataError  # noqa: E402
from qscan.repository import FAILURE_LIMIT, PriceRepository  # noqa: E402
from tests.synthetic import textbook_breakout  # noqa: E402


class _FakeProvider:
    """Serves slices of a fixed frame and counts calls, like a rate-limited API."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls: list[tuple[str, str, str]] = []
        self.fail_for: set[str] = set()

    def __call__(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        if symbol in self.fail_for:
            raise DataError(f"{symbol}: simulated outage")
        return self.frame.loc[str(start) : str(end)]


class RepoTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.frame = textbook_breakout()
        # Anchor the data so "today" sits on the last bar.
        self.today = self.frame.index[-1].date()
        self.provider = _FakeProvider(self.frame)
        # csv_dir avoids the network providers; retry_attempts=1 skips the backoff sleeps.
        self.repo = PriceRepository(root=self.tmp, csv_dir=self.tmp, retry_attempts=1)
        self.repo._fetch = self.provider
        self.repo.provider_name = "fake"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestFirstBuild(RepoTestCase):
    def test_creates_full_history(self):
        action, rows = self.repo.update_symbol("TEST", today=self.today)
        self.assertEqual(action, "created")
        self.assertEqual(rows, len(self.frame))
        stored = self.repo.load("TEST")
        self.assertEqual(len(stored), len(self.frame))

    def test_manifest_records_dates_and_persists(self):
        self.repo.update_symbol("TEST", today=self.today)
        self.repo.save_manifest()
        reopened = PriceRepository(root=self.tmp, csv_dir=self.tmp)
        entry = reopened.entry("TEST")
        self.assertEqual(entry["last_date"], str(self.frame.index[-1].date()))
        self.assertEqual(entry["rows"], len(self.frame))

    def test_empty_provider_response_raises(self):
        self.provider.frame = self.frame.iloc[0:0]
        with self.assertRaises(DataError):
            self.repo.update_symbol("TEST", today=self.today)


class TestIncremental(RepoTestCase):
    def test_second_run_is_a_no_op(self):
        self.repo.update_symbol("TEST", today=self.today)
        calls_before = len(self.provider.calls)
        action, rows = self.repo.update_symbol("TEST", today=self.today)
        self.assertEqual(action, "unchanged")
        self.assertEqual(rows, 0)
        self.assertEqual(len(self.provider.calls), calls_before, "should not have hit the provider")

    def test_new_bars_are_appended_not_refetched(self):
        # Seed the database as of three days before the data ends.
        cutoff = self.frame.index[-4]
        self.provider.frame = self.frame.loc[:cutoff]
        self.repo.update_symbol("TEST", today=cutoff.date())
        seeded = len(self.repo.load("TEST", use_cache=False))

        # Now the provider has the later bars too.
        self.provider.frame = self.frame
        self.provider.calls.clear()
        action, added = self.repo.update_symbol("TEST", today=self.today)

        self.assertEqual(action, "appended")
        self.assertEqual(added, 3)
        self.assertEqual(len(self.repo.load("TEST", use_cache=False)), seeded + 3)
        # The delta fetch must ask for a short overlap, not the whole history.
        _sym, start, _end = self.provider.calls[0]
        self.assertGreater(pd.Timestamp(start), self.frame.index[0])

    def test_appended_rows_keep_the_series_sorted_and_unique(self):
        cutoff = self.frame.index[-6]
        self.provider.frame = self.frame.loc[:cutoff]
        self.repo.update_symbol("TEST", today=cutoff.date())
        self.provider.frame = self.frame
        self.repo.update_symbol("TEST", today=self.today)
        stored = self.repo.load("TEST", use_cache=False)
        self.assertTrue(stored.index.is_monotonic_increasing)
        self.assertFalse(stored.index.has_duplicates)


class TestSplitDetection(RepoTestCase):
    def test_readjusted_history_triggers_a_full_refetch(self):
        cutoff = self.frame.index[-4]
        self.provider.frame = self.frame.loc[:cutoff]
        self.repo.update_symbol("TEST", today=cutoff.date())
        before = self.repo.load("TEST", use_cache=False)

        # A 2:1 split halves every historical price.
        self.provider.frame = self.frame.copy()
        for col in ("open", "high", "low", "close"):
            self.provider.frame[col] = self.provider.frame[col] / 2.0

        action, _ = self.repo.update_symbol("TEST", today=self.today)
        self.assertEqual(action, "readjusted")

        after = self.repo.load("TEST", use_cache=False)
        shared = before.index.intersection(after.index)
        ratio = (after.loc[shared, "close"] / before.loc[shared, "close"]).median()
        self.assertAlmostEqual(ratio, 0.5, places=2, msg="stored history should be fully rewritten")

    def test_tiny_price_drift_does_not_trigger_a_refetch(self):
        cutoff = self.frame.index[-4]
        self.provider.frame = self.frame.loc[:cutoff]
        self.repo.update_symbol("TEST", today=cutoff.date())
        self.provider.frame = self.frame.copy()
        for col in ("open", "high", "low", "close"):
            self.provider.frame[col] = self.provider.frame[col] * 1.0001  # rounding noise
        action, _ = self.repo.update_symbol("TEST", today=self.today)
        self.assertEqual(action, "appended")


class TestFailureHandling(RepoTestCase):
    def test_failures_are_counted_and_do_not_abort_the_run(self):
        self.provider.fail_for = {"BAD"}
        stats = self.repo.update(["TEST", "BAD"], workers=2, today=self.today)
        self.assertEqual(stats.failed, 1)
        self.assertEqual(stats.created, 1)
        self.assertEqual(self.repo.entry("BAD")["failures"], 1)
        self.assertIsNotNone(self.repo.load("TEST"))

    def test_repeated_failures_bench_the_symbol(self):
        self.provider.fail_for = {"BAD"}
        for _ in range(FAILURE_LIMIT):
            self.repo.update(["BAD"], workers=1, today=self.today)
        self.assertGreaterEqual(self.repo.entry("BAD")["failures"], FAILURE_LIMIT)

        calls_before = len(self.provider.calls)
        stats = self.repo.update(["BAD"], workers=1, today=self.today)
        self.assertEqual(stats.skipped, 1)
        self.assertEqual(len(self.provider.calls), calls_before, "benched symbol should not be fetched")

    def test_force_retries_a_benched_symbol(self):
        self.provider.fail_for = {"BAD"}
        for _ in range(FAILURE_LIMIT):
            self.repo.update(["BAD"], workers=1, today=self.today)
        calls_before = len(self.provider.calls)
        self.repo.update(["BAD"], workers=1, today=self.today, force=True)
        self.assertGreater(len(self.provider.calls), calls_before)

    def test_bench_expires_after_the_cooldown(self):
        self.provider.fail_for = {"BAD"}
        for _ in range(FAILURE_LIMIT):
            self.repo.update(["BAD"], workers=1, today=self.today)
        stale = (datetime.now() - timedelta(days=30)).isoformat(timespec="seconds")
        self.repo._record("BAD", last_failure=stale)
        self.assertFalse(self.repo._should_skip("BAD", date.today()))


class TestPriceStoreWindowing(unittest.TestCase):
    """Regression: the in-memory memo must not be keyed by symbol alone."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        df = textbook_breakout()
        df.index.name = "date"
        df.to_csv(self.tmp / "TEST.csv")
        self.frame = df

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_second_call_with_a_wider_window_is_not_served_the_first_slice(self):
        from qscan.data import PriceStore

        store = PriceStore(csv_dir=self.tmp, cache_dir=self.tmp / "cache")
        mid = self.frame.index[len(self.frame) // 2]

        narrow = store.get("TEST", str(mid.date()), "2100-01-01")
        wide = store.get("TEST", "1900-01-01", "2100-01-01")

        self.assertLess(len(narrow), len(self.frame))
        self.assertEqual(len(wide), len(self.frame), "memo returned the earlier narrow slice")

    def test_disjoint_window_returns_empty_not_stale_rows(self):
        from qscan.data import PriceStore

        store = PriceStore(csv_dir=self.tmp, cache_dir=self.tmp / "cache")
        store.get("TEST", "1900-01-01", "2100-01-01")
        self.assertTrue(store.get("TEST", "2099-01-01", "2099-12-31").empty)


class TestParallelAndCoverage(RepoTestCase):
    def test_many_symbols_update_in_parallel(self):
        symbols = [f"SYM{i}" for i in range(12)]
        stats = self.repo.update(symbols, workers=6, today=self.today)
        self.assertEqual(stats.created, 12)
        self.assertEqual(stats.failed, 0)
        self.assertEqual(len(self.repo.symbols()), 12)

    def test_coverage_reports_totals(self):
        self.repo.update(["A", "B"], workers=2, today=self.today)
        cov = self.repo.coverage()
        self.assertEqual(cov["symbols_on_disk"], 2)
        self.assertEqual(cov["symbols_healthy"], 2)
        self.assertEqual(cov["total_rows"], 2 * len(self.frame))
        self.assertEqual(cov["latest_date"], str(self.frame.index[-1].date()))

    def test_manifest_survives_a_corrupt_file(self):
        (self.tmp / "manifest.json").write_text("{not json")
        repo = PriceRepository(root=self.tmp, csv_dir=self.tmp)
        self.assertEqual(repo.entry("ANY"), {})


if __name__ == "__main__":
    unittest.main()
