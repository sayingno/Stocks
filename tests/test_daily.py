"""End-to-end tests for the scheduled `daily` command."""

import json
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qscan.cli import main  # noqa: E402
from tests import synthetic  # noqa: E402

BUILDERS = {
    "GOODBRK": lambda: synthetic.winner_after_signal(synthetic.textbook_breakout()),
    "COILED": synthetic.coiled_setup,
    "DOWNTRND": synthetic.downtrend,
    "UNDER50": synthetic.below_slow_ma,
}


class DailyTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.csv_dir = self.tmp / "csv"
        self.csv_dir.mkdir()
        self.out = self.tmp / "out"
        self.repo = self.tmp / "db"

        for name, builder in BUILDERS.items():
            df = builder().copy()
            # Anchor to today so the freshness check passes.
            df.index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(df))
            df.index.name = "date"
            df.round(4).to_csv(self.csv_dir / f"{name}.csv")

        self.universe = self.tmp / "universe.txt"
        self.universe.write_text("\n".join(BUILDERS) + "\n")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_daily(self, *extra: str) -> tuple[int, dict]:
        argv = [
            "daily",
            "--csv-dir", str(self.csv_dir),
            "--repo", str(self.repo),
            "--out", str(self.out),
            "--universe", str(self.universe),
            *extra,
        ]
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = main(argv)
        text = buf.getvalue()
        start, end = text.find("{"), text.rfind("}")
        summary = json.loads(text[start : end + 1]) if start >= 0 else {}
        return code, summary


class TestHappyPath(DailyTestCase):
    def test_full_pipeline_succeeds(self):
        code, summary = self.run_daily("--charts", "3")
        self.assertEqual(code, 0, summary.get("problems"))
        self.assertEqual(summary["problems"], [])
        self.assertEqual(summary["universe"], len(BUILDERS))
        self.assertGreaterEqual(summary["candidates"], 1)

    def test_writes_report_csv_and_latest(self):
        _code, summary = self.run_daily("--charts", "3")
        self.assertTrue(Path(summary["report"]).exists())
        self.assertTrue(Path(summary["csv"]).exists())
        latest = Path(summary["latest"])
        self.assertTrue(latest.exists())
        self.assertEqual(latest.read_text(), Path(summary["report"]).read_text())

    def test_renders_charts_for_candidates(self):
        _code, summary = self.run_daily("--charts", "3")
        self.assertGreaterEqual(summary["charts"], 1)
        pngs = list((self.out / "charts").rglob("*.png"))
        self.assertEqual(len(pngs), summary["charts"])

    def test_report_embeds_the_charts(self):
        _code, summary = self.run_daily("--charts", "3")
        self.assertIn("data:image/png;base64,", Path(summary["report"]).read_text())

    def test_second_run_is_incremental_and_still_succeeds(self):
        first_code, _ = self.run_daily("--charts", "1")
        second_code, second = self.run_daily("--charts", "1")
        self.assertEqual(first_code, 0)
        self.assertEqual(second_code, 0)
        self.assertEqual(second["problems"], [])

    def test_rejected_names_are_absent_from_the_report(self):
        _code, summary = self.run_daily("--charts", "4")
        rows = pd.read_csv(summary["csv"])
        self.assertNotIn("DOWNTRND", set(rows["symbol"]))


class TestHealthChecks(DailyTestCase):
    def test_broken_feed_exits_non_zero(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        code, summary = self.run_daily("--csv-dir", str(empty))
        self.assertEqual(code, 1)
        self.assertTrue(any("fetches failed" in p for p in summary["problems"]))

    def test_stale_data_is_flagged(self):
        # Rewrite the CSVs so the newest bar is a month old.
        for name, builder in BUILDERS.items():
            df = builder().copy()
            df.index = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=30), periods=len(df))
            df.index.name = "date"
            df.round(4).to_csv(self.csv_dir / f"{name}.csv")
        code, summary = self.run_daily()
        self.assertEqual(code, 1)
        self.assertTrue(any("days old" in p for p in summary["problems"]))

    def test_stale_tolerance_can_be_widened(self):
        for name, builder in BUILDERS.items():
            df = builder().copy()
            df.index = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=10), periods=len(df))
            df.index.name = "date"
            df.round(4).to_csv(self.csv_dir / f"{name}.csv")
        code, summary = self.run_daily("--max-stale-days", "60")
        self.assertEqual(code, 0, summary.get("problems"))

    def test_empty_result_is_success_not_failure(self):
        """A quiet tape means no candidates - that is a valid outcome, not an error."""
        code, summary = self.run_daily("--min-score", "99.9")
        self.assertEqual(code, 0)
        self.assertEqual(summary["candidates"], 0)
        self.assertIn("No candidates passed the filters", Path(summary["report"]).read_text())


class TestFiltering(DailyTestCase):
    def test_state_filter_narrows_results(self):
        _code, everything = self.run_daily()
        _code, only_breakouts = self.run_daily("--state", "breakout")
        self.assertLessEqual(only_breakouts["candidates"], everything["candidates"])
        self.assertEqual(only_breakouts["setups"], 0)

    def test_skip_update_reuses_the_database(self):
        self.run_daily()
        code, summary = self.run_daily("--skip-update")
        self.assertEqual(code, 0)
        self.assertGreaterEqual(summary["candidates"], 1)

    def test_preset_changes_the_result_count(self):
        _code, default = self.run_daily()
        _code, strict = self.run_daily("--preset", "strict")
        self.assertLessEqual(strict["candidates"], default["candidates"])


if __name__ == "__main__":
    unittest.main()
