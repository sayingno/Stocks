import re
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qscan.report import build_report  # noqa: E402


def sample_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-08-07"), "symbol": "AAA", "state": "breakout", "score": 82.0,
             "close": 44.10, "pivot": 43.80, "entry": 43.84, "stop": 41.9, "risk_pct": 0.044,
             "shares": 115, "position_value": 5041.0, "target_2r": 47.7, "adr20": 5.2,
             "impulse_gain": 0.74, "base_len": 18, "base_depth": 0.11, "dist_from_pivot": 0.0},
            {"date": pd.Timestamp("2026-08-07"), "symbol": "BBB", "state": "setup", "score": 68.5,
             "close": 20.20, "pivot": 21.40, "entry": 21.42, "stop": 20.5, "risk_pct": 0.043,
             "shares": 240, "position_value": 5140.0, "target_2r": 23.2, "adr20": 4.1,
             "impulse_gain": 0.52, "base_len": 31, "base_depth": 0.17, "dist_from_pivot": 0.056},
        ]
    )


class TestBuildReport(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_writes_a_self_contained_page(self):
        out = build_report(sample_candidates(), self.tmp / "r.html", universe_size=500)
        text = out.read_text()
        self.assertTrue(out.exists())
        self.assertIn("<!doctype html>", text)
        # No external requests allowed - the report must survive being emailed.
        self.assertNotIn("<script", text.lower())
        for pattern in (r'src="https?://', r'href="https?://', r"@import"):
            self.assertIsNone(re.search(pattern, text), f"external reference {pattern}")

    def test_lists_every_candidate_with_its_state(self):
        text = build_report(sample_candidates(), self.tmp / "r.html").read_text()
        self.assertIn("AAA", text)
        self.assertIn("BBB", text)
        # State is spelled out as text, never colour alone.
        self.assertIn(">breakout<", text)
        self.assertIn(">setup<", text)

    def test_tiles_count_states(self):
        text = build_report(sample_candidates(), self.tmp / "r.html", universe_size=500).read_text()
        self.assertIn("Breaking out", text)
        self.assertIn("Coiled setups", text)
        self.assertIn("from 500 symbols", text)

    def test_empty_scan_still_renders(self):
        out = build_report(pd.DataFrame(columns=["symbol", "state", "score"]), self.tmp / "empty.html")
        text = out.read_text()
        self.assertIn("No candidates passed the filters", text)

    def test_charts_are_inlined_as_base64(self):
        png = self.tmp / "AAA.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)  # header is enough; we only base64 it
        text = build_report(sample_candidates(), self.tmp / "r.html", charts={"AAA": png}).read_text()
        self.assertIn("data:image/png;base64,", text)
        self.assertIn('alt="AAA daily chart"', text)

    def test_missing_chart_file_is_skipped_not_fatal(self):
        text = build_report(
            sample_candidates(), self.tmp / "r.html", charts={"AAA": self.tmp / "nope.png"}
        ).read_text()
        self.assertNotIn("data:image/png;base64,", text)

    def test_symbol_names_are_escaped(self):
        df = sample_candidates()
        df.loc[0, "symbol"] = "<script>x</script>"
        text = build_report(df, self.tmp / "r.html").read_text()
        self.assertNotIn("<script>x</script>", text)
        self.assertIn("&lt;script&gt;", text)

    def test_dark_mode_tokens_are_defined_under_both_scopes(self):
        text = build_report(sample_candidates(), self.tmp / "r.html").read_text()
        self.assertIn("prefers-color-scheme: dark", text)
        self.assertIn(':root[data-theme="dark"]', text)


if __name__ == "__main__":
    unittest.main()
