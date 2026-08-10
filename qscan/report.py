"""Self-contained HTML digest of a day's scan.

One file, no external requests: charts are inlined as base64 PNGs so the report
can be emailed, dropped in Dropbox, or opened months later and still render.

Colour roles come from a validated categorical pair (blue / orange) with badge
text alongside, so state never depends on colour alone.
"""

from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

# Categorical slots 1 and 2, light and dark steps. Validated as a pair in both
# modes (adjacent CVD dE 24.7 light / 26.8 dark).
_CSS = """
:root {
  color-scheme: light;
  --surface-0: #f4f3f0;
  --surface-1: #fcfcfb;
  --border:    #dedcd6;
  --text-1:    #0b0b0b;
  --text-2:    #52514e;
  --text-3:    #77756e;
  --breakout:  #2a78d6;
  --setup:     #eb6834;
  --good:      #0ca30c;
  --critical:  #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #111110;
    --surface-1: #1a1a19;
    --border:    #33322f;
    --text-1:    #ffffff;
    --text-2:    #c3c2b7;
    --text-3:    #8f8e85;
    --breakout:  #3987e5;
    --setup:     #d95926;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #111110;
  --surface-1: #1a1a19;
  --border:    #33322f;
  --text-1:    #ffffff;
  --text-2:    #c3c2b7;
  --text-3:    #8f8e85;
  --breakout:  #3987e5;
  --setup:     #d95926;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 24px 64px;
  background: var(--surface-0); color: var(--text-1);
  font: 15px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 22px; font-weight: 650; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--text-2); font-size: 13px; margin-bottom: 28px; }
h2 { font-size: 15px; font-weight: 650; margin: 40px 0 12px; letter-spacing: -0.005em; }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(148px, 1fr)); gap: 12px; }
.tile {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px;
}
.tile .label { font-size: 12px; color: var(--text-2); margin-bottom: 6px; }
.tile .value { font-size: 27px; font-weight: 620; line-height: 1.1; letter-spacing: -0.02em; }
.tile .note { font-size: 11.5px; color: var(--text-3); margin-top: 4px; }

.scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface-1); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 9px 12px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--border); }
th { font-weight: 600; color: var(--text-2); font-size: 11.5px; text-transform: uppercase;
     letter-spacing: 0.04em; position: sticky; top: 0; background: var(--surface-1); }
td { font-variant-numeric: tabular-nums; color: var(--text-1); }
td.sym, th.sym { text-align: left; font-weight: 620; }
tbody tr:last-child td { border-bottom: none; }

.badge {
  display: inline-block; padding: 1px 8px; border-radius: 999px;
  font-size: 11px; font-weight: 620; letter-spacing: 0.02em;
  border: 1.5px solid currentColor;
}
.badge.breakout { color: var(--breakout); }
.badge.setup    { color: var(--setup); }

.cards { display: grid; grid-template-columns: 1fr; gap: 20px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
        padding: 14px 14px 8px; }
.card h3 { margin: 0 0 2px; font-size: 15px; font-weight: 650; }
.card .meta { color: var(--text-2); font-size: 12.5px; margin-bottom: 10px; }
.card img { width: 100%; height: auto; display: block; border-radius: 6px; }

.empty { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
         padding: 28px; text-align: center; color: var(--text-2); }
footer { margin-top: 44px; padding-top: 16px; border-top: 1px solid var(--border);
         color: var(--text-3); font-size: 12px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
"""

TABLE_COLUMNS: Sequence[tuple[str, str, str]] = (
    # (dataframe column, header, format spec)
    ("symbol", "Symbol", "sym"),
    ("state", "State", "badge"),
    ("score", "Score", "{:.0f}"),
    ("close", "Close", "{:,.2f}"),
    ("pivot", "Pivot", "{:,.2f}"),
    ("entry", "Entry", "{:,.2f}"),
    ("stop", "Stop", "{:,.2f}"),
    ("risk_pct", "Risk", "{:.1%}"),
    ("shares", "Shares", "{:,.0f}"),
    ("position_value", "Position", "${:,.0f}"),
    ("target_2r", "2R", "{:,.2f}"),
    ("adr20", "ADR", "{:.1f}%"),
    ("impulse_gain", "Prior move", "{:.0%}"),
    ("base_len", "Base", "{:.0f}d"),
    ("base_depth", "Depth", "{:.1%}"),
    ("dist_from_pivot", "To pivot", "{:.1%}"),
)


def _fmt(value: Any, spec: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if spec in ("sym", "badge"):
        return html.escape(str(value))
    try:
        return spec.format(value)
    except (ValueError, TypeError):
        return html.escape(str(value))


def _tile(label: str, value: Any, note: str = "") -> str:
    note_html = f'<div class="note">{html.escape(note)}</div>' if note else ""
    return (
        f'<div class="tile"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(str(value))}</div>{note_html}</div>'
    )


def _embed(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode("ascii")


def build_report(
    candidates: pd.DataFrame,
    out_path: str | Path,
    charts: dict[str, str | Path] | None = None,
    update_stats: dict[str, Any] | None = None,
    coverage: dict[str, Any] | None = None,
    universe_size: int = 0,
    config_note: str = "",
    title: str = "Breakout scan",
) -> Path:
    charts = charts or {}
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    as_of = "—"
    if len(candidates) and "date" in candidates:
        as_of = str(pd.Timestamp(candidates["date"].max()).date())
    elif coverage and coverage.get("latest_date"):
        as_of = str(coverage["latest_date"])

    n_breakout = int((candidates.get("state") == "breakout").sum()) if len(candidates) else 0
    n_setup = int((candidates.get("state") == "setup").sum()) if len(candidates) else 0

    tiles = [
        _tile("Candidates", len(candidates), f"from {universe_size:,} symbols" if universe_size else ""),
        _tile("Breaking out", n_breakout, "cleared the pivot today"),
        _tile("Coiled setups", n_setup, "within striking distance"),
    ]
    if len(candidates) and "score" in candidates:
        tiles.append(_tile("Top score", f"{candidates['score'].max():.0f}", "out of 100"))
    if coverage:
        tiles.append(
            _tile(
                "Symbols tracked",
                f"{coverage.get('symbols_healthy', 0):,}",
                f"{coverage.get('total_rows', 0):,} bars on disk",
            )
        )
    if update_stats:
        tiles.append(
            _tile(
                "New bars today",
                f"{update_stats.get('new_rows', 0):,}",
                f"{update_stats.get('failed', 0)} fetch failures",
            )
        )

    # ---- table --------------------------------------------------------------
    if len(candidates):
        cols = [(c, h, f) for c, h, f in TABLE_COLUMNS if c in candidates.columns]
        head = "".join(
            f'<th class="{"sym" if f in ("sym", "badge") else ""}">{html.escape(h)}</th>' for _, h, f in cols
        )
        body_rows = []
        for _, row in candidates.iterrows():
            cells = []
            for col, _h, spec in cols:
                text = _fmt(row.get(col), spec)
                if spec == "badge":
                    cls = "breakout" if str(row.get(col)) == "breakout" else "setup"
                    cells.append(f'<td class="sym"><span class="badge {cls}">{text}</span></td>')
                elif spec == "sym":
                    cells.append(f'<td class="sym">{text}</td>')
                else:
                    cells.append(f"<td>{text}</td>")
            body_rows.append("<tr>" + "".join(cells) + "</tr>")
        table = f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'
    else:
        table = '<div class="empty">No candidates passed the filters today. That is a normal result in a weak tape.</div>'

    # ---- chart cards --------------------------------------------------------
    cards = []
    for _, row in candidates.iterrows():
        key = str(row.get("symbol", ""))
        path = charts.get(key)
        if not path:
            continue
        data = _embed(path)
        if not data:
            continue
        state = str(row.get("state", ""))
        cls = "breakout" if state == "breakout" else "setup"
        # The chart's own title carries the pattern stats; the card carries the
        # trade plan, so the two lines complement rather than repeat each other.
        meta = (
            f"buy above {_fmt(row.get('entry'), '{:,.2f}')} · "
            f"stop {_fmt(row.get('stop'), '{:,.2f}')} ({_fmt(row.get('risk_pct'), '{:.1%}')}) · "
            f"{_fmt(row.get('shares'), '{:,.0f}')} shares = {_fmt(row.get('position_value'), '${:,.0f}')} · "
            f"2R {_fmt(row.get('target_2r'), '{:,.2f}')}"
        )
        cards.append(
            f'<div class="card"><h3>{html.escape(key)} '
            f'<span class="badge {cls}">{html.escape(state)}</span> '
            f'<span style="color:var(--text-3);font-weight:500">score {_fmt(row.get("score"), "{:.0f}")}</span></h3>'
            f'<div class="meta">{meta}</div>'
            f'<img alt="{html.escape(key)} daily chart" src="data:image/png;base64,{data}"></div>'
        )

    charts_html = (
        f'<h2>Charts</h2><div class="cards">{"".join(cards)}</div>' if cards else ""
    )
    errors_html = ""
    if update_stats and update_stats.get("error_sample"):
        items = "".join(
            f"<li><code>{html.escape(str(s))}</code> — {html.escape(str(e)[:140])}</li>"
            for s, e in update_stats["error_sample"]
        )
        errors_html = f"<h2>Fetch problems</h2><ul style='color:var(--text-2);font-size:13px'>{items}</ul>"

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — {as_of}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<h1>{html.escape(title)}</h1>
<div class="sub">Data as of <strong>{html.escape(as_of)}</strong> · generated {html.escape(generated)}{" · " + html.escape(config_note) if config_note else ""}</div>
<div class="tiles">{"".join(tiles)}</div>
<h2>Candidates</h2>
{table}
{charts_html}
{errors_html}
<footer>
Qullamaggie breakout setup · entry is the opening-range high on the trigger day, not the daily pivot ·
stop capped at 1&times;ADR · sell 1/3&ndash;1/2 into strength after 3&ndash;5 days, stop to breakeven, trail the rest on the 10/20&nbsp;MA.<br>
Generated by <code>qscan</code>. Screening output, not investment advice.
</footer>
</div></body></html>"""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out
