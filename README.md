# qscan — a scanner for Qullamaggie's setup #1 (the Breakout)

Kristjan Kullamägi ("Qullamaggie") describes three setups in
[*3 TIMELESS setups that have made me TENS OF MILLIONS*](https://qullamaggie.com/my-3-timeless-setups-that-have-made-me-tens-of-millions/).
This repo implements the **first** one — the Breakout, also called the
continuation or flag breakout — as a scanner you can run daily, and as a
historical sweep that finds every past instance and draws the chart.

---

## 1. What the setup actually is

One sentence: **a stock that already made a big move up, then went quiet and
tight while riding its moving averages, and is now pressing against the high of
that quiet period.**

The logic is continuation, not reversal. The big move is the evidence that
something real is happening — new product, new numbers, a new narrative,
institutions accumulating. The consolidation is those buyers being patient while
weak holders leave. The breakout is the moment demand overwhelms the remaining
supply, and it is the only moment you actually risk money.

### The five components

| # | Component | What it looks like |
|---|---|---|
| 1 | **The prior move** | +30–100%+ in the last 1–3 months, usually built over days-to-weeks, not a slow grind. This is the filter that leaves you with a few hundred names out of thousands. |
| 2 | **The consolidation** | Anywhere from a 3-day tight flag to a 2-month base. Orderly: range contracts, lows hold up or step higher, volume dries out. Sloppy, deep, high-volume chop is not a base. |
| 3 | **MA "surfing"** | Through the base, price hugs a rising 10-day or 20-day MA (slower names, the 50-day). The MAs should be stacked — 10 above 20 above 50 — and you never buy below the 50. |
| 4 | **The trigger** | On breakout day you buy the **opening range high** — the high of the first 1-, 5-, or 60-minute candle. Not the daily pivot, not a limit order overnight. |
| 5 | **ADR** | Average Daily Range. The stock must move enough that a few days of follow-through pays multiples of your risk. He wants roughly 5%+; below ~3% it can't pay you fast enough. |

### The risk rules — the part that actually makes the money

- **Stop = the low of the day**, and **never wider than 1× ADR**. If the low of
  the day is 9% away on a 5% ADR stock, the stop goes at 5%, not 9%. This single
  constraint keeps the arithmetic from breaking.
- **Sell 1/3 to 1/2 into the first burst of strength**, typically 3–5 days after
  entry (or sooner if it prints 2–3R). Move the stop on the rest to breakeven.
- **Trail the remainder on the 10- or 20-day MA** — 10 for fast movers and if
  you're new, 20 for slower names. Exit on the first *close* below it.
- Position size falls out of the stop, not out of conviction:
  `shares = (account × risk%) ÷ (entry − stop)`.

### Expectations

Roughly **a quarter to a third of these work**. That is the design, not a flaw.
You take many small losses at −1R or less and let two or three winners per year
run for 10–20R+ on the trailing MA. If you cut winners at +2R to feel good about
your hit rate, the whole thing inverts and stops working.

> This repo implements a well-known public method. It is not investment advice,
> and a scanner that finds the pattern is not the same as a strategy that makes
> money — the exits and the sizing are where the edge lives.

---

## 2. How the scanner encodes it

Every qualitative phrase above became a measured number in
[`qscan/setup_breakout.py`](qscan/setup_breakout.py). Gates run in order and the
first failure is recorded, so a name that doesn't show up can tell you why.

| Gate | Measured as | Default |
|---|---|---|
| `price` / `liquidity` | close, 20d avg `close × volume` | ≥ $5, ≥ $3M |
| `adr` | mean of `high/low − 1` over 20 bars | 3.5%–25% |
| `no_prior_move` | return over 21 / 63 / 126 bars vs thresholds | +30% / +50% / +100%, any one |
| `base_too_short/long` | bars from the pivot high to yesterday | 3–60 |
| `base_too_deep` | `(base high − base low) / base high` | ≤ 35% |
| `no_contraction` | mean daily range, back half of base ÷ front half | ≤ 0.95 |
| `lows_breaking_down` | least-squares slope of the base's lows | ≥ −0.75%/bar |
| `volume_not_drying_up` | back-half base volume ÷ impulse-leg volume | ≤ 1.0 |
| `stale_move` | bars since the pivot high | ≤ 60 |
| `below_slow_ma` / `ma_not_stacked` | close vs 50MA; 10 > 20 > 50 | required |
| `mid_ma_not_rising` | 20MA slope | > 0 |
| `not_surfing_ma` | mean `|close − MA| / close` over the base, best of 10/20/50 | ≤ 12% |
| `far_from_pivot` | `(pivot − close) / pivot` | ≤ 10% |

Survivors are labelled **`breakout`** (cleared the pivot today on volume) or
**`setup`** (coiled and within striking distance), scored 0–100 on a weighted
blend of momentum, tightness, contraction, volume dry-up, MA-hug, proximity and
ADR, and given a full trade plan: entry, ADR-capped stop, share count, 2R/3R
targets.

Three presets — `--preset relaxed | default | strict` — and every single
threshold is overridable from the command line.

---

## 3. Install

```bash
pip install -r requirements.txt
```

## 4. Run it every day automatically

`daily` is the scheduled job: refresh the price database, apply the filters,
render the charts, write an HTML report. **See [docs/AUTOMATION.md](docs/AUTOMATION.md)
for the full setup** — this is the short version.

```bash
# 1. get a real universe (needs network to nasdaqtrader.com)
python -c "from qscan.universe import download_us_listings as d; \
           open('universes/us_all.txt','w').write('\n'.join(d()))"

# 2. seed the price database once — slow, interruptible, resumable
python -m qscan update --universe us_all --repo data --workers 6 --verbose

# 3. from then on, this is the whole daily job (a couple of minutes)
python -m qscan daily --universe us_all --repo data --charts 25
```

Then schedule step 3 with one of:

| Platform | How | File |
|---|---|---|
| Linux / WSL | cron → `30 17 * * 1-5 /path/Stocks/scripts/daily_scan.sh` | `scripts/daily_scan.sh` |
| macOS | launchd (catches up after sleep; cron does not) | `scripts/com.qscan.daily.plist` |
| No machine of your own | GitHub Actions, 22:00 UTC weekdays | `.github/workflows/daily-scan.yml` |
| Windows | Task Scheduler | snippet in the automation doc |

**Output** — `out/latest.html` is a single self-contained file (charts embedded
as base64, nothing loaded from the network, adapts to light/dark) containing
stat tiles, a table of every candidate with its full trade plan, and the
annotated chart of each. Alongside it: `out/scan_<date>.csv` and
`out/charts/<date>/*.png`.

The refresh is genuinely incremental — the first build downloads years of
history, every run after fetches only the new bars. It detects splits (a 2:1
split rewrites adjusted history, so the symbol is re-downloaded rather than
appended to), benches tickers that fail repeatedly, and writes atomically so an
interrupted run can't corrupt anything.

For automation, `daily` exits non-zero when more than half the fetches fail or
the newest bar is more than 5 days old, so a broken feed reports as a failure
rather than as a quiet "0 candidates". Zero candidates on healthy data is a
success — that's a normal result in a corrective market.

Tuning what comes back:

```bash
python -m qscan daily --universe us_all --repo data --state breakout   # triggering today
python -m qscan daily --universe us_all --repo data --preset strict    # ADR 5%+, $20M+
python -m qscan daily --universe us_all --repo data --preset relaxed   # thin tape
python -m qscan daily --universe us_all --repo data --account-size 250000 --risk-pct 0.0075
python -m qscan daily --universe us_all --repo data --set max_base_depth=0.25 --min-score 65
```

`scan` is the same filter without the refresh or the report — handy for
one-off exploring against the ad-hoc cache.

## 5. Finding the past setups and their charts

This is the historical sweep. It walks every bar of every symbol, finds each
distinct setup, simulates the trade under Qullamaggie's management rules, and
renders the charts.

```bash
python -m qscan history --universe us_all --years 8 --charts 60 --contact-sheet
```

You get:

- `out/history_8y.csv` — one row per setup with every measured feature, the
  trade plan, forward returns at 5/10/20/60 days, MFE/MAE, and the simulated
  R multiple and exit reason.
- `out/history_stats.json` — win rate, average R, expectancy, total R.
- `out/charts/*.png` — the annotated chart of each setup: candles, 10/20/50 MAs,
  the consolidation shaded in, the pivot line, entry and stop, and the fill/exit
  markers showing what happened next.
- `out/contact_sheet.png` — all of them on one page, for pattern-soaking.

Sorting `history_*.csv` by `r_multiple` descending gives you the big winners;
filtering on `outcome == "stopped"` gives you the failures, which are the more
instructive study.

## 6. Diagnosing one name

```bash
python -m qscan explain --symbol NVDA --date 2023-05-24 --chart
```

Prints `PASS`, or `FAIL at gate: <name>` plus every measured value, so you can
see whether the base was 2% too deep or the volume never dried up.

## 7. Data sources

| Provider | Flag | Notes |
|---|---|---|
| yfinance | `--provider yfinance` | default; free, rate-limited on big universes |
| Stooq | `--provider stooq` | free CSV, no key, no library |
| Tiingo | `--provider tiingo` | set `TIINGO_API_KEY`; best free history |
| local CSVs | `--csv-dir DIR` | a directory of `TICKER.csv`; no network at all |

Add `--offline` to work purely from cache.

## 8. Smoke test without network

```bash
python tools/make_demo_data.py --out data/demo
python -m qscan daily   --csv-dir data/demo --repo data/db --universe data/demo/universe.txt --charts 5
python -m qscan history --csv-dir data/demo --universe data/demo/universe.txt --offline --charts 6
open out/latest.html
```

Synthetic tickers with known shapes (`GOODBRK`, `FAILBRK`, `COILED`, `DEEPBASE`,
`DOWNTRND`, `UNDER50`, `THINVOL`) exercise the whole pipeline.

## 9. Tests

```bash
python -m unittest discover -s tests -t .
```

70 tests over hand-built OHLCV series, no network required:

- **detector** — the textbook setup passes; a coiled base classifies as `setup`
  not `breakout`; downtrends, deep bases, sub-50MA and illiquid names are
  rejected at the named gate.
- **risk** — the stop never exceeds 1 ADR; sizing respects the risk budget and
  the position cap.
- **outcomes** — the simulator returns ≈ −1R on a designed failure and multiple
  R on a designed winner.
- **database** — a second run is a no-op; new bars append without refetching
  history; a simulated 2:1 split triggers a full re-download while rounding
  noise does not; repeated failures bench a symbol and `--force` un-benches it.
- **pipeline** — `daily` writes report, CSV and charts; a broken feed and stale
  data each exit non-zero; zero candidates on healthy data exits zero.

## 10. Layout

```
qscan/
  config.py           every threshold, plus relaxed/default/strict presets
  indicators.py       MAs, ADR, ATR, dollar volume, slopes
  setup_breakout.py   the detector: gates, scoring, trade plan
  outcomes.py         forward returns + the trade simulator
  repository.py       the incremental price database (splits, retries, benching)
  data.py             providers (yfinance/stooq/tiingo/csv) + disk cache
  universe.py         ticker lists
  charts.py           annotated candlestick renderer
  report.py           self-contained HTML digest
  cli.py              update / daily / scan / history / explain
scripts/
  daily_scan.sh             cron & launchd wrapper: venv, logging, locking
  com.qscan.daily.plist     macOS launchd job
.github/workflows/daily-scan.yml
docs/AUTOMATION.md
tools/make_demo_data.py
tests/
```

## 11. Known limits

- **Daily bars can't see the opening range.** His real entry is the 1/5/60-minute
  opening range high. The scanner gives you the daily pivot and an ADR-capped
  stop; the intraday trigger is still yours to execute.
- **Survivorship bias.** A universe built from today's listings excludes
  delisted names, which flatters historical results. For honest backtests use a
  point-in-time universe from a paid provider.
- **No gap/liquidity modelling** beyond filling stop-gaps at the open.
- The scanner finds *the pattern*. Qullamaggie also filters by fundamentals,
  sector leadership and general market conditions — none of that is here.
