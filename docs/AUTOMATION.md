# Running the scan every day automatically

The scheduled job is one command:

```bash
python -m qscan daily --universe us_all --repo data --out out --charts 25
```

It does four things in order:

1. **Refresh** — brings the local price database current. First run downloads
   full history; every run after that fetches only the new bars.
2. **Filter** — applies the breakout gates to the latest bar of every symbol.
3. **Chart** — renders annotated candlesticks for the top N candidates.
4. **Report** — writes `out/report_<date>.html`, copies it to
   `out/latest.html`, and writes the same rows to `out/scan_<date>.csv`.

It prints a JSON summary and exits non-zero on failure, so any scheduler can
tell whether it worked.

---

## First: build the database once

The initial download is the slow part — a few thousand symbols against a free
provider takes a while. Do it once, by hand, before scheduling anything:

```bash
pip install -r requirements.txt

# Grab every US-listed common stock (~6,000 symbols).
python -c "from qscan.universe import download_us_listings as d; \
           open('universes/us_all.txt','w').write('\n'.join(d()))"

# Seed 12 years of history. Interruptible — re-run and it picks up where it left off.
python -m qscan update --universe us_all --repo data --workers 6 --pause 0.1 --verbose
```

Expect roughly 1.5–3 GB on disk and 30–90 minutes for the full US market on
yfinance. Every run after that touches only the last few bars per symbol and
finishes in a couple of minutes.

If you get rate-limited, lower `--workers` and raise `--pause`. Failures are
recorded per symbol; a ticker that fails three times in a row is benched for a
week so it stops costing you a request every morning. `--force` un-benches
everything.

### What the database looks like

```
data/prices/AAPL.csv     full daily history, one file per symbol
data/manifest.json       last_date / last_fetch / failure count per symbol
```

Both are plain text — inspect or hand-edit them freely. Writes are atomic, so
killing a run mid-flight cannot corrupt anything.

### Splits are handled

With adjusted prices a 2:1 split rewrites a stock's entire history. Blindly
appending new rows would leave you with a chart that has a 50% cliff in it.
Each update re-fetches a 10-bar overlap and compares it against what's stored;
if the closes moved by more than 0.2%, that symbol is re-downloaded in full.
Corporate actions repair themselves the next morning without you noticing.

---

## Option A — cron (Linux, macOS, WSL)

`scripts/daily_scan.sh` handles venv activation, logging, log rotation, and a
lock so a slow run never overlaps the next one.

```bash
chmod +x scripts/daily_scan.sh
crontab -e
```

Add — cron uses **local** time, so pick a time after your market's close:

```cron
30 17 * * 1-5 QSCAN_UNIVERSE=us_all /home/you/Stocks/scripts/daily_scan.sh
```

Cron runs with almost no environment. Use absolute paths, and if `python3`
isn't on cron's `PATH`, set `QSCAN_PYTHON=/usr/bin/python3` in the crontab line.

Environment knobs the script reads: `QSCAN_UNIVERSE`, `QSCAN_PRESET`,
`QSCAN_PROVIDER`, `QSCAN_CHARTS`, `QSCAN_WORKERS`, `QSCAN_DATA`, `QSCAN_OUT`,
`QSCAN_LOGS`, `QSCAN_PYTHON`.

Test it before trusting it:

```bash
QSCAN_UNIVERSE=sample ./scripts/daily_scan.sh && cat logs/daily-$(date +%F).log
```

## Option B — launchd (macOS, recommended over cron)

cron silently skips runs while a laptop is asleep; launchd catches up on wake.

```bash
cp scripts/com.qscan.daily.plist ~/Library/LaunchAgents/
# edit the two /Users/YOU/Stocks paths inside first
launchctl load -w ~/Library/LaunchAgents/com.qscan.daily.plist
launchctl start com.qscan.daily     # run once now to check it
```

## Option C — GitHub Actions (no machine of your own)

`.github/workflows/daily-scan.yml` runs at 22:00 UTC on weekdays and uploads
the report as an artifact. The price database is kept in the Actions cache, so
runs stay incremental.

Trigger it by hand from the Actions tab (`Run workflow`) to test. Two caveats:

- **Cache eviction.** GitHub evicts caches unused for 7 days and caps each repo
  at 10 GB. A big universe will occasionally rebuild from scratch — slow, but it
  recovers on its own.
- **Scheduling is best-effort.** Scheduled workflows can be delayed by tens of
  minutes at busy times, and are disabled automatically after 60 days without
  repository activity.

## Option D — Windows Task Scheduler

```powershell
$action  = New-ScheduledTaskAction -Execute "C:\Users\you\Stocks\.venv\Scripts\python.exe" `
           -Argument "-m qscan daily --universe us_all --repo data --out out --charts 25" `
           -WorkingDirectory "C:\Users\you\Stocks"
$trigger = New-ScheduledTaskTrigger -Weekly `
           -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 5:30PM
Register-ScheduledTask -TaskName "qscan daily" -Action $action -Trigger $trigger `
           -Description "Qullamaggie breakout scan"
```

---

## Reading the output

`out/latest.html` is a single self-contained file — charts are embedded as
base64, nothing is loaded from the network. Email it, sync it, or open it a year
later; it renders the same. It adapts to your system light/dark setting.

It contains:

- **Stat tiles** — candidates, how many are breaking out vs coiled, top score,
  database coverage, bars added this morning.
- **A table** — every candidate with its trade plan: pivot, entry, ADR-capped
  stop, risk %, share count, position value, 2R target, plus the pattern
  measurements (prior move, base length, depth, ADR, distance to pivot).
- **Chart cards** — the annotated daily chart for each of the top N.

`out/scan_<date>.csv` is the same data for spreadsheets or your own tooling.

## Tuning what it returns

Too few names in a weak tape, too many in a hot one. Adjust rather than stare at
an empty report:

```bash
# looser: smaller prior moves, deeper bases, lower liquidity floor
python -m qscan daily --universe us_all --repo data --preset relaxed

# tighter: ADR 5%+, $20M+ dollar volume, shallow bases only
python -m qscan daily --universe us_all --repo data --preset strict

# only names actually triggering today
python -m qscan daily --universe us_all --repo data --state breakout

# hand-tune any single threshold
python -m qscan daily --universe us_all --repo data \
  --set max_base_depth=0.22 --set min_adr_pct=4.5 --min-score 65
```

When a name you expected doesn't appear, ask the scanner why:

```bash
python -m qscan explain --symbol NVDA --repo data --chart
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Report is empty every day | Normal in a corrective market. Sanity-check with `--preset relaxed`; if that's also empty, run `explain` on a name you know is basing. |
| `not in the price database` | That symbol was never fetched. Run `qscan update` with the same `--universe`. |
| Lots of fetch failures | Rate limiting. `--workers 3 --pause 0.3`, or switch `--provider stooq`/`tiingo`. |
| Cron produces nothing | Cron has no `PATH` and no venv. Use absolute paths, set `QSCAN_PYTHON`, and read `logs/daily-<date>.log`. |
| Data is a day stale | The provider hadn't posted the close yet. Schedule at least 90 minutes after the bell. |
| A chart has a price cliff | Should self-heal on the next update via split detection. Force it: `rm data/prices/SYM.csv && python -m qscan update --universe SYM --repo data`. |

## What this does not do

It does not place orders, and it cannot see the intraday opening range that
Qullamaggie actually enters on — you get the daily pivot and an ADR-capped stop,
and the trigger stays a manual decision. It also knows nothing about earnings
dates, market regime, or sector leadership, all of which he filters on.
