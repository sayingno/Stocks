@echo off
REM ---------------------------------------------------------------------------
REM Step 1 of the plan: load a bulk vendor dump into the price database and
REM print the dataset inventory - what tickers we have, how far back, how
REM liquid, and whether anything looks split-unadjusted.
REM
REM Usage (quote the path - it has spaces):
REM   scripts\windows_pipeline.bat "D:\Claude\data\Ajusted Stock Data till 2025"
REM
REM Re-running is safe: ingest merges into the existing database rather than
REM replacing it, so you can point it at a second dump later.
REM ---------------------------------------------------------------------------
setlocal

cd /d "%~dp0.."

if "%~1"=="" (
    echo Usage: scripts\windows_pipeline.bat "PATH\TO\DATA" [more paths...]
    echo.
    echo   PATH may be a zip, a folder of TICKER.csv files, or one long CSV.
    exit /b 1
)

set VPY=.venv\Scripts\python.exe
if not exist "%VPY%" (
    echo No virtualenv found. Run scripts\windows_setup.bat first.
    exit /b 1
)

echo === 1/2  Ingesting into the price database at .\data ===
"%VPY%" -m qscan.cli ingest %* --repo data || (
    echo.
    echo Ingest failed or found no usable symbols. Send me the output above.
    exit /b 1
)

echo.
echo === 2/2  Dataset inventory ===
"%VPY%" -m qscan.cli summary --repo data --offline --out out --verbose || exit /b 1

echo.
echo ===========================================================
echo  Wrote:  out\dataset_summary.json    (overview + warnings)
echo          out\dataset_symbols.csv     (one row per ticker)
echo          data\universe.txt           (every symbol ingested)
echo  Send me out\dataset_summary.json and I will set the
echo  thresholds and wire up the earnings-gap step.
echo ===========================================================
endlocal
