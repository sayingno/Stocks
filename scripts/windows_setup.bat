@echo off
REM ---------------------------------------------------------------------------
REM One-time setup on Windows: pick a Python, build a venv, install deps,
REM run the test suite. Safe to re-run - it just refreshes what is there.
REM
REM Double-click it, or from a prompt:  scripts\windows_setup.bat
REM
REM The "cd /d" below is the whole reason this file exists. In cmd, plain
REM "cd D:\some\path" from a C: prompt sets D's working directory but does NOT
REM switch drives, and prints nothing while failing to do so. /d switches.
REM ---------------------------------------------------------------------------
setlocal

cd /d "%~dp0.."
echo Repo root: %CD%
echo.

REM --- find an interpreter that has wheels for pandas/numpy -------------------
REM Newest Python is often ahead of the scientific stack. Prefer a known-good
REM version and fall back, rather than letting pip try to compile numpy.
set PYEXE=
for %%V in (3.12 3.13 3.11 3.14) do (
    if not defined PYEXE (
        py -%%V -c "import sys" >nul 2>&1 && set PYEXE=py -%%V
    )
)
if not defined PYEXE (
    py -c "import sys" >nul 2>&1 && set PYEXE=py
)
if not defined PYEXE (
    echo ERROR: no Python found. Install from https://python.org and re-run.
    exit /b 1
)

echo Using: %PYEXE%
%PYEXE% --version
echo.

REM --- venv ------------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtualenv in .venv ...
    %PYEXE% -m venv .venv || (echo ERROR: venv creation failed & exit /b 1)
)
set VPY=.venv\Scripts\python.exe

echo Installing dependencies ...
"%VPY%" -m pip install --upgrade pip --quiet
"%VPY%" -m pip install -r requirements.txt || (
    echo.
    echo ERROR: dependency install failed.
    echo If it tried to COMPILE numpy or pandas, this Python is too new for
    echo prebuilt wheels. Install Python 3.12, delete the .venv folder, re-run.
    exit /b 1
)
echo.

REM --- prove it works --------------------------------------------------------
echo Running the test suite ...
"%VPY%" -m unittest discover -s tests -t . || (
    echo.
    echo Tests failed - stop here and send me the output.
    exit /b 1
)

echo.
echo ===========================================================
echo  Setup complete.
echo  Next:  scripts\windows_pipeline.bat "D:\Claude\data\Ajusted Stock Data till 2025"
echo ===========================================================
endlocal
