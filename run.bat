@echo off
REM Start the Laya playground (Windows cmd.exe).
REM   run.bat                  -> GPU (cuda), port 8000
REM   run.bat --device cpu     -> force CPU
REM   run.bat --port 8080
REM   run.bat --help           -> all the server options
REM Every argument is forwarded to src/server.py.
setlocal
cd /d "%~dp0"

uv sync --python 3.12 || exit /b 1

echo Starting: uv run python src/server.py --warm %*
uv run python src/server.py --warm %*

endlocal
