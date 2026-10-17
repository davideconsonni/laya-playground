#!/usr/bin/env bash
# Start the Laya playground (Linux, macOS, WSL and Git Bash).
#
#   ./run.sh                    -> GPU (cuda), port 8000
#   ./run.sh --device cpu       -> force CPU
#   ./run.sh --port 8080
#   ./run.sh --help             -> all the server options
#
# Every argument is forwarded to src/server.py, so any optional flag works here.
set -euo pipefail

# Run from the project root, wherever the script is called from.
cd "$(dirname "$0")"

uv sync --python 3.12

echo "Starting: uv run python src/server.py --warm $*"
exec uv run python src/server.py --warm "$@"
