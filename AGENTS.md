# AGENTS.md

Guidance for coding agents (and humans) working in this repository. See
[README.md](README.md) for usage and [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit
together.

## What this project is

A small FastAPI app around [Laya](https://github.com/NandhaKishorM/laya) that answers one
question: **given a problem and a list of candidate solutions, which is the best?** It exposes a
single-page UI and a handful of `/api/*` endpoints. There is no database, no background queue and
no build step for the frontend.

## Language conventions (important)

- **Code, comments, docstrings, log messages, CLI output and config files are in English.**
  This includes `src/*.py`, the `Dockerfile`, `pyproject.toml`, `.env.example`, `.gitignore`,
  `.dockerignore` and the launcher scripts.
- **The frontend and everything the user reads in it stays in Italian.**
  `src/static/index.html` (markup, JS/CSS comments and all visible text) is Italian, and the
  strings the API returns to it that are **shown in the UI** are Italian too — e.g. the stability
  `verdict`/`note`, the `notes` banners, and the `detail` of errors such as
  `"il problema e' vuoto"` or `"servono almeno due soluzioni candidate"`.
- Everything else that is not user-facing (logs, `argparse` help, `warmup.py` output) is English.

Rule of thumb: if it can end up rendered in the UI, keep it in Italian; otherwise write English.

## Commands

```bash
uv sync --python 3.12                 # install/refresh dependencies
./run.sh --device cpu                 # Linux/macOS/WSL/Git Bash (run.bat for cmd, run.ps1 for pwsh)
uv run python src/server.py --warm    # run the server directly
uv run python src/warmup.py --device both --full   # smoke test: predict, tournament, stability, determinism
uv run python src/warmup.py --device cuda --determinism   # same input twice -> identical output
```

Docker: see the README. There is no test suite; `warmup.py --full` **is** the smoke test and needs
the checkpoints downloaded (minutes on first run).

## Where things live

- `src/engine.py` — device pool, question building, `solve` / `stability` / `reliability` /
  `status`, budgets. **The place to change behaviour.** It must not import `torch`/`laya` at
  module level.
- `src/server.py` — FastAPI wiring: middleware, request models, endpoints, CLI entry point, static
  mount for the UI.
- `src/static/index.html` — the entire UI in one file. No framework, no CDN.
- `src/warmup.py` — command-line warmup/smoke test.
- `Dockerfile`, `run.sh`, `run.bat`, `run.ps1` — packaging and launchers.

## Invariants to preserve

- **One process.** Every worker holds a full copy of the weights (`workers=1`); scale vertically.
- **Heavy imports stay lazy.** `torch`, `transformers` and `laya` are imported inside
  `engine.preimport()`, never at module scope, and the first import is serialised by
  `_IMPORT_LOCK` to avoid the transformers lazy-names race.
- **The UI is mounted last** in `server.py`, so `/api/*` and `/health` take precedence over `/`.
- **Option keys are opaque** (`s00`, `s01`, ...): checkpoints follow labels like `true`/`false`
  literally, so never use option text as a key.
- **Weights live in the project** (`.hf-cache/`, or `/models` in the container) so they can be
  deleted in one go. `HF_HOME`/`HF_HUB_CACHE` are set before anything imports `huggingface_hub`.
- **Determinism**: same input + same parameters -> same output. The stability rotations are random
  but seeded (`STABILITY_SEED`), so a second run reproduces the orders and the numbers;
  `warmup.py --determinism` (also under `--full`) checks it.
- **User-facing strings stay Italian** (see the language conventions above).

## Configuration

Environment variables and their defaults are documented in `.env.example` and in
[ARCHITECTURE.md](ARCHITECTURE.md). `.env` is git-ignored and is passed to the container at runtime
with `--env-file`, never baked into the image.
