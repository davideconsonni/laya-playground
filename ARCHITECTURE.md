# Architecture

How the pieces of the Laya playground fit together. For *what* Laya is and how to use the UI,
see [README.md](README.md).

## Purpose

One job: **given a problem (the state) and a list of candidate solutions, pick the best one** and
expose how sure the model is. This is a thin, honest shell around Laya's `Router`:

- no text generation, no explanation, no agent loop;
- one non-autoregressive forward pass per question;
- everything the UI shows comes straight from Laya's output plus locally computed metadata
  (latency, budgets, tags).

## High-level view

```
 browser (src/static/index.html)
        |  fetch /api/*
        v
 FastAPI server (src/server.py)  --  validates, fans out per device, shapes errors
        |
        v
 engine.py   --  device pool, question building, strategies, stability, status
        |
        v
 laya.Router  --  routes the text to a checkpoint and runs the forward pass
        |
        v
 checkpoints on disk (.hf-cache/)  --  downloaded from the HF Hub on first use
```

## Components

| File | Responsibility |
|---|---|
| `src/static/index.html` | The whole UI: markup, CSS and JS in one file, no build step, no CDN. Talks to `/api/*` and persists input/last result in `localStorage`. |
| `src/server.py` | FastAPI app: API-key middleware, request models, the `/api/*` endpoints, per-device fan-out for `device: "both"`, static mount for the UI, CLI entry point. |
| `src/engine.py` | Everything else: loading `.env`, the `DevicePool` (one `Router` per device), question construction, the `solve` / `stability` / `status` functions, token-budget heuristics. |
| `src/warmup.py` | Command-line warmup and smoke test: builds the Router, runs a predict cold and warm, and with `--full` also the tournament and stability. |
| `Dockerfile` + launchers | Packaging. `run.sh` (Linux/macOS/WSL/Git Bash), `run.bat` (Windows cmd), `run.ps1` (PowerShell) all `uv sync` then run `src/server.py --warm`. |

`engine.py` deliberately has **no module-level import of `torch` or `laya`**: both are heavy, and
the server imports this module at startup (and `/api/status` must answer while the environment is
still being probed). Heavy imports happen inside `preimport()`, once, under a lock.

## The device pool

`DevicePool` keeps `Router` objects alive, keyed by normalised device name
(`cuda`, `cpu`, `mps`, `xpu`, `auto`; `gpu`/`cu` are aliases for `cuda`).

- A `Router` decides its backend once, at construction. The CPU/GPU toggle in the UI therefore
  needs **two** instances; keeping both alive is what makes switching instant and what makes the
  CPU-vs-GPU comparison possible.
- Construction is lazy and happens **outside** the pool lock (it can download weights and take
  minutes), so toggling the other device is not blocked meanwhile.
- A failed construction is remembered as an error and re-raised on later calls, so the UI shows
  the same message instead of retrying a doomed download.
- `unload()` frees the weights but keeps the `Router`; `loaded()` reports which checkpoints are
  resident.

### Warmup

`start_warmup(device, checkpoints)` returns immediately and does the load in a daemon thread,
recording state (`loading` -> `ready`/`error`) that `/api/status` exposes. The default checkpoint
is only `multilingual`: that is where routing sends non-English Latin text, and preloading
`english` too would cost ~1.7 GB that nobody would use. `english` still loads lazily on first use.

### The transformers import race

`transformers` 5.x resolves names lazily: two threads starting the import at the same moment can
see a half-built module (`cannot import name 'AutoTokenizer' ...`). `preimport()` serialises the
first import (and forces the lazy `Auto*` names) under `_IMPORT_LOCK`, called both from the server
startup and from `DevicePool.get` before the first `Router`.

## The question model

The problem is the **state**; each candidate solution becomes an option of a `choice` question:

```python
{
  "best": {
    "type": "choice",
    "instructions": "<English by default, user-editable>",
    "criteria": {"s00": "<solution 1>", "s01": "<solution 2>", ...},
  }
}
```

Option keys are neutral (`s00`, `s01`, ...) on purpose: checkpoints follow labels like
`true`/`false` literally, so the option *text* is unsafe as a key. `with_quality` adds a `score`
question ("how good is the best solution?") and `with_complete` a `noul` question ("does any
candidate fully solve it?").

### Token budgets

In Laya, options share a budget (`head_max_len`) and an option under ~4 tokens becomes
indistinguishable from the rest. `suggest_budgets(n)` widens the budgets with the number of
options (capped), and the UI mirrors the same heuristic when "automatic budget" is on.

## Strategies

| Strategy | What it does | When it is chosen |
|---|---|---|
| `simple` | One `choice` question with all the options. | `auto` with <= 20 solutions, or forced. |
| `tournament` | Group elimination (`group_size`, default 16), ~256 solutions max. | `auto` with > 20 solutions, or forced. |
| `long` | Reads a problem longer than the window in windows instead of truncating. `predict_long` takes neither the budgets nor `min_confidence`. | Forced only. |
| `stability` | Not a strategy: a separate endpoint that rotates `option_order` (see below). | When the UI checkbox is on. |

`solve()` also emits human-readable **notes** (approximate truncation warning, window count,
"too many solutions for a tournament") that the UI renders as banners.

## Stability

`stability()` asks the same question `k` times with the option order rotated
(`option_order = [i+r mod n]`), all in **a single forward pass**, and reports:

- the **mediated** choice and the probabilities averaged over the rotations;
- the per-rotation choices and confidences;
- an **agreement** score and a verdict (`stabile` / `abbastanza stabile` / `instabile` /
  `molto instabile`).

It exists because Laya has a documented **position bias**: the same option in different slots
receives different logits. Mediating over rotations is more stable (not more calibrated) than a
single answer.

## Concurrency and state

- **Single worker by design.** Every process keeps the weights in memory, so N workers = N copies
  of the model. Scale vertically, not with `--workers`.
- The pool, the warmup state and the torch info cache are process-local and guarded by locks where
  writers exist.
- The server holds **no per-request state**: each request carries everything it needs, which is
  what makes `/api/solve` horizontally replaceable with a load balancer (at the cost of one model
  copy per replica).

## Endpoints

| Endpoint | Notes |
|---|---|
| `GET /api/status` | Environment + pool state + `auth_required`. Open unless a key is set (it is under `/api/`, so it **does** require the key if `LAYA_API_KEY` is set). |
| `GET /health` | Always open; used by the container `HEALTHCHECK` and by hosting platforms. |
| `POST /api/solve` | Validates, then one call to `engine.solve` per target device. `device: "both"` returns both results plus a per-device `errors` map. |
| `POST /api/stability` | One call to `engine.stability`. |
| `POST /api/warmup` | Starts background preloads and returns the state. |
| `POST /api/unload` | Frees weights, returns `{freed, loaded}`. |

### API key

If `LAYA_API_KEY` is set, the middleware requires it on every `/api/*` route (via `X-Api-Key`,
`Authorization: Bearer` or `?key=`), compared with `hmac.compare_digest`. `/` and `/health` stay
open. The UI reads `?key=` once and keeps it in `localStorage`.

## Frontend

`src/static/index.html` is a single self-contained page:

- **No build, no CDN, no framework.** Vanilla JS, one file, mounted by the server at `/` (mounted
  **last** so `/api/*` takes precedence).
- Two panes: input (problem, solutions, device, strategy, advanced options) and result cards.
- Result cards mirror the API payload: badges (device, checkpoint, strategy, latency, gates),
  the ranked options with probability bars, tournament/finalists metadata, extra `score`/`noul`
  blocks, notes, and a raw JSON `<details>`.
- **Persistence**: input and the last result live in `localStorage`
  (`layaPlayground.v1` / `layaPlayground.lastResult.v1`), both versioned; a payload from an older
  version is ignored.
- The UI text is **in Italian** (see [AGENTS.md](AGENTS.md) for the language convention).

## Configuration

Environment variables (loaded from `.env` by `engine._load_dotenv`, overridable by the real
environment):

| Variable | Meaning |
|---|---|
| `HF_TOKEN` | Optional; the checkpoints are public. Only for rate limits / private checkpoints. |
| `LAYA_HF_CACHE` | Weights cache, relative to the project root (default `.hf-cache`, container `/models`). |
| `LAYA_HOST`, `LAYA_PORT` | Bind address and port. |
| `LAYA_DEVICE` | Default device: `cuda` / `cpu` / `mps` / `xpu` / `auto`. |
| `LAYA_API_KEY` | Empty = no auth (local use). Set it if you expose the service. |
| `LAYA_THREADS` | Cap on torch CPU threads; otherwise derived from the cgroup limit/core count, capped at 16. |
| `LAYA_WARM_MODEL` | Checkpoints preloaded by `--warm` (comma-separated); default `multilingual`. |

`.env` is **not** committed and **not** copied into the image; the container gets it at runtime via
`--env-file`.

## Packaging and deployment

- The `Dockerfile` builds CPU (default) or CUDA images that differ **only** in the PyTorch index
  (`cpu` vs `cu130`). The Linux cu130 wheel carries the CUDA runtime, so the `nvidia/cuda` base is
  not needed at build time.
- `pyproject.toml` pins the cu130 index for local `uv sync`; that is why the Dockerfile installs
  torch with `uv pip` explicitly instead of `uv sync`.
- `BAKE=1` downloads the weights at build time for platforms without a persistent volume.
- The container sets `LAYA_HOST=0.0.0.0`, `LAYA_DEVICE=cpu` and `LAYA_HF_CACHE=/models` (the mount
  point).

## Failure modes and guards

- **No CUDA** -> `--warm` logs a warning and skips the CUDA preload instead of failing.
- **Heavy first request** -> warmup runs in the background and the UI shows a spinner; the pool
  never blocks the whole server on a build.
- **Invalid input** -> a 400 (not a 500), so the same bad request is not retried on two devices.
- **One device fails in `device: "both"`** -> the other result is still returned, with the error in
  the `errors` map.
- **Unhandled exception** -> a global handler returns `{detail: "<Type>: <message>"}` as JSON.
- **Exposed without a key** -> the server logs a clear warning at startup.
