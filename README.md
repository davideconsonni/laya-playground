# Laya playground

Local playground for [Laya](https://github.com/NandhaKishorM/laya) (`convaiinnovations/laya`)
applied to a single case: **given a problem and a list of candidate solutions, choose the best
solution**.

It serves to see in practice what the model does, with what confidence, with what latency on GPU
or CPU, and how stable the choice is as the order of the options changes.

---

## What Laya is (and what it is not)

Laya is a non-autoregressive **decision engine**, not a generative LLM:

| | |
|---|---|
| **input** | a *state* (here: the problem text) + typed questions (`choice`, `score`, `noul`) |
| **output** | a chosen label, the **probability distribution over each option**, a calibrated confidence |
| **cost** | a single forward pass: tens of ms on GPU |
| **does not** | generate text, **does not explain why**, does not reason step by step |

The "choose among the solutions" case is exactly a `choice` question: each solution becomes an
option. The option keys are neutral (`s00`, `s01`, ...) because the checkpoints follow labels like
`true`/`false` literally instead of the option text.

**Honest caveat from the repo**: the base checkpoints are close to guessing on domain decisions;
the real value comes from fine-tuning on your domain. Use this playground also to understand
*whether* the probabilities order sensibly on your case.

---

## Setup

```bash
cd laya-playground
uv sync --python 3.12
```

`torch` is resolved from the CUDA channel declared in `pyproject.toml` (`cu130`), which on this
machine matches the driver. For a CPU-only install:

```bash
uv pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
```

The `.env` holds `HF_TOKEN` (optional: the checkpoints are public) and the weights cache moved to
`.hf-cache/`, so everything can be deleted in one go.

## Run

```bash
./run.sh                  # Linux / macOS (also WSL and Git Bash)
run.bat                   # Windows cmd.exe
.\run.ps1                 # Windows PowerShell
```

All three sync the dependencies and forward their arguments to `src/server.py`, so the same
options work everywhere:

```bash
./run.sh --device cpu     # force CPU
./run.sh --port 8080
```

Or directly:

```bash
uv run python src/server.py --warm
```

`--warm` preloads the checkpoints in the background at startup (the first predict downloads the
weights from the Hub: a few minutes). In the UI the **Carica modello** ("load model") button does
the same thing, and the pills at the top show when it is ready.

On Linux/macOS make `run.sh` executable once with `chmod +x run.sh`.

## Command-line smoke test

```bash
uv run python src/warmup.py --device both --full
```

Prints device, dtype, cold and warm latency, the tournament on 30 solutions and the stability test.
`--determinism` (implied by `--full`) runs the same input twice and checks the output is identical,
including the seeded rotation orders.

## Using the UI

1. Paste the **problema** ("problem") into the first box.
2. Paste the **soluzioni** ("solutions"), one per line, into the second box (the counter at the top
   shows how many it read).
3. Choose the **device**: `GPU (cuda)`, `CPU`, or `Confronta GPU / CPU` ("compare GPU / CPU") to
   see the two answers side by side with their respective latencies.
4. Choose the **strategia** ("strategy"):
   - `Auto` &mdash; a single question up to 20 solutions, tournament above;
   - `Semplice` ("simple") &mdash; a single `choice` question with all the solutions;
   - `Tournament` &mdash; group elimination with `group_size` (default 16), handles up to ~256
     solutions, at the cost of a few extra forward passes;
   - `Lungo` ("long") &mdash; for problems longer than the context window: reads the whole
     document in windows instead of truncating.
5. Tick **Testa anche la stabilita'** ("also test stability") to repeat the question with the
   option order permuted `k` times (in a single forward pass) and see whether the choice changes.
   The permutations are **random and seeded** (reproducible run to run), up to 32 (and never more
   than the distinct permutations that exist); the result shows the **distribution** of the
   choices across the rotations, not only the average.
6. **Decidi** ("decide") (or Ctrl+Enter). If the reliability flag is available the result also
   carries an **affidabilita'** card (see below).
7. **Esporta per LLM** ("export for LLM") &mdash; enabled once a result exists (just run, or
   restored from the previous session). It opens a window with a single self-contained Markdown
   document covering **everything the interface shows**: environment, the input as it was sent,
   the per-device result with every field (not just the rendered ones), the stability test
   including the per-rotation probabilities and the distribution, the reliability flag, the
   interface notes/explanations and the raw JSON of both responses. Copy it or download it as
   `.md` (`.json` for the raw payloads) and paste it into another LLM for a third opinion.

In the **Avanzate** ("advanced") section: token budgets, `min_confidence` as an abstention
threshold (with a **suggested 0.5** and a one-click fill; when it is left empty the cards still
flag an answer below the suggested threshold as "astensione non attiva"), forced language, an
**instruction preset menu** (the default *eliminates the problem*, the previous wording and three
variants &mdash; correctness, low risk, speed &mdash; editable to run an A/B), a **multilingual /
english checkpoint comparison** (the english run uses an optional English version of the problem,
so the language effect is isolated) and two extra questions &mdash; `score` ("how good is the best
solution?") and `noul` ("is there a solution that solves it completely?").

### Reliability flag

The **affidabilita'** card is a heuristic, not a probability: it combines the `answer_confidence`,
the agreement between the stability rotations (only if the stability test ran) and the coherence
with `score`/`noul` (only if those questions are on). It returns `affidabile`, `da verificare` or
`poco affidabile` with the list of reasons that fired, so you can disagree with it. Example: a
`score` of `optimal` together with a low rotation agreement is reported as
"confidenza non affidabile".

### Casi e benchmark

The **Casi &amp; benchmark** section runs a set of repeatable cases and reports the accuracy. A case
is `{ "problem", "solutions": [...], "expected" }`, where `expected` is the index (0-based) or the
text of the expected solution (`null`/absent = no expected answer); an optional
`"settings": { ... }` overrides the form for that case. Paste the JSON or import/export a file;
**Usa caso corrente** appends the current form as a case. **Esegui benchmark** runs every case
(optionally with stability) and shows per-case results plus accuracy, mean confidence on the
correct and on the wrong answers, abstention rate and mean agreement. Export the run as CSV or
JSON. The cases live in their own `localStorage` key (`layaPlayground.cases.v1`) and survive
**Pulisci**.

### Browser persistence

What you write **is not lost when you reload the page**: problem, solutions, device, strategy,
budgets, checkboxes and instructions go into `localStorage` (key `layaPlayground.v1`), together
with the **last result** (`layaPlayground.lastResult.v1`), which comes back visible on reload with
a "risultato della sessione precedente" notice. The **Pulisci** ("clear") button empties the fields
and deletes both keys.

They are browser-only data: nothing is sent anywhere and nothing ends up in the container. A
payload saved by an earlier version is ignored if its version does not match (`v: 1`).

### Why stability matters

Laya has a **position bias**: the same option in different slots receives different logits. On
decisions where the order is arbitrary (like a list of solutions) this is a real risk. The
stability mode averages the probabilities over `k` rotations: it is the test that tells you whether
to trust that single answer.

---

## API

| Endpoint | Description |
|---|---|
| `GET /api/status` | python, torch, CUDA device, VRAM, resident checkpoints, warmup state |
| `POST /api/solve` | `{problem, solutions[], device, checkpoint, strategy, ...}` -> choice and probabilities |
| `POST /api/stability` | as above + `rotations` -> choice per rotation, averaged probabilities, distribution |
| `POST /api/assess` | `{solve, stability?}` -> heuristic reliability flag per device (no model needed) |
| `POST /api/warmup` | `{device}` -> starts download/preload in the background |
| `POST /api/unload` | frees the weights from memory |

Example:

```bash
curl -s localhost:8000/api/solve -H 'content-type: application/json' -d '{
  "problem": "Il servizio va in timeout sotto carico e il cliente minaccia di andarsene.",
  "solutions": ["Aggiungere un indice alle query lente.", "Separare il database condiviso.", "Mettere una cache Redis davanti alle letture."],
  "device": "cuda", "strategy": "auto"
}'
```

On Windows, `cmd.exe` does not understand single-quoted JSON: use `curl.exe` and put the body in a
file with `-d "@body.json"`, or run the command from PowerShell or Git Bash.

## Docker

```bash
# CPU
docker build -t laya-playground .
docker run --rm -p 8000:8000 -v laya-models:/models --env-file .env laya-playground

# GPU (needs the NVIDIA driver on the host and nvidia-container-toolkit; on Windows,
# Docker Desktop with the WSL2 backend)
docker build -t laya-playground:cuda --build-arg TORCH_BACKEND=cuda .
docker run --rm --gpus all -p 8000:8000 -v laya-models:/models -e LAYA_DEVICE=cuda --env-file .env laya-playground:cuda
```

The `laya-models` volume keeps the weights outside the container, so they survive `--rm` and are
not re-downloaded. Where there is no persistent volume, bake them into the image:

```bash
docker build -t laya-playground:baked --build-arg BAKE=1 .
```

The difference between the two images is **only the index PyTorch comes from** (`cpu` vs `cu130`).
The Linux cu130 wheel brings the CUDA runtime with it (the `nvidia-*` dependencies), so the base
`nvidia/cuda` image is not needed: only the driver at runtime. The container starts with
`LAYA_DEVICE=cpu` by default; if there is no CUDA the warmup is skipped with a warning instead of
failing.

`--env-file .env` passes `HF_TOKEN` and `LAYA_API_KEY` at runtime: **the `.env` does not enter the
image** (it is in `.dockerignore`).

### Publishing somewhere

| | |
|---|---|
| **RAM** | ~2.5-3 GB free are needed (fp32 weights on CPU), ~1 GB with a GPU. The 512 MB free tiers are not enough |
| **Disk** | ~1.7 GB for the checkpoints |
| **Workers** | **one only**: every process keeps the weights in memory, so scale vertically, not with `--workers` |
| **CPU threads** | the server derives them from the container CPU limit (cgroup) or from the host cores, with a cap at 16. Force them with `LAYA_THREADS` if the container has fewer CPUs than torch sees |
| **Preloaded checkpoints** | `--warm` preloads only `multilingual`; change it with `--warm-model english,multilingual` or `LAYA_WARM_MODEL` |
| **First start** | downloads the weights (~1.7 GB) unless you use a volume: on a platform that recreates the container often, either mount a volume or use `--build-arg BAKE=1` to have them in the image |

Recipes:

- **VPS** (Hetzner, DigitalOcean, ...): build and run in the background, then put a reverse proxy
  with TLS in front. Put `LAYA_API_KEY` in the `.env`.

  ```bash
  docker build -t laya-playground .
  docker run -d --name laya --restart unless-stopped -p 8000:8000 -v laya-models:/models --env-file .env laya-playground
  ```
- **Render / Railway / Fly.io**: they point at the `Dockerfile`; expose port 8000. On these
  platforms **mount a volume on `/models`** or build with `--build-arg BAKE=1`, otherwise every
  deploy re-downloads the weights. As said, the free tiers are too small for the CPU: use a plan
  with at least 2 GB.
- **GPU only**: you need an NVIDIA host (not the free tiers). Pass `--gpus all` and
  `LAYA_DEVICE=cuda`.

### If you expose it on the internet

By default the `/api/*` routes are **open**: anyone who finds the URL can burn your CPU and
download models on your bandwidth. Set a key:

```bash
# in the .env
LAYA_API_KEY=cambia-questa-stringa
```

With the variable set, `/api/*` requires the key via `X-Api-Key`, `Authorization: Bearer` or
`?key=`. The UI picks it up from the URL the first time and keeps it in `localStorage`:

```
http://your-host:8000/?key=cambia-questa-stringa
```

`/` (the page) and `/health` stay open: the first is static, the second is polled by hosting
platforms.

## Structure

```
Dockerfile                image (ARG TORCH_BACKEND=cpu|cuda, BAKE=1 to bake the weights in)
run.sh / run.bat / run.ps1  launchers for Linux-macOS, Windows cmd, Windows PowerShell
src/engine.py             device pool, question construction, solve / stability / status
src/server.py             FastAPI + UI
src/warmup.py             command-line warmup and smoke test
src/static/index.html     UI (HTML + CSS + JS, no build, no CDN)
ARCHITECTURE.md           how the pieces fit together
AGENTS.md                 conventions and commands for coding agents
```

## Observed behaviour (measured on this machine, RTX 4070)

| | |
|---|---|
| GPU vs CPU, 5 solutions, warm | **~42 ms** vs **~170 ms** (4x), same choice |
| probability gap GPU (bf16) vs CPU (fp32) | max **0.54 percentage points** |
| first forward pass on GPU | **+5-12 s** (checkpoint build and load into VRAM) |
| tournament on 30 solutions | ~95 ms on GPU, ~670 ms on CPU, 2 finalists, 1 round |
| 5 rotations of the same problem | 4 votes for one solution, 1 for another (measured position bias) |

Three things worth knowing before drawing conclusions:

- **Language detection gets short Italian texts wrong.** `"Il servizio va in timeout sotto
  carico."` is classified as `english`; a longer sentence correctly routes to `multilingual`.
  This is why the playground default is `multilingual`: outside English the English checkpoint
  collapses **while staying confident**, so the error does not show up in the confidence and no
  threshold catches it.
- **The English checkpoint loads with a `RuntimeWarning`**: some temperatures are out of range
  (`choice:11+`) and are clamped to 0.5. Its confidence should be treated as uncalibrated.
- **The position bias is real and measurable** (see the stability mode). On decisions where the
  order of the solutions is arbitrary, a single answer is not as reliable as the probability
  averaged over several rotations.

## Known limits

- The model **does not explain** the choice: it only shows probabilities and confidence.
- With many solutions the descriptions share a token budget: the UI widens the budget
  automatically, but beyond ~50 options the tournament is worth it.
- The probabilities are well ordered, but "high confidence" does not mean "correct answer": the
  abstention threshold must be tuned on your domain.
- The CPU/GPU comparison can show small probability differences: the GPU runs in bf16, the CPU in
  fp32.
