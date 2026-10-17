# Laya playground

Playground locale per [Laya](https://github.com/NandhaKishorM/laya) (`convaiinnovations/laya`)
applicato a un caso solo: **dato un problema e una lista di soluzioni candidate, scegliere la
soluzione migliore**.

Serve a vedere in pratica cosa fa il modello, con quale confidenza, con quale latenza su GPU o
CPU, e quanto la scelta e' stabile al variare dell'ordine delle opzioni.

---

## Cos'e' Laya (e cosa non e')

Laya e' un **decision engine** non-autoregressivo, non un LLM generativo:

| | |
|---|---|
| **input** | uno *state* (qui: il testo del problema) + domande tipizzate (`choice`, `score`, `noul`) |
| **output** | un'etichetta scelta, la **distribuzione di probabilita' su ogni opzione**, una confidenza calibrata |
| **costo** | un singolo forward pass: decine di ms su GPU |
| **non fa** | non genera testo, **non spiega il perche'**, non ragiona a passi |

Il caso "scegli fra le soluzioni" e' esattamente una domanda `choice`: ogni soluzione diventa
un'opzione. Le chiavi delle opzioni sono neutre (`s00`, `s01`, ...) perche' i checkpoint seguono
letteralmente etichette tipo `true`/`false` invece del testo dell'opzione.

**Avvertenza onesta dal repo**: i checkpoint base sono vicini al caso su decisioni di dominio;
il valore vero arriva dal fine-tuning sul tuo dominio. Usa questo playground anche per capire
*se* le probabilita' ordinano in modo sensato sul tuo caso.

---

## Setup

```powershell
cd C:\cloud\laya-playground
uv sync --python 3.12
```

`torch` viene risolto dal canale CUDA dichiarato in `pyproject.toml` (`cu130`), che su questa
macchina corrisponde al driver. Per una installazione CPU-only:

```powershell
uv pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
```

Il `.env` contiene `HF_TOKEN` (opzionale: i checkpoint sono pubblici) e la cache dei pesi
spostata in `.hf-cache/`, cosi' tutto e' cancellabile in un colpo.

## Avvio

```powershell
.\run.ps1                 # GPU, http://127.0.0.1:8000
.\run.ps1 --device cpu    # forza la CPU
.\run.ps1 --port 8080
```

oppure direttamente:

```powershell
uv run python src/server.py --warm
```

`--warm` precarica i checkpoint in background all'avvio (il primo predict scarica i pesi dal
Hub: qualche minuto). Nella UI il pulsante **Carica modello** fa la stessa cosa, e le pill in
alto mostrano quando e' pronto.

## Smoke test da riga di comando

```powershell
uv run python src/warmup.py --device both --full
```

Stampa device, dtype, latenza a freddo e a caldo, il tournament su 30 soluzioni e il test di
stabilita'.

## Uso della UI

1. Incolla il **problema** nella prima casella.
2. Incolla le **soluzioni**, una per riga, nella seconda casella (il contatore in alto mostra
   quante ne ha lette).
3. Scegli il **device**: `GPU (cuda)`, `CPU`, oppure `Confronta GPU / CPU` per vedere le due
   risposte affiancate con le rispettive latenze.
4. Scegli la **strategia**:
   - `Auto` &mdash; una sola domanda fino a 20 soluzioni, tournament sopra;
   - `Semplice` &mdash; una sola domanda `choice` con tutte le soluzioni;
   - `Tournament` &mdash; eliminazione a gruppi da `group_size` (default 16), regge fino a
     ~256 soluzioni, a costo di qualche forward pass in piu';
   - `Lungo` &mdash; per problemi piu' lunghi della finestra di contesto: legge l'intero
     documento a finestre invece di troncare.
5. Spunta **Testa anche la stabilita'** per ripetere la domanda con l'ordine delle opzioni
   ruotato `k` volte (in un solo forward pass) e vedere se la scelta cambia.
6. **Decidi** (o Ctrl+Enter).

Nella sezione **Avanzate**: budget di token, `min_confidence` come soglia di astensione,
lingua forzata, istruzioni della domanda (modificabili per fare A/B tra istruzioni in inglese e
in italiano), e due domande extra &mdash; `score` ("quanto e' buona la migliore soluzione?") e
`noul` ("esiste una soluzione che risolve del tutto?").

### Persistenza nel browser

Quello che scrivi **non si perde ricaricando la pagina**: problema, soluzioni, device,
strategia, budget, checkbox e istruzioni finiscono in `localStorage` (chiave
`layaPlayground.v1`), insieme all'**ultimo risultato** (`layaPlayground.lastResult.v1`), che
al reload torna visibile con un avviso "risultato della sessione precedente". Il pulsante
**Pulisci** svuota i campi e cancella entrambe le chiavi.

Sono dati solo del tuo browser: non vengono inviati da nessuna parte e non finiscono nel
container. Un payload salvato da una versione precedente viene ignorato se la sua versione
non corrisponde (`v: 1`).

### Perche' la stabilita' conta

Laya ha una **position bias**: la stessa opzione in slot diversi riceve logit diversi. Su
decisioni in cui l'ordine e' arbitrario (come un elenco di soluzioni) questo e' un rischio
reale. La modalita' stabilita' media le probabilita' su `k` rotazioni: e' il test che dice se
fidarsi di quella singola risposta.

---

## API

| Endpoint | Descrizione |
|---|---|
| `GET /api/status` | python, torch, dispositivo CUDA, VRAM, checkpoint residenti, stato warmup |
| `POST /api/solve` | `{problem, solutions[], device, checkpoint, strategy, ...}` -> scelta e probabilita' |
| `POST /api/stability` | come sopra + `rotations` -> scelta per rotazione e probabilita' mediate |
| `POST /api/warmup` | `{device}` -> avvia download/preload in background |
| `POST /api/unload` | libera i pesi dalla memoria |

Esempio:

```powershell
curl.exe -s localhost:8000/api/solve -H 'content-type: application/json' -d '{
  "problem": "Il servizio va in timeout sotto carico e il cliente minaccia di andarsene.",
  "solutions": ["Aggiungere un indice alle query lente.", "Separare il database condiviso.", "Mettere una cache Redis davanti alle letture."],
  "device": "cuda", "strategy": "auto"
}'
```

## Docker

```powershell
# CPU
docker build -t laya-playground .
docker run --rm -p 8000:8000 -v laya-models:/models --env-file .env laya-playground

# GPU (serve il driver NVIDIA sull'host e nvidia-container-toolkit; su Windows,
# Docker Desktop con backend WSL2)
docker build -t laya-playground:cuda --build-arg TORCH_BACKEND=cuda .
docker run --rm --gpus all -p 8000:8000 -v laya-models:/models -e LAYA_DEVICE=cuda --env-file .env laya-playground:cuda
```

Il volume `laya-models` tiene i pesi fuori dal container, così sopravvivono a `--rm` e non
vengono riscaricati. Dove non c'è un volume persistente, mettili dentro l'immagine:

```powershell
docker build -t laya-playground:baked --build-arg BAKE=1 .
```

La differenza fra le due immagini è **solo l'indice da cui arriva PyTorch** (`cpu` vs `cu130`).
La wheel Linux cu130 porta dentro il runtime CUDA (dipendenze `nvidia-*`), quindi non serve
l'immagine base `nvidia/cuda`: serve solo il driver a runtime. Il container parte con
`LAYA_DEVICE=cpu` di default; se non c'è la CUDA il warmup viene saltato con un avviso invece
di fallire.

`--env-file .env` passa `HF_TOKEN` e `LAYA_API_KEY` a runtime: **il `.env` non entra
nell'immagine** (è in `.dockerignore`).

### Pubblicare da qualche parte

| | |
|---|---|
| **RAM** | servono ~2.5-3 GB liberi (i pesi fp32 su CPU), ~1 GB con GPU. Le free tier da 512 MB non bastano |
| **Disco** | ~1.7 GB per i checkpoint |
| **Worker** | **uno solo**: ogni processo tiene i pesi in memoria, quindi si scala verticalmente, non con `--workers` |
| **Thread CPU** | il server li ricava dal limite di CPU del container (cgroup) o dai core dell'host, con tetto a 16. Forzali con `LAYA_THREADS` se il container ha meno CPU di quante ne veda torch |
| **Checkpoint precaricati** | `--warm` precarica il solo `multilingual`; cambia con `--warm-model english,multilingual` o `LAYA_WARM_MODEL` |
| **Primo avvio** | scarica i pesi (~1.7 GB) se non usi un volume: su una piattaforma che ricrea il container spesso, o monti un volume o usi `--build-arg BAKE=1` per averli nell'immagine |

Ricette:

- **VPS** (Hetzner, DigitalOcean, ...): builda e lancia in background, poi metti davanti un
  reverse proxy con TLS. Metti `LAYA_API_KEY` nel `.env`.

  ```powershell
  docker build -t laya-playground .
  docker run -d --name laya --restart unless-stopped -p 8000:8000 -v laya-models:/models --env-file .env laya-playground
  ```
- **Render / Railway / Fly.io**: puntano al `Dockerfile`; esponi la porta 8000. Su queste
  piattaforme **monta un volume su `/models`** o builda con `--build-arg BAKE=1`, altrimenti
  ogni deploy riscarica i pesi. Le free tier come dette sono troppo piccole per la CPU: usa
  un piano da almeno 2 GB.
- **Solo GPU**: serve un host con NVIDIA (non le free tier). Passa `--gpus all` e
  `LAYA_DEVICE=cuda`.

### Se lo esponi su internet

Di default le rotte `/api/*` sono **aperte**: chiunque trova l'URL può consumare la tua CPU e
scaricare modelli sulla tua banda. Imposta una chiave:

```powershell
# nel .env
LAYA_API_KEY=cambia-questa-stringa
```

Con la variabile impostata, `/api/*` richiede la chiave via `X-Api-Key`, `Authorization: Bearer`
o `?key=`. La UI la prende dall'URL la prima volta e la tiene in `localStorage`:

```
http://tuo-host:8000/?key=cambia-questa-stringa
```

`/` (la pagina) e `/health` restano aperti: la prima è statica, il secondo lo interrogano le
piattaforme di hosting.

## Struttura

```
Dockerfile            immagine (ARG TORCH_BACKEND=cpu|cuda, BAKE=1 per i pesi dentro)
src/engine.py         pool di device, costruzione della domanda, solve / stability / status
src/server.py         FastAPI + UI
src/warmup.py         warmup e smoke test da riga di comando
src/static/index.html UI (HTML + CSS + JS, nessuna build, nessuna CDN)
```

## Comportamenti osservati (misurati su questa macchina, RTX 4070)

| | |
|---|---|
| GPU vs CPU, 5 soluzioni, a caldo | **~42 ms** vs **~170 ms** (4x), stessa scelta |
| scarto di probabilita' GPU (bf16) vs CPU (fp32) | max **0.54 punti percentuali** |
| primo forward su GPU | **+5-12 s** (build del checkpoint e caricamento in VRAM) |
| tournament su 30 soluzioni | ~95 ms su GPU, ~670 ms su CPU, 2 finaliste, 1 round |
| 5 rotazioni dello stesso problema | 4 voti a una soluzione, 1 a un'altra (position bias misurata) |

Tre cose che è utile sapere prima di trarre conclusioni:

- **Il rilevamento della lingua sbaglia sui testi italiani brevi.** `"Il servizio va in timeout
  sotto carico."` viene classificato come `english`; una frase più lunga manda correttamente a
  `multilingual`. Per questo il default del playground è `multilingual`: fuori dall'inglese il
  checkpoint inglese crolla **restando confidente**, quindi l'errore non si vede nella
  confidenza e nessuna soglia lo intercetta.
- **Il checkpoint inglese carica con un `RuntimeWarning`**: alcune temperature sono fuori
  intervallo (`choice:11+`) e vengono clampate a 0.5. La sua confidenza è da trattare come non
  calibrata.
- **La position bias è reale e misurabile** (vedi la modalità stabilità). Su decisioni dove
  l'ordine delle soluzioni è arbitrario, una singola risposta non è affidabile quanto la
  probabilita' mediata su più rotazioni.

## Limiti noti

- Il modello **non spiega** la scelta: mostra solo probabilita' e confidenza.
- Con molte soluzioni le descrizioni condividono un budget di token: la UI allarga il budget
  automaticamente, ma oltre ~50 opzioni conviene il tournament.
- Le probabilita' sono ordinate bene, ma "confidenza alta" non significa "risposta giusta":
  la soglia di astensione va tarata sul tuo dominio.
- Il confronto CPU/GPU puo' mostrare piccole differenze di probabilita': la GPU gira in bf16,
  la CPU in fp32.
