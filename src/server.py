"""Server FastAPI del playground Laya.

Serve la UI (``src/static/index.html``) e tre endpoint:

* ``GET  /api/status``    -- ambiente, device disponibili, checkpoint residenti, warmup;
* ``POST /api/solve``     -- il problema + le soluzioni -> scelta, probabilita', latenza;
* ``POST /api/stability`` -- la stessa decisione sotto k rotazioni dell'ordine delle opzioni;
* ``POST /api/warmup``    -- avvia in background il download/load dei checkpoint.

Avvio:  ``uv run python src/server.py``  (oppure ``.\run.ps1``)
"""

from __future__ import annotations

import argparse
import hmac
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import engine

log = logging.getLogger("laya-playground")

app = FastAPI(title="Laya playground", version="0.1.0")


def api_key() -> str:
    """Chiave che protegge le rotte /api/*. Vuota (default) = nessuna protezione."""
    return (os.environ.get("LAYA_API_KEY") or "").strip()


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Se `LAYA_API_KEY` e' impostata, tutto /api/* la richiede.

    Serve quando il playground e' esposto su internet: senza, chiunque puo' consumare la tua
    CPU/GPU. `/` (la UI) e `/health` restano aperti: la prima e' solo una pagina statica, il
    secondo lo interrogano le piattaforme di hosting. Accetta `X-Api-Key`, `Authorization:
    Bearer` o `?key=`, quest'ultimo perche' la UI possa prenderla dall'URL la prima volta.
    """
    key = api_key()
    if not key or not request.url.path.startswith("/api/"):
        return await call_next(request)

    supplied = (request.headers.get("x-api-key") or "").strip()
    if not supplied:
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
    if not supplied:
        supplied = (request.query_params.get("key") or "").strip()

    if not supplied or not hmac.compare_digest(supplied, key):
        return JSONResponse(status_code=401, content={"detail": "API key mancante o errata"})
    return await call_next(request)


# --------------------------------------------------------------------------------------
# Modelli di richiesta
# --------------------------------------------------------------------------------------


class SolveRequest(BaseModel):
    problem: str = Field(..., description="Il problema da risolvere (lo 'state' del modello).")
    solutions: List[str] = Field(..., description="Le soluzioni candidate, una per elemento.")
    device: str = Field("cuda", description="cuda | cpu | both")
    checkpoint: str = Field("auto", description="auto | english | multilingual | typed-decisions")
    strategy: str = Field("auto", description="auto | simple | tournament | long")
    group_size: int = Field(engine.DEFAULT_GROUP_SIZE, ge=2, le=64)
    max_len: Optional[int] = Field(None, ge=64, le=8192)
    head_max_len: Optional[int] = Field(None, ge=16, le=4096)
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    instructions: Optional[str] = None
    with_quality: bool = False
    with_complete: bool = False
    lang: Optional[str] = None


class StabilityRequest(BaseModel):
    problem: str
    solutions: List[str]
    device: str = "cuda"
    checkpoint: str = "auto"
    rotations: Optional[int] = Field(None, ge=1, le=engine.MAX_ROTATIONS)
    max_len: Optional[int] = Field(None, ge=64, le=8192)
    head_max_len: Optional[int] = Field(None, ge=16, le=4096)
    instructions: Optional[str] = None
    lang: Optional[str] = None


class WarmupRequest(BaseModel):
    device: str = "cuda"
    checkpoints: Optional[List[str]] = None

# --------------------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------------------


@app.get("/api/status")
def api_status() -> Dict[str, Any]:
    return {**engine.status(), "auth_required": bool(api_key())}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _validate(problem: str, solutions: List[str]) -> None:
    """Input non valido e' un 400, non un 500: non ha senso provarlo su due device."""
    if not (problem or "").strip():
        raise HTTPException(status_code=400, detail="il problema e' vuoto")
    if len(engine.clean_solutions(solutions)) < 2:
        raise HTTPException(status_code=400, detail="servono almeno due soluzioni candidate distinte")


@app.post("/api/solve")
def api_solve(request: SolveRequest) -> Dict[str, Any]:
    _validate(request.problem, request.solutions)
    targets = ["cuda", "cpu"] if request.device == "both" else [request.device]

    results: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    for device in targets:
        try:
            results.append(
                engine.solve(
                    problem=request.problem,
                    solutions=request.solutions,
                    device=device,
                    checkpoint=request.checkpoint,
                    strategy=request.strategy,
                    group_size=request.group_size,
                    max_len=request.max_len,
                    head_max_len=request.head_max_len,
                    min_confidence=request.min_confidence,
                    instructions=request.instructions,
                    with_quality=request.with_quality,
                    with_complete=request.with_complete,
                    lang=request.lang,
                )
            )
        except Exception as exc:  # una riga di errore per device, senza perdere l'altra
            log.exception("solve su %s fallito", device)
            errors[device] = str(exc)

    if not results:
        raise HTTPException(status_code=500, detail=errors or "nessun risultato")

    return {"mode": request.device, "results": results, "errors": errors}


@app.post("/api/stability")
def api_stability(request: StabilityRequest) -> Dict[str, Any]:
    _validate(request.problem, request.solutions)
    try:
        return engine.stability(
            problem=request.problem,
            solutions=request.solutions,
            device=request.device,
            checkpoint=request.checkpoint,
            rotations=request.rotations,
            max_len=request.max_len,
            head_max_len=request.head_max_len,
            instructions=request.instructions,
            lang=request.lang,
        )
    except Exception as exc:
        log.exception("stability fallita")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/warmup")
def api_warmup(request: WarmupRequest) -> Dict[str, Any]:
    targets = ["cuda", "cpu"] if request.device == "both" else [request.device]
    # Default: solo il checkpoint su cui il routing fa cadere un testo latino non inglese.
    # Precaricare anche `english` scaricherebbe ~1.7 GB in piu' senza che nessuno lo usi.
    names = request.checkpoints or ["multilingual"]
    return {device: engine.pool().start_warmup(device, names) for device in targets}


@app.post("/api/unload")
def api_unload(device: Optional[str] = None) -> Dict[str, Any]:
    freed = engine.pool().unload(device)
    return {"freed": freed, "loaded": engine.pool().loaded()}


@app.exception_handler(Exception)
async def unhandled(_request, exc: Exception) -> JSONResponse:  # pragma: no cover
    log.exception("errore non gestito")
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


# La UI sta su "/": va montata per ultima, cosi' gli endpoint /api/* hanno la precedenza.
app.mount("/", StaticFiles(directory=str(engine.STATIC_DIR), html=True), name="ui")


# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Playground Laya: scelta della soluzione dato un problema")
    parser.add_argument("--host", default=os.environ.get("LAYA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LAYA_PORT", "8000")))
    parser.add_argument("--device", default=os.environ.get("LAYA_DEVICE", "cuda"),
                        help="cuda | cpu | both | auto (default dal .env)")
    parser.add_argument("--warm", action="store_true",
                        help="precarica i checkpoint in background all'avvio")
    parser.add_argument("--warm-model", default=os.environ.get("LAYA_WARM_MODEL", "multilingual"),
                        help="checkpoint da precaricare con --warm (lista separata da virgole)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("UI: http://%s:%s   (device di default: %s)", args.host, args.port, args.device)
    if api_key():
        log.info("LAYA_API_KEY impostata: le rotte /api/* richiedono la chiave")
    else:
        log.warning("LAYA_API_KEY non impostata: se esponi il servizio su internet, chiunque puo' usarlo")

    # Un import solo, nel thread principale, prima che uvicorn accetti richieste: evita la
    # corsa fra il primo `import transformers` di un thread di warmup e quello di /api/status.
    torch_info: Dict[str, Any] = {}
    try:
        log.info("preimport: %s", engine.preimport())
        torch_info = engine.status()["torch"]
    except Exception as exc:
        log.warning("preimport fallito (%s: %s): l'ambiente potrebbe essere incompleto", type(exc).__name__, exc)

    if args.warm:
        names = [name.strip() for name in args.warm_model.split(",") if name.strip()]
        for device in (["cuda", "cpu"] if args.device == "both" else [args.device]):
            if device == "cuda" and not torch_info.get("cuda_available"):
                log.warning("warmup cuda saltato: torch non vede la CUDA (immagine CPU o nessuna GPU)")
                continue
            log.info("warmup %s (%s): %s", device, ",".join(names), engine.pool().start_warmup(device, names))

    import uvicorn

    # Un solo worker: ogni processo tiene i pesi in memoria, quindi N worker = N copie del
    # modello. Per servire piu' richieste si scala verticalmente, non con --workers.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", workers=1)


if __name__ == "__main__":
    main()
