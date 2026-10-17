"""FastAPI server for the Laya playground.

Serves the UI (``src/static/index.html``) and these endpoints:

* ``GET  /api/status``    -- environment, available devices, resident checkpoints, warmup;
* ``POST /api/solve``     -- the problem + the solutions -> choice, probabilities, latency;
* ``POST /api/stability`` -- the same decision under k rotations of the option order;
* ``POST /api/warmup``    -- starts the download/load of the checkpoints in the background.

Startup:  ``uv run python src/server.py``  (or one of the launchers: ``run.sh``, ``run.bat``,
``run.ps1``)
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
    """Key that protects the /api/* routes. Empty (default) = no protection."""
    return (os.environ.get("LAYA_API_KEY") or "").strip()


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """If `LAYA_API_KEY` is set, everything under /api/* requires it.

    Needed when the playground is exposed on the internet: without it, anyone can burn your
    CPU/GPU. `/` (the UI) and `/health` stay open: the first is just a static page, the second
    is polled by hosting platforms. It accepts `X-Api-Key`, `Authorization: Bearer` or `?key=`,
    the latter so the UI can pick the key up from the URL the first time.
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
# Request models
# --------------------------------------------------------------------------------------


class SolveRequest(BaseModel):
    problem: str = Field(..., description="The problem to solve (the model's 'state').")
    solutions: List[str] = Field(..., description="The candidate solutions, one per item.")
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
# Endpoints
# --------------------------------------------------------------------------------------


@app.get("/api/status")
def api_status() -> Dict[str, Any]:
    return {**engine.status(), "auth_required": bool(api_key())}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _validate(problem: str, solutions: List[str]) -> None:
    """Invalid input is a 400, not a 500: no point in trying it on two devices."""
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
        except Exception as exc:  # one error line per device, without losing the other
            log.exception("solve on %s failed", device)
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
        log.exception("stability failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/warmup")
def api_warmup(request: WarmupRequest) -> Dict[str, Any]:
    targets = ["cuda", "cpu"] if request.device == "both" else [request.device]
    # Default: only the checkpoint that routing drops a non-English Latin text on. Preloading
    # `english` too would download ~1.7 GB more that nobody would use.
    names = request.checkpoints or ["multilingual"]
    return {device: engine.pool().start_warmup(device, names) for device in targets}


@app.post("/api/unload")
def api_unload(device: Optional[str] = None) -> Dict[str, Any]:
    freed = engine.pool().unload(device)
    return {"freed": freed, "loaded": engine.pool().loaded()}


@app.exception_handler(Exception)
async def unhandled(_request, exc: Exception) -> JSONResponse:  # pragma: no cover
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


# The UI lives at "/": it must be mounted last, so the /api/* endpoints take precedence.
app.mount("/", StaticFiles(directory=str(engine.STATIC_DIR), html=True), name="ui")


# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Laya playground: choose the solution given a problem")
    parser.add_argument("--host", default=os.environ.get("LAYA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LAYA_PORT", "8000")))
    parser.add_argument("--device", default=os.environ.get("LAYA_DEVICE", "cuda"),
                        help="cuda | cpu | both | auto (default from .env)")
    parser.add_argument("--warm", action="store_true",
                        help="preload the checkpoints in the background at startup")
    parser.add_argument("--warm-model", default=os.environ.get("LAYA_WARM_MODEL", "multilingual"),
                        help="checkpoint to preload with --warm (comma-separated list)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("UI: http://%s:%s   (default device: %s)", args.host, args.port, args.device)
    if api_key():
        log.info("LAYA_API_KEY is set: the /api/* routes require the key")
    else:
        log.warning("LAYA_API_KEY is not set: if you expose the service on the internet, anyone can use it")

    # A single import, in the main thread, before uvicorn accepts requests: it avoids the race
    # between the first `import transformers` of a warmup thread and the one of /api/status.
    torch_info: Dict[str, Any] = {}
    try:
        log.info("preimport: %s", engine.preimport())
        torch_info = engine.status()["torch"]
    except Exception as exc:
        log.warning("preimport failed (%s: %s): the environment may be incomplete", type(exc).__name__, exc)

    if args.warm:
        names = [name.strip() for name in args.warm_model.split(",") if name.strip()]
        for device in (["cuda", "cpu"] if args.device == "both" else [args.device]):
            if device == "cuda" and not torch_info.get("cuda_available"):
                log.warning("cuda warmup skipped: torch does not see CUDA (CPU image or no GPU)")
                continue
            log.info("warmup %s (%s): %s", device, ",".join(names), engine.pool().start_warmup(device, names))

    import uvicorn

    # A single worker: every process keeps the weights in memory, so N workers = N copies of the
    # model. To serve more requests, scale vertically, not with --workers.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", workers=1)


if __name__ == "__main__":
    main()
