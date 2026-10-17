"""Motore del playground Laya.

Un solo punto di verita' per tre cose:

* il **pool di device** -- un ``Router`` per device (``cpu`` / ``cuda`` / ...), costruito al
  primo uso e poi riutilizzato, cosi' il toggle CPU/GPU nella UI non ripaga il load ad ogni
  richiesta;
* la **costruzione della domanda** -- il problema e' lo *state*, ogni soluzione candidata e'
  una opzione di una domanda ``choice`` con chiavi neutre (``s00``, ``s01``, ...). Le chiavi
  sono opache di proposito: i checkpoint seguono letteralmente etichette tipo ``true``/``false``
  invece del testo dell'opzione;
* le **modalita' di test** -- ``simple`` (una sola domanda), ``tournament`` (eliminazione a
  gruppi, regge fino a 256 soluzioni), ``long`` (problema piu' lungo della finestra) e
  ``stability`` (k rotazioni dell'ordine delle opzioni, mediate insieme).

Nessun import di ``torch`` o ``laya`` a livello di modulo: entrambi sono pesanti e questo file
viene importato dal server all'avvio.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# --------------------------------------------------------------------------------------
# Percorsi e configurazione
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
STATIC_DIR = SRC_DIR / "static"


def _load_dotenv(path: Path) -> None:
    """Carica un ``.env`` minimale (KEY=VALUE) senza dipendere da python-dotenv."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")

# I pesi finiscono dentro il progetto, non nella cache utente: cosi' si cancella tutto
# insieme. Va impostato prima che qualsiasi cosa importi huggingface_hub.
_HF_CACHE = os.environ.get("LAYA_HF_CACHE", ".hf-cache")
HF_CACHE_DIR = (ROOT / _HF_CACHE).resolve() if not Path(_HF_CACHE).is_absolute() else Path(_HF_CACHE)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))

# Il checkpoint non e' un LLM: non "spiega", sceglie. Le istruzioni sono in inglese di
# default perche' e' la lingua con piu' dati di training; i testi delle soluzioni restano
# nella lingua dell'utente. Dalla UI sono modificabili per fare A/B.
INSTRUCTIONS_EN = (
    "Which candidate solution best solves the problem described in the state? "
    "Pick the single most effective and practically applicable one."
)
INSTRUCTIONS_IT = (
    "Quale soluzione candidata risolve meglio il problema descritto? "
    "Scegli quella piu' efficace e applicabile in pratica."
)

QUALITY_INSTRUCTIONS = "How good is the best solution available for this problem?"
QUALITY_LEVELS = ["useless", "weak", "workable", "good", "optimal"]

COMPLETE_INSTRUCTIONS = "Does at least one candidate solution fully solve the problem?"
COMPLETE_CRITERIA = {
    "false": "no candidate solution fully solves the problem",
    "true": "at least one candidate solution fully solves the problem",
}

# Sotto questa soglia una sola domanda `choice` basta; sopra conviene il tournament.
AUTO_TOURNAMENT_AT = 20
# Dimensione dei gruppi del tournament (il default della libreria e' 16).
DEFAULT_GROUP_SIZE = 16
# Oltre questo numero la modalita' stabilita' costa piu' di quanto dica.
MAX_ROTATIONS = 8

STRATEGIES = ("auto", "simple", "tournament", "long")


def hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return token.strip() or None if token else None


def clean_solutions(raw: Iterable[str]) -> List[str]:
    """Una soluzione per riga: scarta righe vuote, tiene l'ordine, rimuove i duplicati."""
    out: List[str] = []
    seen = set()
    for item in raw:
        text = (item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


# --------------------------------------------------------------------------------------
# Pool di device
# --------------------------------------------------------------------------------------

_DEVICE_ALIASES = {
    "gpu": "cuda",
    "cu": "cuda",
    "cuda": "cuda",
    "cpu": "cpu",
    "mps": "mps",
    "xpu": "xpu",
    "auto": "auto",
}


def normalise_device(device: Optional[str]) -> str:
    value = (device or os.environ.get("LAYA_DEVICE") or "cuda").strip().lower()
    return _DEVICE_ALIASES.get(value, value)


class DevicePool:
    """Un ``Router`` per device, costruito al primo uso e riusato.

    ``Router(device=...)`` decide il backend una volta per oggetto, quindi il toggle CPU/GPU
    richiede due istanze. Tenerle entrambe vive e' cio' che rende istantaneo il passaggio e
    che rende possibile il confronto CPU-vs-GPU nella UI.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._routers: Dict[str, Any] = {}
        self._errors: Dict[str, str] = {}
        self._init_seconds: Dict[str, float] = {}
        self._warm: Dict[str, Dict[str, Any]] = {}

    # -- stato -------------------------------------------------------------------------

    def warm_state(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {device: dict(info) for device, info in self._warm.items()}

    def loaded(self) -> Dict[str, List[str]]:
        with self._lock:
            # `Router.loaded` e' una property, non un metodo.
            return {device: list(router.loaded) for device, router in self._routers.items()}

    def router_init_seconds(self) -> Dict[str, float]:
        """Quanto e' costata la *costruzione* del Router (che e' lazy: i pesi no)."""
        with self._lock:
            return dict(self._init_seconds)

    def errors(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._errors)

    # -- costruzione -------------------------------------------------------------------

    def get(self, device: Optional[str]) -> Any:
        """Restituisce il Router per il device richiesto, costruendolo se serve."""
        key = normalise_device(device)
        with self._lock:
            router = self._routers.get(key)
            if router is not None:
                return router
            failure = self._errors.get(key)
            if failure is not None:
                raise RuntimeError(failure)

        # Costruito fuori dal lock: il build scarica i pesi e puo' durare minuti, e non
        # vogliamo bloccare il toggle dell'altro device nel frattempo.
        try:
            preimport()
            from laya import Router

            kwargs: Dict[str, Any] = {
                "token": hf_token(),
                "default": "multilingual",
                "max_loaded": 2,
            }
            if key != "auto":
                kwargs["device"] = key
            started = time.perf_counter()
            router = Router(**kwargs)
            seconds = time.perf_counter() - started
        except Exception as exc:  # pragma: no cover - dipende dall'ambiente
            message = f"device '{key}': non riesco a costruire il Router ({type(exc).__name__}: {exc})"
            with self._lock:
                self._errors[key] = message
            raise RuntimeError(message) from exc

        with self._lock:
            self._routers[key] = router
            self._init_seconds[key] = seconds
            # Non calpestare un warmup in corso: chi lo ha avviato scrive "ready" a fine
            # preload, e sovrascriverlo qui farebbe credere che i pesi siano gia' residenti.
            if self._warm.get(key, {}).get("state") != "loading":
                self._warm[key] = {"state": "ready", "seconds": round(seconds, 3), "detail": None}
        return router

    # -- preload in background ---------------------------------------------------------

    def start_warmup(self, device: Optional[str], checkpoints: Optional[List[str]] = None) -> Dict[str, Any]:
        """Avvia il load/preload in un thread e torna subito.

        Il primo ``predict`` scarica un checkpoint dal Hub: puo' durare minuti, e la UI deve
        poterlo mostrare invece di sembrare bloccata.

        Il default e' il solo ``multilingual``: e' dove il routing fa cadere un testo latino
        non inglese, e precaricare anche ``english`` scaricherebbe ~1.7 GB in piu' che nessuno
        userebbe in un caso come questo (``english`` resta comunque a load lazy al primo uso).
        """
        key = normalise_device(device)
        with self._lock:
            current = self._warm.get(key, {})
            if current.get("state") == "loading":
                return {"started": False, "reason": "gia' in caricamento", **current}
            self._warm[key] = {"state": "loading", "seconds": None, "detail": None}

        def _worker() -> None:
            started = time.perf_counter()
            try:
                router = self.get(key)
                names = checkpoints or ["multilingual"]
                router.preload(names)
                elapsed = round(time.perf_counter() - started, 2)
                with self._lock:
                    self._warm[key] = {
                        "state": "ready",
                        "seconds": elapsed,
                        "detail": "checkpoint residenti: " + ", ".join(router.loaded),
                    }
            except Exception as exc:  # pragma: no cover - dipende dall'ambiente
                with self._lock:
                    self._warm[key] = {
                        "state": "error",
                        "seconds": round(time.perf_counter() - started, 2),
                        "detail": f"{type(exc).__name__}: {exc}",
                    }

        threading.Thread(target=_worker, name=f"laya-warmup-{key}", daemon=True).start()
        return {"started": True, "state": "loading"}

    def unload(self, device: Optional[str] = None) -> List[str]:
        """Libera la memoria dei checkpoint (il Router resta, i pesi no)."""
        key = normalise_device(device) if device else None
        freed: List[str] = []
        with self._lock:
            targets = [key] if key else list(self._routers)
        for name in targets:
            with self._lock:
                router = self._routers.get(name)
                # `Router.unload()` libera ma non ritorna nulla: i nomi li leggiamo prima.
                resident = list(router.loaded) if router is not None else []
            if router is not None:
                router.unload()
                freed.extend(resident)
        return freed


_POOL = DevicePool()

# transformers 5.x risolve i nomi pigramente: se due thread avviano l'import nello stesso
# momento, uno puo' vedere il modulo a meta' e fallire con
# `cannot import name 'AutoTokenizer' from 'transformers'`. Serializzare il primo import
# (dopo, `import transformers` e' solo un lookup in sys.modules) elimina la corsa.
_IMPORT_LOCK = threading.RLock()
_preimported: Optional[str] = None


def _auto_threads() -> int:
    """Thread da usare per l'inferenza su CPU.

    In un container torch vede i core dell'**host**, non il limite dato a Docker: su una
    macchina grossa un container da 2 vCPU finirebbe con 64 thread e diventerebbe lentissimo
    per over-subscription. Quindi: si rispetta il limite di CPU del cgroup se c'e', altrimenti
    si prende il numero di core, con un tetto a 16 (il massimo misurato nel README di laya).
    """
    try:  # cgroup v2
        quota, period = (Path("/sys/fs/cgroup/cpu.max").read_text().split() + ["100000"])[:2]
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    try:  # cgroup v1
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            return max(1, int(quota / period))
    except Exception:
        pass
    return max(1, min(os.cpu_count() or 1, 16))


def preimport() -> Optional[str]:
    """Importa una volta sola, sotto lock, torch/transformers/laya e i nomi lazy di transformers."""
    global _preimported
    with _IMPORT_LOCK:
        if _preimported is not None:
            return _preimported
        import torch
        import transformers

        from laya import Agent, Router  # noqa: F401

        for name in ("AutoTokenizer", "AutoConfig", "AutoModel"):
            getattr(transformers, name, None)  # forza la lazy module a risolverli ora

        # LAYA_THREADS vince; senza, si calcola dal cgroup/numero di core (vedi _auto_threads).
        threads = (os.environ.get("LAYA_THREADS") or "").strip()
        count = int(threads) if threads.isdigit() and int(threads) > 0 else _auto_threads()
        torch.set_num_threads(count)

        _preimported = (
            f"torch {torch.__version__} / threads {torch.get_num_threads()} "
            f"/ transformers {transformers.__version__}"
        )
        return _preimported


def pool() -> DevicePool:
    return _POOL


# --------------------------------------------------------------------------------------
# Domanda e budget
# --------------------------------------------------------------------------------------


def build_questions(
    solutions: List[str],
    *,
    instructions: Optional[str] = None,
    with_quality: bool = False,
    with_complete: bool = False,
) -> Dict[str, Any]:
    """Costruisce il dizionario di domande tipizzate per una lista di soluzioni."""
    keys = [f"s{i:02d}" for i in range(len(solutions))]
    criteria = {key: text for key, text in zip(keys, solutions)}

    questions: Dict[str, Any] = {
        "best": {
            "type": "choice",
            "instructions": instructions or INSTRUCTIONS_EN,
            "criteria": criteria,
        }
    }
    if with_quality:
        questions["quality"] = {
            "type": "score",
            "instructions": QUALITY_INSTRUCTIONS,
            "criteria": list(QUALITY_LEVELS),
        }
    if with_complete:
        questions["complete"] = {
            "type": "noul",
            "instructions": COMPLETE_INSTRUCTIONS,
            "criteria": dict(COMPLETE_CRITERIA),
        }
    return questions


def suggest_budgets(n_options: int, checkpoint: str = "auto") -> Dict[str, int]:
    """Budget di token sensati per un problema con ``n_options`` soluzioni.

    In Laya le opzioni condividono un budget (``head_max_len``) e ogni opzione sotto ~4 token
    diventa indistinguibile dalle altre: oltre ~20 opzioni i default (192/256) troncano le
    descrizioni. Qui si allarga il budget in proporzione, con un tetto.
    """
    base_head, base_max = (192, 512) if checkpoint == "english" else (256, 1024)
    head = min(max(base_head, 16 + 24 * max(1, n_options)), 4096)
    max_len = min(max(base_max, head + 512), 8192)
    return {"head_max_len": int(head), "max_len": int(max_len)}


# --------------------------------------------------------------------------------------
# Inferenza
# --------------------------------------------------------------------------------------


def _quality_criteria() -> Dict[str, str]:
    """Mappa dei livelli `score` valida sia che le probabilita' siano per indice sia per testo."""
    mapping = {str(index): level for index, level in enumerate(QUALITY_LEVELS)}
    mapping.update({level: level for level in QUALITY_LEVELS})
    return mapping


def _answer_view(
    answer: Dict[str, Any], criteria: Dict[str, str], *, compute_eliminated: bool = True
) -> Dict[str, Any]:
    """Normalizza la risposta di una domanda in qualcosa che la UI legge facile."""
    probabilities = answer.get("probabilities") or {}
    present = [key for key in criteria if key in probabilities] or list(probabilities)
    ranked = sorted(
        (
            {"key": key, "text": criteria.get(key, key), "prob": float(probabilities.get(key, 0.0))}
            for key in present
        ),
        key=lambda item: item["prob"],
        reverse=True,
    )
    return {
        "choice": answer.get("choice"),
        "top": ranked[0] if ranked else None,
        "ranked": ranked,
        "eliminated": (
            [{"key": k, "text": criteria[k]} for k in criteria if k not in probabilities]
            if compute_eliminated
            else []
        ),
        "confidence": answer.get("confidence"),
        "answer_confidence": answer.get("answer_confidence"),
        "low_confidence": answer.get("low_confidence"),
        "abstention": answer.get("abstention"),
        "abstention_threshold": answer.get("abstention_threshold"),
    }


def _agent_facts(router: Any, routing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Device e dtype reali del checkpoint che ha risposto."""
    facts: Dict[str, Any] = {}
    name = (routing or {}).get("model")
    try:
        agent = router.load(name) if name else None
    except Exception:
        agent = None
    if agent is not None:
        for attr in ("device", "dtype"):
            value = getattr(agent, attr, None)
            if value is not None:
                facts[attr] = str(value)
    return facts


def _truncation_note(problem: str, max_len: int, head_max_len: int) -> Optional[str]:
    """Avviso approssimativo quando il problema rischia di essere troncato."""
    room = max_len - head_max_len - 1
    approx_tokens = max(1, len(problem) // 4)
    if approx_tokens > room:
        return (
            f"il problema e' lungo ~{approx_tokens} token stimati ma restano ~{room} token "
            f"dopo le opzioni: passa a strategia 'long' per leggerlo tutto"
        )
    return None


def solve(
    *,
    problem: str,
    solutions: List[str],
    device: Optional[str] = None,
    checkpoint: str = "auto",
    strategy: str = "auto",
    group_size: int = DEFAULT_GROUP_SIZE,
    max_len: Optional[int] = None,
    head_max_len: Optional[int] = None,
    min_confidence: Optional[float] = None,
    instructions: Optional[str] = None,
    with_quality: bool = False,
    with_complete: bool = False,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """Sceglie la soluzione migliore fra quelle date, su un solo device."""
    problem = (problem or "").strip()
    solutions = clean_solutions(solutions)
    if not problem:
        raise ValueError("il problema e' vuoto")
    if len(solutions) < 2:
        raise ValueError("servono almeno due soluzioni candidate")

    requested_strategy = strategy
    if strategy == "auto":
        strategy = "tournament" if len(solutions) > AUTO_TOURNAMENT_AT else "simple"
    if strategy not in STRATEGIES:
        raise ValueError(f"strategia sconosciuta: {requested_strategy!r}")

    questions = build_questions(
        solutions, instructions=instructions, with_quality=with_quality, with_complete=with_complete
    )
    criteria = questions["best"]["criteria"]
    n = len(solutions)

    budgets = suggest_budgets(n, checkpoint if checkpoint != "auto" else "multilingual")
    resolved_max_len = int(max_len) if max_len else budgets["max_len"]
    resolved_head_max_len = int(head_max_len) if head_max_len else budgets["head_max_len"]

    router = pool().get(device)
    state = {"problem": problem}

    # `predict_long` non accetta i budget (`window` li sostituisce) ne' `min_confidence`:
    # i kwargs comuni e quelli di budget restano separati.
    common: Dict[str, Any] = {}
    if lang:
        common["lang"] = lang
    if checkpoint and checkpoint != "auto":
        common["model"] = checkpoint

    budget_kwargs: Dict[str, Any] = {
        "max_len": resolved_max_len,
        "head_max_len": resolved_head_max_len,
    }
    if min_confidence is not None:
        budget_kwargs["min_confidence"] = float(min_confidence)

    started = time.perf_counter()
    if strategy == "tournament":
        from laya import predict_tournament

        result = predict_tournament(
            router, state, questions, group_size=int(group_size), **common, **budget_kwargs
        )
    elif strategy == "long":
        result = router.predict_long(state, questions, **common)
    else:
        result = router.predict(state, questions, **common, **budget_kwargs)
    latency_ms = (time.perf_counter() - started) * 1000.0

    answers = result.get("answers") or {}
    best = answers.get("best") or {}
    view = _answer_view(best, criteria)
    routing = result.get("routing")
    facts = _agent_facts(router, routing)

    tournament = result.get("tournament") or {}
    tournament_meta = None
    if strategy == "tournament" and "best" in tournament:
        entry = tournament["best"]
        finalists = [str(label) for label in entry.get("labels", [])]
        tournament_meta = {
            "rounds": entry.get("rounds", 0),
            "group_size": int(group_size),
            "finalists": [{"key": k, "text": criteria.get(k, k)} for k in finalists],
        }

    notes: List[str] = []
    if strategy == "long":
        windows = (result.get("usage") or {}).get("windows")
        if windows is not None:
            notes.append(f"problema letto a finestre: {windows}")
    else:
        note = _truncation_note(problem, resolved_max_len, resolved_head_max_len)
        if note:
            notes.append(note)
    if strategy == "tournament" and n > int(group_size) * 4:
        notes.append("con piu' di 256 soluzioni il tournament non basta: riduci la lista")

    return {
        "device": normalise_device(device),
        "requested": {
            "device": device,
            "checkpoint": checkpoint,
            "strategy": requested_strategy,
            "max_len": max_len,
            "head_max_len": head_max_len,
            "min_confidence": min_confidence,
            "lang": lang,
        },
        "strategy": strategy,
        "strategy_reason": (
            f"{n} soluzioni: {'tournament' if strategy == 'tournament' else strategy}"
            if requested_strategy == "auto"
            else f"forzata: {strategy}"
        ),
        "n_solutions": n,
        "budgets": {"max_len": resolved_max_len, "head_max_len": resolved_head_max_len},
        "latency_ms": round(latency_ms, 1),
        "routing": routing,
        "checkpoint": (routing or {}).get("model"),
        "reason": (routing or {}).get("reason"),
        "dtype": facts.get("dtype"),
        "usage": result.get("usage"),
        "tournament": tournament_meta,
        "extras": {
            "quality": _answer_view(answers["quality"], _quality_criteria(), compute_eliminated=False)
            if "quality" in answers
            else None,
            "complete": answers.get("complete"),
        },
        "notes": notes,
        **view,
    }


def stability(
    *,
    problem: str,
    solutions: List[str],
    device: Optional[str] = None,
    checkpoint: str = "auto",
    rotations: Optional[int] = None,
    max_len: Optional[int] = None,
    head_max_len: Optional[int] = None,
    instructions: Optional[str] = None,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """Testa la stabilita' della scelta ruotando l'ordine delle opzioni.

    Laya ha una position bias documentata: la stessa opzione in slot diversi riceve logit
    diversi. Qui la stessa domanda viene posta ``k`` volte con ``option_order`` ruotato, tutte
    in **un solo forward pass**, e si riporta sia la scelta mediata sia quante volte la scelta
    e' cambiata. E' il test che dice se fidarsi di quella decisione.
    """
    problem = (problem or "").strip()
    solutions = clean_solutions(solutions)
    n = len(solutions)
    if not problem:
        raise ValueError("il problema e' vuoto")
    if n < 2:
        raise ValueError("servono almeno due soluzioni candidate")

    k = int(rotations) if rotations else min(n, 4)
    k = max(1, min(k, n, MAX_ROTATIONS))

    base = build_questions(solutions, instructions=instructions)["best"]
    criteria = base["criteria"]
    keys = list(criteria)

    rotations_questions: Dict[str, Any] = {}
    orders: Dict[str, List[int]] = {}
    for r in range(k):
        order = [(i + r) % n for i in range(n)]
        rotations_questions[f"r{r}"] = dict(base, option_order=order)
        orders[f"r{r}"] = order

    budgets = suggest_budgets(n, checkpoint if checkpoint != "auto" else "multilingual")
    resolved_max_len = int(max_len) if max_len else budgets["max_len"]
    resolved_head_max_len = int(head_max_len) if head_max_len else budgets["head_max_len"]

    router = pool().get(device)
    kwargs: Dict[str, Any] = {"max_len": resolved_max_len, "head_max_len": resolved_head_max_len}
    if lang:
        kwargs["lang"] = lang
    if checkpoint and checkpoint != "auto":
        kwargs["model"] = checkpoint

    started = time.perf_counter()
    result = router.predict({"problem": problem}, rotations_questions, **kwargs)
    latency_ms = (time.perf_counter() - started) * 1000.0

    answers = result.get("answers") or {}
    totals = {key: 0.0 for key in keys}
    per_rotation: List[Dict[str, Any]] = []
    for qid, order in orders.items():
        answer = answers.get(qid) or {}
        probabilities = answer.get("probabilities") or {}
        for key in keys:
            totals[key] += float(probabilities.get(key, 0.0)) / k
        per_rotation.append(
            {
                "rotation": int(qid[1:]) if qid[1:].isdigit() else qid,
                "order": order,
                "choice": answer.get("choice"),
                "probabilities": {key: float(probabilities.get(key, 0.0)) for key in keys},
                "confidence": answer.get("confidence"),
            }
        )

    mean_ranked = sorted(
        ({"key": key, "text": criteria[key], "prob": totals[key]} for key in keys),
        key=lambda item: item["prob"],
        reverse=True,
    )
    mediated = mean_ranked[0]["key"] if mean_ranked else None
    votes = {}
    for entry in per_rotation:
        votes[entry["choice"]] = votes.get(entry["choice"], 0) + 1
    agreement = votes.get(mediated, 0) / k if k else 0.0

    if agreement == 1.0:
        verdict = "stabile"
    elif agreement >= 0.75:
        verdict = "abbastanza stabile"
    elif agreement >= 0.5:
        verdict = "instabile"
    else:
        verdict = "molto instabile: non fidarti di questa scelta"

    return {
        "device": normalise_device(device),
        "checkpoint": (result.get("routing") or {}).get("model"),
        "routing": result.get("routing"),
        "budgets": {"max_len": resolved_max_len, "head_max_len": resolved_head_max_len},
        "rotations": k,
        "n_solutions": n,
        "latency_ms": round(latency_ms, 1),
        "per_rotation": per_rotation,
        "votes": votes,
        "mediated": mediated,
        "mean_ranked": mean_ranked,
        "agreement": round(agreement, 3),
        "distinct_choices": len(votes),
        "verdict": verdict,
        "note": (
            "la probabilita' mediata e' la media delle probabilita' per opzione su tutte le "
            "rotazioni: e' piu' stabile della singola risposta, non piu' calibrata"
        ),
    }


# --------------------------------------------------------------------------------------
# Stato dell'ambiente
# --------------------------------------------------------------------------------------


_TORCH_INFO: Optional[Dict[str, Any]] = None


def _torch_info() -> Dict[str, Any]:
    global _TORCH_INFO
    if _TORCH_INFO is not None:
        return _TORCH_INFO

    info: Dict[str, Any] = {
        "torch": None,
        "cuda_available": False,
        "gpu_name": None,
        "vram_total_mb": None,
        "vram_free_mb": None,
        "threads": None,
        "error": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["threads"] = int(torch.get_num_threads())
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            info["gpu_name"] = props.name
            info["vram_total_mb"] = int(props.total_memory / (1024 * 1024))
            free, _total = torch.cuda.mem_get_info(index)
            info["vram_free_mb"] = int(free / (1024 * 1024))
    except Exception as exc:  # pragma: no cover - dipende dall'ambiente
        info["error"] = f"{type(exc).__name__}: {exc}"

    _TORCH_INFO = info
    return info


def status() -> Dict[str, Any]:
    # La VRAM libera cambia durante la sessione: il resto e' statico e resta in cache.
    torch_info = dict(_torch_info())
    if torch_info.get("cuda_available"):
        try:
            import torch

            free, _total = torch.cuda.mem_get_info(torch.cuda.current_device())
            torch_info["vram_free_mb"] = int(free / (1024 * 1024))
        except Exception:  # pragma: no cover
            pass
    try:
        import laya

        laya_version = laya.__version__
    except Exception as exc:  # pragma: no cover
        laya_version = f"non importabile ({type(exc).__name__})"
    try:
        # Sotto lo stesso lock del preimport: senza, questo `import transformers` puo'
        # correre contro il primo import di un thread di warmup.
        with _IMPORT_LOCK:
            import transformers

        transformers_version = transformers.__version__
    except Exception:  # pragma: no cover
        transformers_version = None

    return {
        "python": sys.version.split()[0],
        "laya": laya_version,
        "transformers": transformers_version,
        "torch": torch_info,
        "threads": torch_info.get("threads"),
        "hf_cache": str(HF_CACHE_DIR),
        "hf_cache_exists": HF_CACHE_DIR.exists(),
        "token_present": bool(hf_token()),
        "warmup": pool().warm_state(),
        "loaded": pool().loaded(),
        "router_init_seconds": pool().router_init_seconds(),
        "errors": pool().errors(),
        "defaults": {
            "device": normalise_device(None),
            "auto_tournament_at": AUTO_TOURNAMENT_AT,
            "group_size": DEFAULT_GROUP_SIZE,
            "max_rotations": MAX_ROTATIONS,
            "instructions_en": INSTRUCTIONS_EN,
            "instructions_it": INSTRUCTIONS_IT,
            "quality_levels": QUALITY_LEVELS,
        },
    }
