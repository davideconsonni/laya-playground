"""Laya playground engine.

Single source of truth for three things:

* the **device pool** -- one ``Router`` per device (``cpu`` / ``cuda`` / ...), built on first
  use and reused afterwards, so the CPU/GPU toggle in the UI does not pay the load on every
  request;
* the **question construction** -- the problem is the *state*, each candidate solution is an
  option of a ``choice`` question with neutral keys (``s00``, ``s01``, ...). The keys are
  opaque on purpose: the checkpoints follow labels like ``true``/``false`` literally instead
  of the option text;
* the **test modes** -- ``simple`` (a single question), ``tournament`` (group elimination,
  handles up to 256 solutions), ``long`` (problem longer than the window) and ``stability``
  (k rotations of the option order, averaged together).

No module-level import of ``torch`` or ``laya``: both are heavy and this file is imported by
the server at startup.
"""

from __future__ import annotations

import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# --------------------------------------------------------------------------------------
# Paths and configuration
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
STATIC_DIR = SRC_DIR / "static"


def _load_dotenv(path: Path) -> None:
    """Load a minimal ``.env`` (KEY=VALUE) without depending on python-dotenv."""
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

# The weights land inside the project, not in the user cache: that way everything can be
# deleted together. This must be set before anything imports huggingface_hub.
_HF_CACHE = os.environ.get("LAYA_HF_CACHE", ".hf-cache")
HF_CACHE_DIR = (ROOT / _HF_CACHE).resolve() if not Path(_HF_CACHE).is_absolute() else Path(_HF_CACHE)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))

# The checkpoint is not an LLM: it does not "explain" and it does not reason. It chooses, and its
# instructions were trained to be short and direct ("Which department should handle this?"), so
# the default is short too. The long, constraint-heavy wording is kept as a preset: it is
# off-distribution for the base checkpoints and reads as a reasoning request the model cannot do.
INSTRUCTIONS_EN = "Which candidate solution best solves the problem described in the state?"
INSTRUCTIONS_IT = "Quale soluzione candidata risolve meglio il problema descritto nello stato?"

# The long, constraint-heavy wording (preset, off-distribution for the base checkpoints).
INSTRUCTIONS_LONG_EN = (
    "Which candidate fully eliminates the problem under all stated constraints, including "
    "restarts, concurrency and late retries, and does not merely reduce its probability?"
)

# The first wording, kept as a preset so a change can be measured instead of felt.
INSTRUCTIONS_LEGACY_EN = (
    "Which candidate solution best solves the problem described in the state? "
    "Pick the single most effective and practically applicable one."
)

# Instruction variants the UI shows in its preset menu. Labels are Italian (what the user reads);
# the instruction text stays English (the model's training language), editable in the textarea.
INSTRUCTION_PRESETS = [
    {"id": "best", "label": "Migliore (default, corta)", "text": INSTRUCTIONS_EN},
    {"id": "eliminates", "label": "Elimina il problema (vincoli completi)", "text": INSTRUCTIONS_LONG_EN},
    {"id": "legacy", "label": "Versione precedente (efficace e applicabile)", "text": INSTRUCTIONS_LEGACY_EN},
    {
        "id": "correctness",
        "label": "Correttezza forte",
        "text": (
            "Which candidate solution is most likely to be fully correct? Favour correctness "
            "over speed, cost or simplicity."
        ),
    },
    {
        "id": "low_risk",
        "label": "Minimo rischio",
        "text": (
            "Which candidate solution carries the lowest risk of making things worse? Prefer the "
            "option with the smallest blast radius, even when its upside is smaller."
        ),
    },
    {
        "id": "fast",
        "label": "Rapidita' di intervento",
        "text": (
            "Which candidate solution can be deployed and start helping the soonest, with the "
            "fewest moving parts?"
        ),
    },
]

QUALITY_INSTRUCTIONS = "How good is the best solution available for this problem?"
QUALITY_LEVELS = ["useless", "weak", "workable", "good", "optimal"]

COMPLETE_INSTRUCTIONS = "Does at least one candidate solution fully solve the problem?"
COMPLETE_CRITERIA = {
    "false": "no candidate solution fully solves the problem",
    "true": "at least one candidate solution fully solves the problem",
}

# Below this threshold a single `choice` question is enough; above it the tournament pays off.
AUTO_TOURNAMENT_AT = 20
# Tournament group size (the library default is 16).
DEFAULT_GROUP_SIZE = 16
# Beyond this number the stability mode costs more than it tells you.
MAX_ROTATIONS = 32
# Fixed seed for the stability rotations: random permutations, reproducible run to run.
STABILITY_SEED = 0
# Suggested abstention threshold shown in the UI when the user leaves `min_confidence` empty.
# It is a nudge, not a calibrated value: the shipped checkpoints are over-confident.
SUGGESTED_MIN_CONFIDENCE = 0.5

# Thresholds of the composite reliability flag (see `reliability`). Heuristics, not calibration.
RELIABILITY_AGREEMENT_OK = 0.75
RELIABILITY_AGREEMENT_CAUTION = 0.5
RELIABILITY_CONFIDENCE_LOW = 0.34
RELIABILITY_GOOD_LEVELS = ("good", "optimal")
RELIABILITY_WEAK_LEVELS = ("useless", "weak")

STRATEGIES = ("auto", "simple", "tournament", "long", "per_solution")

# Checkpoints the playground can route to, and the aliases Laya accepts for them. Kept here so
# `/v1/models` and the UI share one list; the playground does not register extra checkpoints.
CHECKPOINTS: List[str] = ["english", "multilingual", "typed-decisions"]
CHECKPOINT_ALIASES: Dict[str, str] = {
    "en": "english", "laya": "english", "default": "english",
    "multi": "multilingual", "ml": "multilingual", "laya-multilingual": "multilingual",
    "typed": "typed-decisions", "typed_decisions": "typed-decisions", "decisions": "typed-decisions",
}

# The per-solution strategy asks one short yes/no choice per candidate instead of one holistic
# choice over all of them: a judgment the base checkpoints handle better than "which is best".
# Neutral keys ("A"/"B") on purpose, so the option labels cannot dominate the answer.
PER_SOLUTION_INSTRUCTIONS = "Does this candidate solution fully eliminate the problem described in the state?"
PER_SOLUTION_CRITERIA = {
    "A": "yes, it fully eliminates the problem",
    "B": "no, it does not",
}


def hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return token.strip() or None if token else None


def clean_solutions(raw: Iterable[str]) -> List[str]:
    """One solution per line: drop empty lines, keep the order, remove duplicates."""
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
# Device pool
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
    """One ``Router`` per device, built on first use and reused afterwards.

    ``Router(device=...)`` decides the backend once per object, so the CPU/GPU toggle needs
    two instances. Keeping both alive is what makes the switch instant and what makes the
    CPU-vs-GPU comparison in the UI possible.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._routers: Dict[str, Any] = {}
        self._errors: Dict[str, str] = {}
        self._init_seconds: Dict[str, float] = {}
        self._warm: Dict[str, Dict[str, Any]] = {}

    # -- state -------------------------------------------------------------------------

    def warm_state(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {device: dict(info) for device, info in self._warm.items()}

    def loaded(self) -> Dict[str, List[str]]:
        with self._lock:
            # `Router.loaded` is a property, not a method.
            return {device: list(router.loaded) for device, router in self._routers.items()}

    def router_init_seconds(self) -> Dict[str, float]:
        """What the *construction* of the Router cost (which is lazy: the weights are not)."""
        with self._lock:
            return dict(self._init_seconds)

    def errors(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._errors)

    # -- construction ------------------------------------------------------------------

    def get(self, device: Optional[str]) -> Any:
        """Return the Router for the requested device, building it if needed."""
        key = normalise_device(device)
        with self._lock:
            router = self._routers.get(key)
            if router is not None:
                return router
            failure = self._errors.get(key)
            if failure is not None:
                raise RuntimeError(failure)

        # Built outside the lock: the build downloads the weights and can take minutes, and
        # we do not want to block the toggle of the other device in the meantime.
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
        except Exception as exc:  # pragma: no cover - depends on the environment
            message = f"device '{key}': non riesco a costruire il Router ({type(exc).__name__}: {exc})"
            with self._lock:
                self._errors[key] = message
            raise RuntimeError(message) from exc

        with self._lock:
            self._routers[key] = router
            self._init_seconds[key] = seconds
            # Do not clobber a warmup in progress: whoever started it writes "ready" at the
            # end of the preload, and overwriting it here would make it look like the weights
            # are already resident.
            if self._warm.get(key, {}).get("state") != "loading":
                self._warm[key] = {"state": "ready", "seconds": round(seconds, 3), "detail": None}
        return router

    # -- background preload --------------------------------------------------------------

    def start_warmup(self, device: Optional[str], checkpoints: Optional[List[str]] = None) -> Dict[str, Any]:
        """Start the load/preload in a thread and return immediately.

        The first ``predict`` downloads a checkpoint from the Hub: it can take minutes, and the
        UI must be able to show it instead of looking stuck.

        The default is only ``multilingual``: that is where routing drops a non-English Latin
        text, and preloading ``english`` too would download ~1.7 GB more that nobody would use
        in a case like this (``english`` is still loaded lazily on first use).
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
            except Exception as exc:  # pragma: no cover - depends on the environment
                with self._lock:
                    self._warm[key] = {
                        "state": "error",
                        "seconds": round(time.perf_counter() - started, 2),
                        "detail": f"{type(exc).__name__}: {exc}",
                    }

        threading.Thread(target=_worker, name=f"laya-warmup-{key}", daemon=True).start()
        return {"started": True, "state": "loading"}

    def unload(self, device: Optional[str] = None) -> List[str]:
        """Free the memory of the checkpoints (the Router stays, the weights do not)."""
        key = normalise_device(device) if device else None
        freed: List[str] = []
        with self._lock:
            targets = [key] if key else list(self._routers)
        for name in targets:
            with self._lock:
                router = self._routers.get(name)
                # `Router.unload()` frees but returns nothing: we read the names beforehand.
                resident = list(router.loaded) if router is not None else []
            if router is not None:
                router.unload()
                freed.extend(resident)
        return freed


_POOL = DevicePool()

# transformers 5.x resolves names lazily: if two threads start the import at the same moment,
# one can see the module half-built and fail with
# `cannot import name 'AutoTokenizer' from 'transformers'`. Serialising the first import
# (after that, `import transformers` is just a sys.modules lookup) removes the race.
_IMPORT_LOCK = threading.RLock()
_preimported: Optional[str] = None


def _auto_threads() -> int:
    """Number of threads to use for CPU inference.

    In a container torch sees the **host** cores, not the limit given to Docker: on a big
    machine a 2-vCPU container would end up with 64 threads and become very slow because of
    over-subscription. So: respect the cgroup CPU limit if there is one, otherwise take the
    number of cores, capped at 16 (the maximum measured in the laya README).
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
    """Import once, under lock, torch/transformers/laya and the lazy names of transformers."""
    global _preimported
    with _IMPORT_LOCK:
        if _preimported is not None:
            return _preimported
        import torch
        import transformers

        from laya import Agent, Router  # noqa: F401

        for name in ("AutoTokenizer", "AutoConfig", "AutoModel"):
            getattr(transformers, name, None)  # force the lazy module to resolve them now

        # LAYA_THREADS wins; without it, it is computed from the cgroup/core count (see
        # _auto_threads).
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
# Question and budget
# --------------------------------------------------------------------------------------


def build_questions(
    solutions: List[str],
    *,
    instructions: Optional[str] = None,
    with_quality: bool = False,
    with_complete: bool = False,
) -> Dict[str, Any]:
    """Build the dictionary of typed questions for a list of solutions."""
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
    """Sensible token budgets for a problem with ``n_options`` solutions.

    In Laya the options share a budget (``head_max_len``) and each option below ~4 tokens
    becomes indistinguishable from the others: beyond ~20 options the defaults (192/256) truncate
    the descriptions. Here the budget is widened proportionally, with a cap.
    """
    base_head, base_max = (192, 512) if checkpoint == "english" else (256, 1024)
    head = min(max(base_head, 16 + 24 * max(1, n_options)), 4096)
    max_len = min(max(base_max, head + 512), 8192)
    return {"head_max_len": int(head), "max_len": int(max_len)}


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------


def _quality_criteria() -> Dict[str, str]:
    """Map of the `score` levels, valid whether the probabilities are by index or by text."""
    mapping = {str(index): level for index, level in enumerate(QUALITY_LEVELS)}
    mapping.update({level: level for level in QUALITY_LEVELS})
    return mapping


def _answer_view(
    answer: Dict[str, Any], criteria: Dict[str, str], *, compute_eliminated: bool = True
) -> Dict[str, Any]:
    """Normalise the answer of a question into something the UI reads easily."""
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
    """Real device and dtype of the checkpoint that answered."""
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
    """Approximate warning when the problem risks being truncated."""
    room = max_len - head_max_len - 1
    approx_tokens = max(1, len(problem) // 4)
    if approx_tokens > room:
        return (
            f"il problema e' lungo ~{approx_tokens} token stimati ma restano ~{room} token "
            f"dopo le opzioni: passa a strategia 'long' per leggerlo tutto"
        )
    return None


def _per_solution_questions(
    solutions: List[str],
    *,
    instructions: Optional[str] = None,
    with_quality: bool = False,
    with_complete: bool = False,
) -> Dict[str, Any]:
    """One short yes/no `choice` per candidate, plus the optional type-2 questions.

    Every question rides in the same forward pass. The candidate text goes in the instructions
    (there is no per-question state), the criteria stay neutral ("A"/"B") so the labels cannot
    dominate, and the answer is ranked by P("A") = P(the candidate eliminates the problem).
    """
    base = (instructions or PER_SOLUTION_INSTRUCTIONS).strip()
    questions: Dict[str, Any] = {}
    for i, text in enumerate(solutions):
        questions[f"q{i:02d}"] = {
            "type": "choice",
            "instructions": f"{base} Candidate: {text}",
            "criteria": dict(PER_SOLUTION_CRITERIA),
        }
    # Reuse the extra questions exactly as the single-choice path builds them.
    extra = build_questions(solutions, with_quality=with_quality, with_complete=with_complete)
    for qid in ("quality", "complete"):
        if qid in extra:
            questions[qid] = extra[qid]
    return questions


def _per_solution_view(
    answers: Dict[str, Any], solutions: List[str]
) -> Dict[str, Any]:
    """Rank the candidates by P("A") from the per-solution answers, in the usual view shape."""
    ranked = []
    for i, text in enumerate(solutions):
        answer = answers.get(f"q{i:02d}") or {}
        probabilities = answer.get("probabilities") or {}
        p_yes = float(probabilities.get("A", probabilities.get("yes", 0.0)))
        ranked.append({"key": f"s{i:02d}", "text": text, "prob": p_yes})
    ranked.sort(key=lambda item: item["prob"], reverse=True)
    top = ranked[0] if ranked else None
    return {
        "choice": top["key"] if top else None,
        "top": top,
        "ranked": ranked,
        "eliminated": [],
        "confidence": None,
        "answer_confidence": top["prob"] if top else None,
        "low_confidence": None,
        "abstention": None,
        "abstention_threshold": None,
    }


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
    """Choose the best solution among the given ones, on a single device."""
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

    n = len(solutions)
    if strategy == "per_solution":
        # One short yes/no question per candidate, answered in the same forward pass.
        questions = _per_solution_questions(
            solutions, instructions=instructions,
            with_quality=with_quality, with_complete=with_complete,
        )
        criteria: Dict[str, str] = {}
    else:
        questions = build_questions(
            solutions, instructions=instructions, with_quality=with_quality, with_complete=with_complete
        )
        criteria = questions["best"]["criteria"]

    budgets = suggest_budgets(n, checkpoint if checkpoint != "auto" else "multilingual")
    resolved_max_len = int(max_len) if max_len else budgets["max_len"]
    resolved_head_max_len = int(head_max_len) if head_max_len else budgets["head_max_len"]

    router = pool().get(device)
    state = {"problem": problem}

    # `predict_long` accepts neither the budgets (`window` replaces them) nor `min_confidence`:
    # the common kwargs and the budget kwargs stay separate.
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
    if strategy == "per_solution":
        view = _per_solution_view(answers, solutions)
    else:
        view = _answer_view(answers.get("best") or {}, criteria)
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
    if strategy == "tournament" and n <= int(group_size):
        notes.append(
            f"tournament inattivo (0 round): con {n} soluzioni e group_size {int(group_size)} "
            "equivale alla strategia Semplice"
        )
    if strategy == "per_solution":
        notes.append(
            f"giudizio per soluzione: {n} domande si'/no in una sola passata, classifica per "
            "P(elimina il problema); le istruzioni impostate sono il prefisso di ogni domanda"
        )

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


def _rotation_cap(n: int) -> int:
    """Largest number of *distinct* permutations of n slots we would ever ask for."""
    total = 1
    for i in range(2, n + 1):
        total *= i
        if total >= MAX_ROTATIONS:
            return MAX_ROTATIONS
    return total


def stability(
    *,
    problem: str,
    solutions: List[str],
    device: Optional[str] = None,
    checkpoint: str = "auto",
    rotations: Optional[int] = None,
    seed: Optional[int] = None,
    max_len: Optional[int] = None,
    head_max_len: Optional[int] = None,
    instructions: Optional[str] = None,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """Test the stability of the choice by rotating the option order.

    Laya has a documented position bias: the same option in different slots receives different
    logits. Here the same question is asked ``k`` times with a rotated ``option_order``, all in
    **a single forward pass**, and it reports both the mediated choice and how many times the
    choice changed. It is the test that tells you whether to trust that decision.
    """
    problem = (problem or "").strip()
    solutions = clean_solutions(solutions)
    n = len(solutions)
    if not problem:
        raise ValueError("il problema e' vuoto")
    if n < 2:
        raise ValueError("servono almeno due soluzioni candidate")

    k = int(rotations) if rotations else min(n, 4)
    k = max(1, min(k, MAX_ROTATIONS, _rotation_cap(n)))
    seed = STABILITY_SEED if seed is None else int(seed)

    base = build_questions(solutions, instructions=instructions)["best"]
    criteria = base["criteria"]
    keys = list(criteria)

    # Random permutations, not just the cyclic shifts: a cyclic set only moves each option through
    # a fixed sequence of slots. With k <= n distinct permutations always exist, so the retry
    # finds a fresh one. `random.Random(seed)` keeps the rotation set reproducible.
    rng = random.Random(seed)
    rotations_questions: Dict[str, Any] = {}
    orders: Dict[str, List[int]] = {}
    seen: set = set()
    for r in range(k):
        order = list(range(n))
        for _ in range(64):
            rng.shuffle(order)
            if tuple(order) not in seen:
                break
        seen.add(tuple(order))
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

    distribution = sorted(
        (
            {"choice": choice, "count": count, "share": round(count / k, 3)}
            for choice, count in votes.items()
        ),
        key=lambda item: item["count"],
        reverse=True,
    )

    return {
        "device": normalise_device(device),
        "checkpoint": (result.get("routing") or {}).get("model"),
        "routing": result.get("routing"),
        "budgets": {"max_len": resolved_max_len, "head_max_len": resolved_head_max_len},
        "rotations": k,
        "seed": seed,
        "n_solutions": n,
        "latency_ms": round(latency_ms, 1),
        "per_rotation": per_rotation,
        "votes": votes,
        "distribution": distribution,
        "mediated": mediated,
        "mean_ranked": mean_ranked,
        "agreement": round(agreement, 3),
        "distinct_choices": len(votes),
        "verdict": verdict,
        "note": (
            "ogni rotazione e' una permutazione casuale degli slot (seed fisso, riproducibile): "
            "la probabilita' mediata e' la media delle probabilita' per opzione su tutte le "
            "rotazioni ed e' piu' stabile della singola risposta, non piu' calibrata"
        ),
    }


# --------------------------------------------------------------------------------------
# Composite reliability flag
# --------------------------------------------------------------------------------------


def reliability(
    solve_result: Dict[str, Any],
    stability_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One flag combining confidence, rotation agreement and score/noul coherence.

    It is a **heuristic**: Laya exposes probabilities and a confidence, and this is a rule of
    thumb over them, not a calibration and not a probability. The ``reasons`` list says which rule
    fired, so the caller can disagree with it; read it together with the raw numbers.
    """
    order = {"ok": 0, "caution": 1, "unreliable": 2}
    level = "ok"
    reasons: List[str] = []

    def bump(new: str) -> None:
        nonlocal level
        if order[new] > order[level]:
            level = new

    if solve_result.get("abstention") == "abstained":
        bump("unreliable")
        reasons.append("il gate di astensione ha rifiutato la scelta")
    if solve_result.get("low_confidence"):
        bump("caution")
        reasons.append("low_confidence: il gate interno segnala confidenza bassa")

    answer_confidence = solve_result.get("answer_confidence")
    if (
        isinstance(answer_confidence, (int, float))
        and not isinstance(answer_confidence, bool)
        and float(answer_confidence) < RELIABILITY_CONFIDENCE_LOW
    ):
        bump("caution")
        reasons.append(f"answer_confidence bassa ({float(answer_confidence):.3f})")

    agreement: Optional[float] = None
    if stability_result:
        raw = stability_result.get("agreement")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            agreement = float(raw)
            if agreement < RELIABILITY_AGREEMENT_CAUTION:
                bump("unreliable")
                reasons.append(f"accordo tra rotazioni basso ({agreement:.0%})")
            elif agreement < RELIABILITY_AGREEMENT_OK:
                bump("caution")
                reasons.append(f"accordo tra rotazioni medio ({agreement:.0%})")
        distinct = stability_result.get("distinct_choices")
        if isinstance(distinct, int) and not isinstance(distinct, bool) and distinct > 2:
            bump("caution")
            reasons.append(f"{distinct} scelte distinte tra le rotazioni")
    else:
        reasons.append("stabilita' non testata: giudizio basato su confidenza e coerenza")

    extras = solve_result.get("extras") or {}
    quality = extras.get("quality") or {}
    q_level = (quality.get("top") or {}).get("key")
    if q_level in RELIABILITY_GOOD_LEVELS:
        if agreement is not None and agreement < RELIABILITY_AGREEMENT_OK:
            bump("unreliable")
            reasons.append(
                f"score '{q_level}' ma la scelta non e' stabile: confidenza non affidabile"
            )
    elif q_level in RELIABILITY_WEAK_LEVELS:
        bump("caution")
        reasons.append(f"score '{q_level}': la migliore soluzione disponibile e' debole")

    complete = extras.get("complete")
    noul = complete.get("noul") if isinstance(complete, dict) else None
    if (
        isinstance(noul, (int, float))
        and not isinstance(noul, bool)
        and float(noul) < 0.5
    ):
        reasons.append(f"noul: nessuna soluzione risolve del tutto (P(si)={float(noul):.3f})")

    labels = {"ok": "affidabile", "caution": "da verificare", "unreliable": "poco affidabile"}
    return {
        "level": level,
        "label": labels[level],
        "reasons": reasons,
        "inputs": {
            "answer_confidence": answer_confidence,
            "low_confidence": solve_result.get("low_confidence"),
            "abstention": solve_result.get("abstention"),
            "agreement": agreement,
            "stability_tested": bool(stability_result),
            "score_level": q_level,
            "noul": noul,
        },
        "note": (
            "euristica su confidenza + accordo tra rotazioni + coerenza con score/noul: non e' "
            "una probabilita' ne' una calibrazione"
        ),
    }


# --------------------------------------------------------------------------------------
# Environment state
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
    except Exception as exc:  # pragma: no cover - depends on the environment
        info["error"] = f"{type(exc).__name__}: {exc}"

    _TORCH_INFO = info
    return info


def status() -> Dict[str, Any]:
    # Free VRAM changes during the session: the rest is static and stays cached.
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
        # Under the same lock as the preimport: without it, this `import transformers` can
        # race against the first import of a warmup thread.
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
            "instruction_presets": INSTRUCTION_PRESETS,
            "suggested_min_confidence": SUGGESTED_MIN_CONFIDENCE,
            "quality_levels": QUALITY_LEVELS,
        },
    }
