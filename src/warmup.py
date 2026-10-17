"""Pre-download the checkpoints and run a smoke test on CPU and/or GPU.

    uv run python src/warmup.py --device both
    uv run python src/warmup.py --device cuda --full

The first `predict` downloads a checkpoint from the Hub (hundreds of MB, minutes): this script
does it once, so the UI answers immediately afterwards. With `--full` it also tries the
tournament on 30 solutions and the stability mode.
"""

from __future__ import annotations

import argparse
import sys
import time

import engine

# Sample text in Italian on purpose: it exercises the multilingual checkpoint and the
# short-text language-detection quirk described in the README.
PROBLEM = (
    "Il servizio di pagamento va in timeout sotto carico e il cliente minaccia di andarsene. "
    "Il database condiviso e' gia' al 90% di utilizzo e il team ha due settimane di tempo."
)

SOLUTIONS = [
    "Aggiungere un indice alla tabella delle transazioni e rivedere le query piu' lente.",
    "Migrare il servizio su un database dedicato, separato da quello condiviso.",
    "Introdurre una cache Redis davanti alle letture piu' frequenti.",
    "Aumentare il timeout e riprovare automaticamente le richieste fallite.",
    "Assumere un consulente esterno che riscriva il servizio da zero.",
]

MANY_SOLUTIONS = [f"Soluzione candidata numero {i}: intervento tecnico alternativo sul servizio." for i in range(1, 31)]


def _report(device: str, result: dict) -> None:
    top = result.get("top") or {}
    print(f"  checkpoint     : {result.get('checkpoint')}  ({result.get('dtype') or 'dtype n/a'})")
    print(f"  routing        : {result.get('reason')}")
    print(f"  strategy       : {result.get('strategy')}  ({result.get('strategy_reason')})")
    print(f"  token budget   : {result.get('budgets')}")
    print(f"  latency        : {result.get('latency_ms')} ms")
    print(f"  choice         : {top.get('key')}  p={top.get('prob'):.4f}")
    print(f"                   {(top.get('text') or '')[:90]}")
    for row in (result.get("ranked") or []):
        bar = "#" * max(1, int(round(row["prob"] * 24)))
        print(f"    {row['key']}  {row['prob']*100:6.2f}%  {bar:<24}")
    for note in result.get("notes") or []:
        print(f"  note           : {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Warmup and smoke test for the Laya playground")
    parser.add_argument("--device", default="cuda", help="cuda | cpu | both | auto")
    parser.add_argument("--checkpoint", default="auto",
                        help="auto | english | multilingual | typed-decisions")
    parser.add_argument("--full", action="store_true",
                        help="also try the tournament (30 solutions) and stability")
    args = parser.parse_args()

    info = engine.status()
    torch_info = info["torch"]
    print(f"python         : {info['python']}")
    print(f"laya           : {info['laya']}   transformers: {info['transformers']}")
    print(f"torch          : {torch_info['torch']}")
    print(f"cuda           : available={torch_info['cuda_available']}  "
          f"gpu={torch_info['gpu_name']}  vram={torch_info['vram_total_mb']} MB")
    print(f"HF token       : {'present' if info['token_present'] else 'absent (ok, the checkpoints are public)'}")
    print(f"HF cache       : {info['hf_cache']}")
    print()

    devices = ["cuda", "cpu"] if args.device == "both" else [args.device]
    failures = 0

    for device in devices:
        print(f"=== device: {device} ===")
        if device == "cuda" and not torch_info["cuda_available"]:
            print("  skipped: torch does not see CUDA (CPU build?)")
            failures += 1
            continue
        try:
            started = time.perf_counter()
            engine.pool().get(device)
            print(f"  Router built in {time.perf_counter() - started:.1f} s")
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        try:
            started = time.perf_counter()
            result = engine.solve(problem=PROBLEM, solutions=SOLUTIONS, device=device,
                                  checkpoint=args.checkpoint)
            print(f"  (first predict, including download/load: {time.perf_counter() - started:.1f} s)")
            _report(device, result)

            # second run: warm, without the load
            warm = engine.solve(problem=PROBLEM, solutions=SOLUTIONS, device=device,
                                checkpoint=args.checkpoint)
            print(f"  warm latency: {warm['latency_ms']} ms")
        except Exception as exc:
            print(f"  ERROR in predict: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        if args.full:
            try:
                many = engine.solve(problem=PROBLEM, solutions=MANY_SOLUTIONS, device=device,
                                    checkpoint=args.checkpoint, strategy="auto")
                print(f"  tournament on 30 solutions: strategy={many['strategy']} "
                      f"latency={many['latency_ms']} ms "
                      f"finalists={len((many.get('tournament') or {}).get('finalists') or [])}")
                print(f"    choice: {(many.get('top') or {}).get('key')} "
                      f"p={(many.get('top') or {}).get('prob')}")
            except Exception as exc:
                print(f"  ERROR in tournament: {type(exc).__name__}: {exc}")
                failures += 1

            try:
                stab = engine.stability(problem=PROBLEM, solutions=SOLUTIONS, device=device,
                                        checkpoint=args.checkpoint, rotations=4)
                print(f"  stability on 4 rotations: {stab['verdict']} "
                      f"(agreement {stab['agreement']:.0%}, {stab['distinct_choices']} distinct choices, "
                      f"{stab['latency_ms']} ms)")
            except Exception as exc:
                print(f"  ERROR in stability: {type(exc).__name__}: {exc}")
                failures += 1
        print()

    print("RESULT:", "all good" if failures == 0 else f"{failures} problems")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
