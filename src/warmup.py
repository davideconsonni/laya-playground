"""Pre-scarica i checkpoint e fa uno smoke test su CPU e/o GPU.

    uv run python src/warmup.py --device both
    uv run python src/warmup.py --device cuda --full

Il primo `predict` scarica un checkpoint dal Hub (centinaia di MB, minuti): questo script lo
fa una volta sola, cosi' la UI risponde subito dopo. Con `--full` prova anche il tournament
su 30 soluzioni e la modalita' stabilita'.
"""

from __future__ import annotations

import argparse
import sys
import time

import engine

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
    print(f"  checkpoint     : {result.get('checkpoint')}  ({result.get('dtype') or 'dtype n/d'})")
    print(f"  routing        : {result.get('reason')}")
    print(f"  strategia      : {result.get('strategy')}  ({result.get('strategy_reason')})")
    print(f"  budget token   : {result.get('budgets')}")
    print(f"  latenza        : {result.get('latency_ms')} ms")
    print(f"  scelta         : {top.get('key')}  p={top.get('prob'):.4f}")
    print(f"                   {(top.get('text') or '')[:90]}")
    for row in (result.get("ranked") or []):
        bar = "#" * max(1, int(round(row["prob"] * 24)))
        print(f"    {row['key']}  {row['prob']*100:6.2f}%  {bar:<24}")
    for note in result.get("notes") or []:
        print(f"  nota           : {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Warmup e smoke test del playground Laya")
    parser.add_argument("--device", default="cuda", help="cuda | cpu | both | auto")
    parser.add_argument("--checkpoint", default="auto",
                        help="auto | english | multilingual | typed-decisions")
    parser.add_argument("--full", action="store_true",
                        help="prova anche tournament (30 soluzioni) e stabilita'")
    args = parser.parse_args()

    info = engine.status()
    torch_info = info["torch"]
    print(f"python         : {info['python']}")
    print(f"laya           : {info['laya']}   transformers: {info['transformers']}")
    print(f"torch          : {torch_info['torch']}")
    print(f"cuda           : available={torch_info['cuda_available']}  "
          f"gpu={torch_info['gpu_name']}  vram={torch_info['vram_total_mb']} MB")
    print(f"token HF       : {'presente' if info['token_present'] else 'assente (ok, i checkpoint sono pubblici)'}")
    print(f"cache HF       : {info['hf_cache']}")
    print()

    devices = ["cuda", "cpu"] if args.device == "both" else [args.device]
    failures = 0

    for device in devices:
        print(f"=== device: {device} ===")
        if device == "cuda" and not torch_info["cuda_available"]:
            print("  saltato: torch non vede la CUDA (build CPU?)")
            failures += 1
            continue
        try:
            started = time.perf_counter()
            engine.pool().get(device)
            print(f"  Router costruito in {time.perf_counter() - started:.1f} s")
        except Exception as exc:
            print(f"  ERRORE: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        try:
            started = time.perf_counter()
            result = engine.solve(problem=PROBLEM, solutions=SOLUTIONS, device=device,
                                  checkpoint=args.checkpoint)
            print(f"  (primo predict, incluso download/load: {time.perf_counter() - started:.1f} s)")
            _report(device, result)

            # secondo giro: a caldo, senza load
            warm = engine.solve(problem=PROBLEM, solutions=SOLUTIONS, device=device,
                                checkpoint=args.checkpoint)
            print(f"  latenza a caldo: {warm['latency_ms']} ms")
        except Exception as exc:
            print(f"  ERRORE nel predict: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        if args.full:
            try:
                many = engine.solve(problem=PROBLEM, solutions=MANY_SOLUTIONS, device=device,
                                    checkpoint=args.checkpoint, strategy="auto")
                print(f"  tournament su 30 soluzioni: strategia={many['strategy']} "
                      f"latenza={many['latency_ms']} ms "
                      f"finalisti={len((many.get('tournament') or {}).get('finalists') or [])}")
                print(f"    scelta: {(many.get('top') or {}).get('key')} "
                      f"p={(many.get('top') or {}).get('prob')}")
            except Exception as exc:
                print(f"  ERRORE nel tournament: {type(exc).__name__}: {exc}")
                failures += 1

            try:
                stab = engine.stability(problem=PROBLEM, solutions=SOLUTIONS, device=device,
                                        checkpoint=args.checkpoint, rotations=4)
                print(f"  stabilita' su 4 rotazioni: {stab['verdict']} "
                      f"(accordo {stab['agreement']:.0%}, {stab['distinct_choices']} scelte distinte, "
                      f"{stab['latency_ms']} ms)")
            except Exception as exc:
                print(f"  ERRORE nella stabilita': {type(exc).__name__}: {exc}")
                failures += 1
        print()

    print("ESITO:", "tutto ok" if failures == 0 else f"{failures} problemi")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
