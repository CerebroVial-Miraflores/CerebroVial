"""Orquestador end-to-end SUMO + TraCI + motor adaptativo HTTP.

CT-07.5: una corrida demostrable enciende SUMO, el motor consume estado,
el motor decide fase, la simulación responde, los KPIs son capturables.

Uso (requiere motor corriendo en otra terminal con `invoke up`):

    python -m cerebrovial_simulation.traci_adapter.run_e2e \\
        --pattern am_peak --seed 1 --end 600

Catches implementados:
- Catch C: flow es tasa de paso (state_reader cuenta arrivals en 30s).
- Catch 3 + A: actuación en borde de ciclo (tllogic_applier).
- Corrección 3: PhaseTiming → 3 sub-fases SUMO.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import traci

from . import tllogic_applier
from .engine_client import EngineUnavailable, recommend
from .phase_aggregator import aggregate_to_phase_flows
from .state_reader import StateTracker


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent.parent
SCENARIOS_DIR = SIM_ROOT / "conf" / "scenarios"
NETWORK_DIR = SIM_ROOT / "conf" / "network"
LINKSTATES_JSON = NETWORK_DIR / "linkstates.json"

INTERSECTION_ID = "larco_schell"  # nodo del seed del core (graph_nodes); ver README §e2e
TLS_ID = "center"

SENSE_INTERVAL_S = 30  # Catch C: ventana de sensado.


def _prepare_addfiles(out_dir: Path) -> list[Path]:
    """Copia add-files al out_dir para que el `file=` relativo resuelva acá."""
    out_dir.mkdir(parents=True, exist_ok=True)
    local: list[Path] = []
    for add_name in ["miraflores_4way.tllogic.add.xml", "edgedata.add.xml", "lanearea.add.xml"]:
        dst = out_dir / add_name
        shutil.copy(NETWORK_DIR / add_name, dst)
        local.append(dst)
    return local


def run_e2e(
    pattern: str,
    seed: int,
    end_s: int,
    out_dir: Path,
    engine_recommend_fn: Optional[Callable] = None,
) -> dict:
    """Orquesta la corrida e2e.

    `engine_recommend_fn` permite inyectar un mock para tests. Si None,
    usa `engine_client.recommend` (HTTP real al motor).

    Retorna dict con estadísticas (n_engine_calls, n_applications,
    engine_modes, total_steps).
    """
    if engine_recommend_fn is None:
        engine_recommend_fn = recommend

    sumo_bin = Path(os.environ["SUMO_HOME"]) / "bin" / "sumo"
    cfg = SCENARIOS_DIR / f"{pattern}.sumocfg"
    add_files = _prepare_addfiles(out_dir)

    sumo_cmd = [
        str(sumo_bin),
        "-c", str(cfg),
        "--additional-files", ",".join(str(p) for p in add_files),
        "--seed", str(seed),
        "--end", str(end_s),
        "--summary-output", str(out_dir / "summary.parquet"),
        "--tripinfo-output", str(out_dir / "tripinfo.parquet"),
        "--no-step-log", "--no-warnings",
    ]

    linkstates = tllogic_applier.load_linkstates(LINKSTATES_JSON)
    applier = tllogic_applier.TllogicApplier(TLS_ID, linkstates)
    tracker = StateTracker(window_s=SENSE_INTERVAL_S)

    n_engine_calls = 0
    engine_modes: list[str] = []
    engine_errors: list[str] = []

    # Traci.start lanza sumo como subprocess y conecta.
    traci.start(sumo_cmd, port=None, label=f"e2e_{pattern}_{seed}")
    try:
        prev_commit_time = 0.0
        step = 0
        while step < end_s:
            traci.simulationStep()
            step += 1
            sim_time = traci.simulation.getTime()

            # Observa estado en cada step (acumula IDs).
            tracker.observe(sim_time)

            # Cada SENSE_INTERVAL_S: cierra ventana, invoca motor.
            if step % SENSE_INTERVAL_S == 0 and step > 0:
                dir_states = tracker.commit_window(sim_time)
                phase_flows = aggregate_to_phase_flows(dir_states)
                try:
                    rec = engine_recommend_fn(
                        intersection_id=INTERSECTION_ID,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        phase_flows=phase_flows,
                        lost_time=10.0,
                    )
                    n_engine_calls += 1
                    engine_modes.append(rec.get("mode", "unknown"))
                    applier.update_pending(rec)
                except EngineUnavailable as exc:
                    engine_errors.append(f"step={step}: {exc}")

            # En cada step, intenta aplicar (solo en borde de ciclo).
            applier.maybe_apply(sim_time)
    finally:
        traci.close()

    return {
        "n_engine_calls": n_engine_calls,
        "n_applications": applier.applied_count,
        "engine_modes": engine_modes,
        "engine_errors": engine_errors,
        "total_steps": step,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True, choices=["am_peak", "pm_peak", "offpeak", "weekend"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--end", type=int, default=600)
    parser.add_argument("--out", default=str(SIM_ROOT / "data" / "_e2e"))
    args = parser.parse_args()

    out_dir = Path(args.out) / f"{args.pattern}_seed{args.seed}"
    stats = run_e2e(
        pattern=args.pattern,
        seed=args.seed,
        end_s=args.end,
        out_dir=out_dir,
    )
    print(f"\nE2E {args.pattern} seed={args.seed} end={args.end}s")
    print(f"  steps:              {stats['total_steps']}")
    print(f"  engine calls:       {stats['n_engine_calls']}")
    print(f"  setProgramLogic:    {stats['n_applications']} (Catch A: ≤ ciclos/{args.end // 60 + 1})")
    print(f"  modes ejercitados:  {set(stats['engine_modes'])}")
    if stats["engine_errors"]:
        print(f"  errores motor:")
        for e in stats["engine_errors"][:5]:
            print(f"    {e}")
    # Catch C: bajo am/pm peak debe rutear a max_pressure al menos una vez.
    if args.pattern in {"am_peak", "pm_peak"} and "max_pressure" not in set(stats["engine_modes"]):
        print(f"\n⚠️  Catch C STOP-CONDITION: bajo {args.pattern} engine "
              f"NUNCA ruteó a max_pressure (modes={set(stats['engine_modes'])}). "
              f"Estimación de flow probablemente subestimada.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
