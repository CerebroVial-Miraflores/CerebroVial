"""Control adaptativo per-node (Max Pressure / Webster) sobre el corredor Larco
vía TraCI ↔ motor HTTP. Experimento IE05: adaptativo vs control fijo.

Generaliza el lazo de TTH-07 de 1 TLS a 2 TLS INDEPENDIENTES (Benavides y Schell;
Diez Canseco es paso libre → se deja con su control fijo, no se adapta). Un solo
proceso SUMO; cada cruce sensa/decide en SU propio borde de ciclo.

REUSA tal cual: engine_client.recommend (POST /control/recommend), y de
tllogic_applier: TllogicApplier (aplica en borde de ciclo) +
expand_timings_to_sumo_phases.

ANTI-ALIASING (lección TTH-07 §7.1): la ventana de sensado = el CICLO COMPLETO
anterior (no 30 s fijos), y la decisión es en el borde de ciclo de cada nodo.
Flujo integrado sobre el ciclo; cola = PROMEDIO del ciclo (decisión B: la
instantánea en el borde tendría sesgo de fase a favor de Larco). Sin desalineación
posible → sin el empate espurio de TTH-07.

CERO core: el motor no se toca, se consume por HTTP en :8001.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import traci

from . import tllogic_applier
from .engine_client import EngineUnavailable, recommend

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent.parent
CONF = SIM_ROOT / "conf" / "corredor_larco"
CFG = CONF / "peak_s_n.sumocfg"
DET = CONF / "corredor_larco.det.add.xml"
LINKSTATES_JSON = CONF / "linkstates_corredor.json"

QUEUE_VEH_LENGTH_M = 7.5     # HCM, igual a TTH-07 (no usado: cola en vehículos directa)
SAT_PER_LANE = 1800.0        # flujo de saturación por carril (veh/h), parámetro físico estándar
LAST_SUBPHASE_IDX = 5        # 6 sub-fases: LARCO_g,y,r,TRANSV_g,y,r → última = TRANSV_r
ALLRED_PHASE_IDX = (2, 5)    # LARCO_r y TRANSV_r (para contabilizar tiempo perdido en all-red)


# --- Config per-node. Saturación = (carriles generales que sirve la fase) × SAT_PER_LANE.
#     Carril 0 del eje Larco es bus/bici (excluido). Ver demand_params / net.xml. ---
@dataclass
class PhaseCfg:
    phase_id: str           # "LARCO" | "TRANSV"
    edges: list[str]        # aproches que sirve la fase (cruces + halting se suman)
    lanes: float            # carriles generales que descargan en la fase

    @property
    def saturation(self) -> float:
        return self.lanes * SAT_PER_LANE


@dataclass
class NodeCfg:
    intersection_id: str    # node_id en graph_nodes (resuelve en el motor)
    tls_id: str             # id del TLS en SUMO
    phases: list[PhaseCfg]  # orden = [LARCO, TRANSV]
    seed_green: dict        # verde inicial por fase para el bootstrap


NODES: list[NodeCfg] = [
    NodeCfg(
        intersection_id="larco_benavides",
        tls_id="cluster_108178119_263630444_2673400749_3245705958_#6more",
        phases=[
            PhaseCfg("LARCO", ["129466113#0"], lanes=2),               # Larco S→N (2 grales; lane0 bus)
            PhaseCfg("TRANSV", ["344159559#2", "406007422#0"], lanes=4),  # Benavides E(2 gral) + O(2)
        ],
        seed_green={"LARCO": 41.0, "TRANSV": 41.0},
    ),
    NodeCfg(
        intersection_id="larco_schell",
        tls_id="133925753",
        phases=[
            PhaseCfg("LARCO", ["279893875#2"], lanes=2),    # Larco S→N (2 grales; lane0 bus)
            PhaseCfg("TRANSV", ["430180649"], lanes=3),     # Schell E (3 carriles, todos grales)
        ],
        seed_green={"LARCO": 42.0, "TRANSV": 42.0},
    ),
]


@dataclass
class NodeSensor:
    """Acumula cruces (flujo) y halting (cola) por aproche durante un ciclo.

    Flujo = vehículos que SALIERON del aproche entre steps (tasa de paso, Catch C),
    excluyendo teleports. Cola = PROMEDIO de getLastStepHaltingNumber sobre el ciclo
    (de-aliasada respecto de la fase; decisión B).
    """
    cfg: NodeCfg
    _prev_ids: dict = field(default_factory=dict)
    _crossed: dict = field(default_factory=dict)
    _halt_samples: dict = field(default_factory=dict)
    _t0: float = 0.0

    def __post_init__(self) -> None:
        for ph in self.cfg.phases:
            for e in ph.edges:
                self._prev_ids[e] = set()
                self._crossed[e] = 0
                self._halt_samples[e] = []

    def observe(self, teleporting: set) -> None:
        for e in self._crossed:
            current = set(traci.edge.getLastStepVehicleIDs(e))
            departed = (self._prev_ids[e] - current) - teleporting
            self._crossed[e] += len(departed)
            self._prev_ids[e] = current
            self._halt_samples[e].append(int(traci.edge.getLastStepHaltingNumber(e)))

    def commit_cycle(self, sim_time: float) -> list[dict]:
        """Cierra el ciclo: devuelve la lista de PhaseFlow para el motor."""
        window = max(1.0, sim_time - self._t0)
        phase_flows = []
        for ph in self.cfg.phases:
            crossed = sum(self._crossed[e] for e in ph.edges)
            # cola = suma de halting medio por aproche (cola que alimenta la fase)
            per_edge_mean = [
                (sum(self._halt_samples[e]) / len(self._halt_samples[e]))
                if self._halt_samples[e] else 0.0
                for e in ph.edges
            ]
            queue = int(round(sum(per_edge_mean)))
            flow_vph = crossed * (3600.0 / window)
            phase_flows.append({
                "phase_id": ph.phase_id,
                "flow": flow_vph,
                "saturation_flow": ph.saturation,
                "queue": queue,
                "has_pedestrian": False,
            })
        # reset acumuladores de ciclo (prev_ids NO se resetea: continuidad entre ciclos)
        for e in self._crossed:
            self._crossed[e] = 0
            self._halt_samples[e].clear()
        self._t0 = sim_time
        return phase_flows


def _is_cycle_edge(tls_id: str, sim_time: float) -> bool:
    if traci.trafficlight.getPhase(tls_id) != LAST_SUBPHASE_IDX:
        return False
    nxt = traci.trafficlight.getNextSwitch(tls_id)
    return sim_time <= nxt <= sim_time + 1.001


def _bootstrap_program(node: NodeCfg, linkstates: dict) -> None:
    """Instala el programa adaptativo de 6 sub-fases (verde seed) y lo activa.

    Necesario: los programas del net.xml tienen 4 sub-fases (sin all-red) → getPhase
    nunca llega a idx 5. Con seed timings (verde = split fijo, yellow=3, all_red=2).
    """
    timings = [
        {"phase_id": ph.phase_id,
         "green": node.seed_green[ph.phase_id], "yellow": 3.0, "all_red": 2.0}
        for ph in node.phases
    ]
    sumo_phases = tllogic_applier.expand_timings_to_sumo_phases(timings, linkstates)
    logic = traci.trafficlight.Logic(programID="adaptive", type=0, currentPhaseIndex=0,
                                     phases=sumo_phases)
    traci.trafficlight.setProgramLogic(node.tls_id, logic)
    traci.trafficlight.setProgram(node.tls_id, "adaptive")
    traci.trafficlight.setPhase(node.tls_id, 0)


def run_adaptive(seed: int, end_s: int, out_dir: Path,
                 engine_recommend_fn: Optional[Callable] = None) -> dict:
    if engine_recommend_fn is None:
        engine_recommend_fn = recommend
    out_dir.mkdir(parents=True, exist_ok=True)
    det_local = out_dir / DET.name
    det_local.write_bytes(DET.read_bytes())

    sumo_bin = Path(os.environ["SUMO_HOME"]) / "bin" / "sumo"
    sumo_cmd = [
        str(sumo_bin), "-c", str(CFG),
        "--additional-files", str(det_local),
        "--seed", str(seed), "--end", str(end_s),
        "--summary-output", str(out_dir / "summary.parquet"),
        "--tripinfo-output", str(out_dir / "tripinfo.parquet"),
        "--no-step-log", "--log", str(out_dir / "sumo.log"),
    ]

    all_linkstates = json.loads(LINKSTATES_JSON.read_text())
    sensors = {n.intersection_id: NodeSensor(n) for n in NODES}
    appliers = {
        n.intersection_id: tllogic_applier.TllogicApplier(
            n.tls_id, all_linkstates[n.intersection_id], program_id="adaptive")
        for n in NODES
    }
    stats = {n.intersection_id: {"calls": 0, "modes": [], "applications": 0,
                                 "allred_steps": 0, "errors": []} for n in NODES}

    traci.start(sumo_cmd, label="corredor_adaptive")
    try:
        # Bootstrap: instalar el programa de 6 sub-fases en cada nodo controlado.
        for n in NODES:
            _bootstrap_program(n, all_linkstates[n.intersection_id])

        step = 0
        while step < end_s:
            traci.simulationStep()
            step += 1
            sim_time = traci.simulation.getTime()
            teleporting = set(traci.simulation.getStartingTeleportIDList())

            for n in NODES:
                iid = n.intersection_id
                sensors[iid].observe(teleporting)
                # contabiliza tiempo perdido en all-red (para el desglose del RD%)
                if traci.trafficlight.getPhase(n.tls_id) in ALLRED_PHASE_IDX:
                    stats[iid]["allred_steps"] += 1

                if _is_cycle_edge(n.tls_id, sim_time):
                    phase_flows = sensors[iid].commit_cycle(sim_time)
                    try:
                        rec = engine_recommend_fn(
                            intersection_id=iid,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            phase_flows=phase_flows,
                            lost_time=10.0,
                        )
                        stats[iid]["calls"] += 1
                        stats[iid]["modes"].append(rec.get("mode", "unknown"))
                        appliers[iid].update_pending(rec)
                    except EngineUnavailable as exc:
                        stats[iid]["errors"].append(f"step={step}: {exc}")
                appliers[iid].maybe_apply(sim_time)
    finally:
        for iid in stats:
            stats[iid]["applications"] = appliers[iid].applied_count
        traci.close()

    (out_dir / "adaptive_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def _print_stats(stats: dict, end_s: int) -> None:
    print(f"\nADAPTATIVO corredor Larco — end={end_s}s")
    for iid, s in stats.items():
        modes = s["modes"]
        mp = sum(1 for m in modes if m == "max_pressure")
        web = sum(1 for m in modes if m == "webster")
        print(f"  [{iid}]")
        print(f"    engine calls / applications : {s['calls']} / {s['applications']}")
        print(f"    modos: max_pressure={mp}  webster={web}  (otros={len(modes)-mp-web})")
        print(f"    tiempo perdido all-red      : {s['allred_steps']} s")
        if s["errors"]:
            print(f"    errores motor ({len(s['errors'])}): {s['errors'][:3]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--end", type=int, default=1800)
    ap.add_argument("--out", type=Path,
                    default=SIM_ROOT / "data" / "corredor_larco" / "peak_s_n_adaptive_seed42_end1800")
    args = ap.parse_args()

    stats = run_adaptive(seed=args.seed, end_s=args.end, out_dir=args.out)
    _print_stats(stats, args.end)

    # Catch C: bajo peak ambos nodos deberían rutear a max_pressure al menos una vez.
    bad = [iid for iid, s in stats.items()
           if "max_pressure" not in set(s["modes"]) and not s["errors"]]
    if bad:
        print(f"\n⚠️  Catch C: nodos sin max_pressure bajo peak: {bad} "
              "(flow probablemente subestimado).", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
