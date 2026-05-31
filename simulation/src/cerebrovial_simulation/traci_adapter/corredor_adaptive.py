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
    # Max Pressure de red (Etapa 2): links downstream a los que descarga la fase.
    # El sensor mide su halting medio y, con --downstream, lo manda como
    # downstream=[{queue, turn_ratio}] en el payload. Vacío ⇒ sin término (sink
    # libre, p.ej. Schell→DC). Solo Benavides-LARCO lo usa en este corredor.
    downstream_edges: list[str] = field(default_factory=list)
    turn_ratio: float = 1.0  # τ: fracción del flujo de la fase que va al link downstream

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
            # LARCO descarga al link interno Benavides→Schell (279893875#1): el tapón
            # demostrado (~99% en IE05). τ=1.0 (≈89% del inflow de #1 es este pasante;
            # ver auditoría Etapa 2). x_local 129466113#0 es de un solo sentido → resta limpia.
            PhaseCfg("LARCO", ["129466113#0"], lanes=2,                 # Larco S→N (2 grales; lane0 bus)
                     downstream_edges=["279893875#1"], turn_ratio=1.0),
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
    _down_halt_samples: dict = field(default_factory=dict)  # halting de links downstream
    _t0: float = 0.0

    def __post_init__(self) -> None:
        for ph in self.cfg.phases:
            for e in ph.edges:
                self._prev_ids[e] = set()
                self._crossed[e] = 0
                self._halt_samples[e] = []
            for e in ph.downstream_edges:
                self._down_halt_samples.setdefault(e, [])

    def observe(self, teleporting: set) -> None:
        for e in self._crossed:
            current = set(traci.edge.getLastStepVehicleIDs(e))
            departed = (self._prev_ids[e] - current) - teleporting
            self._crossed[e] += len(departed)
            self._prev_ids[e] = current
            self._halt_samples[e].append(int(traci.edge.getLastStepHaltingNumber(e)))
        # Halting de los links downstream (cola del vecino, p.ej. Benavides→Schell).
        # Muestreado sobre el MISMO ciclo que x_local (sin lag stale).
        for e in self._down_halt_samples:
            self._down_halt_samples[e].append(int(traci.edge.getLastStepHaltingNumber(e)))

    def commit_cycle(self, sim_time: float, downstream: bool = False,
                     turn_ratio: Optional[float] = None) -> list[dict]:
        """Cierra el ciclo: devuelve la lista de PhaseFlow para el motor.

        Con ``downstream=True``, las fases con ``downstream_edges`` adjuntan
        ``downstream=[{queue, turn_ratio}]`` (Max Pressure de red). Con
        ``downstream=False`` (default) el payload NO lleva la clave → idéntico
        byte-a-byte al per-node (retrocompat / brazo de control del experimento).

        ``turn_ratio`` (override) reemplaza el τ de cada PhaseCfg (barrido del eje
        de acoplamiento). None ⇒ usa el τ configurado en NODES.
        """
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
            phase_flow = {
                "phase_id": ph.phase_id,
                "flow": flow_vph,
                "saturation_flow": ph.saturation,
                "queue": queue,
                "has_pedestrian": False,
            }
            if downstream and ph.downstream_edges:
                down_mean = sum(
                    (sum(self._down_halt_samples[e]) / len(self._down_halt_samples[e]))
                    if self._down_halt_samples[e] else 0.0
                    for e in ph.downstream_edges
                )
                tau = ph.turn_ratio if turn_ratio is None else turn_ratio
                phase_flow["downstream"] = [
                    {"queue": int(round(down_mean)), "turn_ratio": tau}
                ]
            phase_flows.append(phase_flow)
        # reset acumuladores de ciclo (prev_ids NO se resetea: continuidad entre ciclos)
        for e in self._crossed:
            self._crossed[e] = 0
            self._halt_samples[e].clear()
        for e in self._down_halt_samples:
            self._down_halt_samples[e].clear()
        self._t0 = sim_time
        return phase_flows


def _is_cycle_edge(tls_id: str, sim_time: float) -> bool:
    if traci.trafficlight.getPhase(tls_id) != LAST_SUBPHASE_IDX:
        return False
    nxt = traci.trafficlight.getNextSwitch(tls_id)
    return sim_time <= nxt <= sim_time + 1.001


# --- Onda verde (ciclo común fijo + offset). Cero core: normaliza la respuesta
#     del motor en el adaptador; el motor sigue decidiendo el ratio por presión. ---
MIN_GREEN = 7.0     # MTC (motor): no se viola, solo se estrecha (ver normalize_cycle)
MAX_GREEN = 60.0


def normalize_cycle(timings: list[dict], cycle: float = 90.0) -> list[dict]:
    """Reescala los verdes del motor para que el ciclo total sea exactamente `cycle`.

    Solo para 2 fases (LARCO, TRANSV). El motor decide el RATIO por presión; acá se
    fija el ciclo. presupuesto_verde = cycle - Σ(yellow+all_red). Con 2 fases sumando
    el presupuesto, la cota g≤MAX_GREEN fuerza g≥(presupuesto-MAX_GREEN) y viceversa →
    el rango factible por fase es [presupuesto-MAX_GREEN, MAX_GREEN] ⊂ [MIN_GREEN,
    MAX_GREEN] (para cycle=90 → [20,60]). No viola MTC; solo estrecha el piso.

    El `round` único (no dos por separado) mantiene la suma exacta → ciclo invariante,
    lo que preserva el offset entre nodos sin drift.
    """
    if len(timings) != 2:
        raise ValueError(f"normalize_cycle asume 2 fases, recibió {len(timings)}")
    lost = sum(t["yellow"] + t["all_red"] for t in timings)
    budget = cycle - lost
    lo = max(MIN_GREEN, budget - MAX_GREEN)
    hi = min(MAX_GREEN, budget - MIN_GREEN)
    g = [t["green"] for t in timings]
    s = sum(g)
    g0 = g[0] * budget / s if s > 0 else budget / 2.0
    g0 = float(round(min(max(g0, lo), hi)))
    out = [dict(t) for t in timings]
    out[0]["green"] = g0
    out[1]["green"] = budget - g0
    return out


def _phase_at_pos(durs: list[float], pos: float) -> tuple[int, float]:
    """Dada la lista de duraciones de sub-fase y una posición en el ciclo, devuelve
    (idx de la sub-fase que contiene `pos`, tiempo restante en esa sub-fase).

    Para materializar el offset: arrancar Schell en (idx, rem) hace que su fase 0
    (LARCO verde) empiece exactamente en t = offset.
    """
    acc = 0.0
    for idx, d in enumerate(durs):
        if pos < acc + d:
            return idx, (acc + d) - pos
        acc += d
    return 0, durs[0]   # pos == ciclo → arranque limpio en fase 0


def _bootstrap_program(node: NodeCfg, linkstates: dict,
                       cycle: Optional[float] = None, offset: float = 0.0) -> None:
    """Instala el programa adaptativo de 6 sub-fases (verde seed) y lo activa.

    Necesario: los programas del net.xml tienen 4 sub-fases (sin all-red) → getPhase
    nunca llega a idx 5. Con seed timings (verde = split fijo, yellow=3, all_red=2).

    Onda verde: si `cycle` está fijo, los verdes seed se normalizan a ese ciclo. Si
    `offset > 0`, el nodo arranca "metido" en su ciclo (posición = (cycle-offset)%cycle)
    para que su fase 0 empiece en t=offset (SUMO 1.26 no aplica Logic.offset en runtime).
    """
    timings = [
        {"phase_id": ph.phase_id,
         "green": node.seed_green[ph.phase_id], "yellow": 3.0, "all_red": 2.0}
        for ph in node.phases
    ]
    if cycle is not None:
        timings = normalize_cycle(timings, cycle)
    sumo_phases = tllogic_applier.expand_timings_to_sumo_phases(timings, linkstates)
    logic = traci.trafficlight.Logic(programID="adaptive", type=0, currentPhaseIndex=0,
                                     phases=sumo_phases)
    traci.trafficlight.setProgramLogic(node.tls_id, logic)
    traci.trafficlight.setProgram(node.tls_id, "adaptive")

    if cycle is not None and offset:
        durs = [p.duration for p in sumo_phases]
        pos = (cycle - offset) % cycle
        idx, rem = _phase_at_pos(durs, pos)
        traci.trafficlight.setPhase(node.tls_id, idx)
        traci.trafficlight.setPhaseDuration(node.tls_id, rem)
    else:
        traci.trafficlight.setPhase(node.tls_id, 0)


def run_adaptive(seed: int, end_s: int, out_dir: Path,
                 engine_recommend_fn: Optional[Callable] = None,
                 cycle: Optional[float] = None, offset: float = 0.0,
                 downstream: bool = False, turn_ratio: Optional[float] = None) -> dict:
    """Lazo adaptativo. Onda verde opcional: `cycle` fija el ciclo común (None = ciclo
    variable, comportamiento histórico); `offset` desfasa el arranque de Schell respecto
    a Benavides (solo aplica si `cycle` está fijo). `downstream` activa el término de
    Max Pressure de red (manda la cola del link interno Benavides→Schell). Defaults =
    comportamiento per-node histórico (downstream OFF ⇒ payload byte-idéntico)."""
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
        # El offset solo aplica a Schell (Benavides es la referencia, arranca en t=0).
        for n in NODES:
            node_offset = offset if n.intersection_id == "larco_schell" else 0.0
            _bootstrap_program(n, all_linkstates[n.intersection_id],
                               cycle=cycle, offset=node_offset)

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
                    phase_flows = sensors[iid].commit_cycle(sim_time, downstream=downstream,
                                                            turn_ratio=turn_ratio)
                    try:
                        rec = engine_recommend_fn(
                            intersection_id=iid,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            phase_flows=phase_flows,
                            lost_time=10.0,
                        )
                        stats[iid]["calls"] += 1
                        stats[iid]["modes"].append(rec.get("mode", "unknown"))
                        if cycle is not None:
                            rec = {**rec,
                                   "phase_timings": normalize_cycle(rec["phase_timings"], cycle)}
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
    ap.add_argument("--cycle", type=float, default=None,
                    help="ciclo común fijo (onda verde); None = ciclo variable (default histórico)")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="desfase del arranque de Schell respecto a Benavides (solo con --cycle)")
    ap.add_argument("--downstream", action="store_true",
                    help="activa el término Max Pressure de red (cola del link interno Benavides→Schell)")
    args = ap.parse_args()

    stats = run_adaptive(seed=args.seed, end_s=args.end, out_dir=args.out,
                         cycle=args.cycle, offset=args.offset, downstream=args.downstream)
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
