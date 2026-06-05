"""Lazo adaptativo PER-NODE PURO sobre la red completa de Miraflores (F3b).

Generaliza ``corredor_adaptive.run_adaptive`` (que itera 2 nodos hardcodeados de Larco) a
los N TLS de 2 fases que produce el generador ``node_config_gen`` (F3a). NO toca
``corredor_adaptive.py`` (que sigue sirviendo a Larco / IE05): REUSA sus piezas por import
—``NodeSensor``, ``_is_cycle_edge``, ``_bootstrap_program``, ``ALLRED_PHASE_IDX``,
``NodeCfg``/``PhaseCfg``— y reescribe solo el lazo, sin lo atado al corredor.

Diferencias con el lazo de Larco (decisiones F3b, no reabrir):
  - **per-node PURO:** sin onda verde (offset hardcodeado a ``larco_schell`` eliminado) ni
    ciclo común fijo. ``cycle=None`` siempre.
  - **sin downstream / MP-red** (scope opcional diferido): ``commit_cycle(downstream=False)``.
  - **config de los nodos del generador**, no del ``NODES`` hardcodeado ni de
    ``linkstates_corredor.json``: se deriva del net en memoria.
  - **mapeo al motor por ``intersection_id = "sumo_" + tls_id``** (Decisión 2): el motor
    solo verifica existencia + FK de ``motor_decisions``, no lee atributos del nodo
    (auditado). DEUDA CONSCIENTE: reusar el namespace ``sumo_`` (geometría de congestión)
    para el control cruza namespaces; aceptable para un experimento de validación
    (``motor_decisions`` es subproducto; el KPI es ``net_timeLoss``). Queda anotado.
  - **SUMO por ``-n NET -r ROU``** (mismos parámetros que F1/F2: tt=300, collision warn,
    seed, 0..86400) en vez del ``.sumocfg`` del corredor, para que el brazo adaptativo sea
    apples-to-apples con el fijo (única diferencia: el control).

NO generaliza a N fases (eso es F6): los supuestos de 2 fases del núcleo reusado
(``ALLRED_PHASE_IDX=(2,5)``, bootstrap de 6 sub-fases) valen porque los N son todos de 2 fases.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import traci

from . import tllogic_applier
from .corredor_adaptive import (
    ALLRED_PHASE_IDX,
    NodeCfg,
    NodeSensor,
    PhaseCfg,
    _bootstrap_program,
    _is_cycle_edge,
)
from .engine_client import EngineUnavailable, recommend
from .node_config_gen import derive_node_specs

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent.parent
DEFAULT_NET = SIM_ROOT / "conf" / "network" / "miraflores.net.xml"
DEFAULT_ROU = (SIM_ROOT / "conf" / "miraflores_red_completa" / "routes"
               / "miraflores_seed051_laborable.rou.xml")


def _specs_to_nodecfgs(net_path: Path, tls_ids: Optional[list[str]]):
    """Convierte los NodeSpec del generador (F3a) a NodeCfg + linkstates por intersection_id.

    intersection_id = "sumo_" + tls_id (Decisión 2). downstream/turn_ratio quedan en su
    default (MP-red diferido).
    """
    specs = derive_node_specs(net_path, tls_ids)
    nodes: list[NodeCfg] = []
    linkstates: dict[str, dict] = {}
    for s in specs:
        iid = f"sumo_{s.tls_id}"
        nodes.append(NodeCfg(
            intersection_id=iid,
            tls_id=s.tls_id,
            phases=[PhaseCfg(phase_id=p.phase_id, edges=list(p.edges), lanes=p.lanes)
                    for p in s.phases],
            seed_green=dict(s.seed_green),
        ))
        linkstates[iid] = dict(s.linkstates)
    return nodes, linkstates


def run_adaptive_red(seed: int, end_s: int, out_dir: Path, *,
                     net: Path = DEFAULT_NET, rou: Path = DEFAULT_ROU,
                     tls_ids: Optional[list[str]] = None,
                     engine_recommend_fn: Optional[Callable] = None) -> dict:
    """Lazo adaptativo per-node puro sobre los TLS de 2 fases (todos si tls_ids=None).

    Mismos parámetros SUMO que el fijo de F1/F2 (tt=300, collision warn, seed, 0..end_s);
    única diferencia con el fijo = el control adaptativo por TraCI. Escribe
    summary.parquet + tripinfo.parquet en out_dir (métrica vía collect_red_metrics.py).
    """
    if engine_recommend_fn is None:
        engine_recommend_fn = recommend
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes, all_linkstates = _specs_to_nodecfgs(net, tls_ids)

    sumo_bin = Path(os.environ["SUMO_HOME"]) / "bin" / "sumo"
    sumo_cmd = [
        str(sumo_bin), "-n", str(net), "-r", str(rou),
        "--begin", "0", "--end", str(end_s), "--step-length", "1",
        "--time-to-teleport", "300", "--collision.action", "warn", "--seed", str(seed),
        "--summary-output", str(out_dir / "summary.parquet"),
        "--tripinfo-output", str(out_dir / "tripinfo.parquet"),
        "--no-step-log", "--log", str(out_dir / "sumo.log"),
    ]

    sensors = {n.intersection_id: NodeSensor(n) for n in nodes}
    appliers = {
        n.intersection_id: tllogic_applier.TllogicApplier(
            n.tls_id, all_linkstates[n.intersection_id], program_id="adaptive")
        for n in nodes
    }
    stats = {n.intersection_id: {"calls": 0, "modes": [], "applications": 0,
                                 "allred_steps": 0, "errors": []} for n in nodes}

    traci.start(sumo_cmd, label="red_adaptive")
    try:
        # Bootstrap: instalar el programa adaptativo de 6 sub-fases en cada nodo (per-node
        # puro: sin offset). cycle=None ⇒ ciclo variable, el motor decide el ratio.
        for n in nodes:
            _bootstrap_program(n, all_linkstates[n.intersection_id], cycle=None, offset=0.0)

        step = 0
        while step < end_s:
            traci.simulationStep()
            step += 1
            sim_time = traci.simulation.getTime()
            teleporting = set(traci.simulation.getStartingTeleportIDList())

            for n in nodes:
                iid = n.intersection_id
                sensors[iid].observe(teleporting)
                if traci.trafficlight.getPhase(n.tls_id) in ALLRED_PHASE_IDX:
                    stats[iid]["allred_steps"] += 1

                if _is_cycle_edge(n.tls_id, sim_time):
                    phase_flows = sensors[iid].commit_cycle(sim_time, downstream=False)
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

    (out_dir / "adaptive_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def _summarize(stats: dict) -> None:
    n = len(stats)
    total_calls = sum(s["calls"] for s in stats.values())
    total_apps = sum(s["applications"] for s in stats.values())
    modes = [m for s in stats.values() for m in s["modes"]]
    mp = sum(1 for m in modes if m == "max_pressure")
    web = sum(1 for m in modes if m == "webster")
    err_nodes = [iid for iid, s in stats.items() if s["errors"]]
    print(f"\nADAPTATIVO red — {n} nodos controlados")
    print(f"  engine calls / applications : {total_calls} / {total_apps}")
    print(f"  modos: max_pressure={mp}  webster={web}  (otros={len(modes)-mp-web})")
    print(f"  nodos con errores de motor  : {len(err_nodes)} {err_nodes[:5]}")


_LARCO_TLS = ["133925753", "cluster_108178119_263630444_2673400749_3245705958_#6more"]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Lazo adaptativo per-node puro sobre la red (F3b).")
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--end", type=int, default=86400)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--net", type=Path, default=DEFAULT_NET)
    ap.add_argument("--rou", type=Path, default=DEFAULT_ROU)
    ap.add_argument("--larco", action="store_true",
                    help="controlar SOLO los 2 TLS de Larco (validación del wiring)")
    args = ap.parse_args(argv)

    tls_ids = _LARCO_TLS if args.larco else None
    stats = run_adaptive_red(seed=args.seed, end_s=args.end, out_dir=args.out,
                             net=args.net, rou=args.rou, tls_ids=tls_ids)
    _summarize(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
