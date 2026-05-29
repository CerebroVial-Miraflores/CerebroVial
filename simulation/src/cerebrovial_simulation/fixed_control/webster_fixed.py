"""Webster fijo offline (Catch B de Cesar).

Calcula el plan Webster con los flujos AM-peak directamente desde la
fórmula transcrita literalmente de
``core_management_api/src/control/application/webster.py:7-12`` —
**SIN llamar al motor HTTP**. Razón (Catch B): AM-peak total cruza
``PEAK_THRESHOLD=1500`` ⇒ el engine rutea a max_pressure, no a Webster.
Llamarlo daría un plan max_pressure congelado y viciaría la comparación
adaptive-engine vs. fixed-Webster.

Persiste el resultado como ``conf/network/webster_fixed.add.xml`` con
las 6 sub-fases SUMO expandidas (programID="webster_fixed").

Fórmula:
    y_i      = flow_i / sat_i
    Y        = Σ y_i
    C_opt    = (1.5 · L + 5) / (1 − Y)        si Y < 0.95, else infeasible
    cycle    = clamp(C_opt, MIN_CYCLE, MAX_CYCLE)
    g_i      = (cycle − L) · (y_i / Y)
    g_i      = clamp(g_i, MIN_GREEN, MAX_GREEN)

Constantes MTC lockeadas (mtc_constraints.py):
    MIN_CYCLE=30, MAX_CYCLE=120, MIN_GREEN=7, MAX_GREEN=60,
    MIN_YELLOW=3, ALL_RED=2.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


# MTC + Webster constants — lockeados al motor en core_management_api.
MIN_CYCLE = 30.0
MAX_CYCLE = 120.0
MIN_GREEN = 7.0
MAX_GREEN = 60.0
MIN_YELLOW = 3.0
ALL_RED = 2.0
Y_MAX = 0.95


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent.parent
PATTERN_PARAMS = SIM_ROOT / "conf" / "scenarios" / "pattern_params.yaml"
NETWORK_PARAMS = SIM_ROOT / "conf" / "network" / "network_params.yaml"
LINKSTATES_JSON = SIM_ROOT / "conf" / "network" / "linkstates.json"
ADD_FILE = SIM_ROOT / "conf" / "network" / "webster_fixed.add.xml"


@dataclass
class WebsterPlan:
    cycle_seconds: float
    green_NS: float
    green_EW: float
    yellow_s: float
    all_red_s: float
    lost_time_s: float
    y_total: float


class WebsterInfeasible(Exception):
    pass


def compute(
    flow_ns_vph: float,
    flow_ew_vph: float,
    sat_ns_vph: float,
    sat_ew_vph: float,
    lost_time_s: float = 10.0,
) -> WebsterPlan:
    """Webster directo + MTC clamp local. NO invoca el motor HTTP."""
    if flow_ns_vph < 0 or flow_ew_vph < 0:
        raise ValueError("flows deben ser ≥ 0")
    if sat_ns_vph <= 0 or sat_ew_vph <= 0:
        raise ValueError("saturations deben ser > 0")

    y_ns = flow_ns_vph / sat_ns_vph
    y_ew = flow_ew_vph / sat_ew_vph
    y_total = y_ns + y_ew

    if y_total >= Y_MAX:
        raise WebsterInfeasible(
            f"Intersección sobre-saturada: Y={y_total:.3f} >= Y_MAX={Y_MAX}"
        )

    c_opt = (1.5 * lost_time_s + 5.0) / (1.0 - y_total)
    cycle = max(MIN_CYCLE, min(MAX_CYCLE, c_opt))
    usable_green = max(0.0, cycle - lost_time_s)

    if y_total == 0.0:
        # Edge case: sin demanda. Reparto burdo.
        g_ns = g_ew = usable_green / 2.0
    else:
        g_ns = usable_green * (y_ns / y_total)
        g_ew = usable_green * (y_ew / y_total)

    # MTC clamp local (idéntico al motor).
    g_ns = max(MIN_GREEN, min(MAX_GREEN, g_ns))
    g_ew = max(MIN_GREEN, min(MAX_GREEN, g_ew))

    # Si tras clamp el ciclo total cambia, recomputarlo coherentemente.
    cycle_total = g_ns + g_ew + 2 * (MIN_YELLOW + ALL_RED)
    if cycle_total > MAX_CYCLE:
        # Scale down proporcionalmente.
        available = MAX_CYCLE - 2 * (MIN_YELLOW + ALL_RED)
        sum_g = g_ns + g_ew
        if sum_g > 0:
            factor = available / sum_g
            g_ns *= factor
            g_ew *= factor
        cycle_total = g_ns + g_ew + 2 * (MIN_YELLOW + ALL_RED)

    return WebsterPlan(
        cycle_seconds=cycle_total,
        green_NS=g_ns,
        green_EW=g_ew,
        yellow_s=MIN_YELLOW,
        all_red_s=ALL_RED,
        lost_time_s=lost_time_s,
        y_total=y_total,
    )


def compute_from_pattern(pattern: str = "am_peak") -> WebsterPlan:
    """Lee pattern_params.yaml + network_params.yaml y retorna el plan Webster."""
    with open(PATTERN_PARAMS) as f:
        patterns = yaml.safe_load(f)
    with open(NETWORK_PARAMS) as f:
        network = yaml.safe_load(f)

    flows = patterns["patterns"][pattern]["flows_vph"]
    flow_ns = flows["N"] + flows["S"]
    flow_ew = flows["E"] + flows["W"]

    ns_lanes = network["intersection"]["ns"]["lanes_per_approach"]
    ew_lanes = network["intersection"]["ew"]["lanes_per_approach"]
    sat_per_lane = network["intersection"]["ns"]["saturation_flow_vph_per_lane"]
    # 2 direcciones por eje (N+S comparten fase NS, E+W comparten fase EW).
    sat_ns = 2 * ns_lanes * sat_per_lane
    sat_ew = 2 * ew_lanes * sat_per_lane

    lost_time = network["tllogic"]["lost_time_s"]

    return compute(
        flow_ns_vph=flow_ns,
        flow_ew_vph=flow_ew,
        sat_ns_vph=sat_ns,
        sat_ew_vph=sat_ew,
        lost_time_s=lost_time,
    )


def write_addfile(
    plan: WebsterPlan,
    linkstates: Optional[dict[str, str]] = None,
    out_file: Path = ADD_FILE,
    program_id: str = "webster_fixed",
) -> Path:
    """Persiste el plan como additional-file con 6 sub-fases SUMO expandidas."""
    if linkstates is None:
        linkstates = json.loads(LINKSTATES_JSON.read_text())

    sub_phases = [
        ("NS_g", plan.green_NS, linkstates["NS_g"]),
        ("NS_y", plan.yellow_s, linkstates["NS_y"]),
        ("NS_r", plan.all_red_s, linkstates["NS_r"]),
        ("EW_g", plan.green_EW, linkstates["EW_g"]),
        ("EW_y", plan.yellow_s, linkstates["EW_y"]),
        ("EW_r", plan.all_red_s, linkstates["EW_r"]),
    ]
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<!-- Webster fijo precomputado offline (Catch B): cycle={plan.cycle_seconds:.1f}s, '
        f'Y={plan.y_total:.3f}, green_NS={plan.green_NS:.1f}, green_EW={plan.green_EW:.1f}. -->',
        '<additional>',
        f'  <tlLogic id="center" type="static" programID="{program_id}" offset="0">',
    ]
    for name, dur, state in sub_phases:
        lines.append(f'    <phase duration="{dur:.2f}" state="{state}"/>  <!-- {name} -->')
    lines.append("  </tlLogic>")
    lines.append("</additional>")
    out_file.write_text("\n".join(lines) + "\n")
    return out_file


def main() -> int:
    plan = compute_from_pattern("am_peak")
    out = write_addfile(plan)
    print(f"Webster fijo (am_peak):")
    print(f"  cycle = {plan.cycle_seconds:.2f}s")
    print(f"  green_NS = {plan.green_NS:.2f}s")
    print(f"  green_EW = {plan.green_EW:.2f}s")
    print(f"  Y = {plan.y_total:.4f}")
    print(f"  add-file: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
