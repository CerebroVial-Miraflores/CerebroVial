"""Agrega 4 direcciones → 2 PhaseFlow (Option A — NS+EW).

Mapping aprobado:
- PhaseFlow("NS"): flow = flow_N + flow_S; saturation = 6 lanes × 1800
  = 10800 veh/h (suma sat de ambas direcciones del eje); queue = max
  entre N y S; has_pedestrian = True.
- PhaseFlow("EW"): flow = flow_E + flow_W; saturation = 4 × 1800 =
  7200; idem.

`flow` ya es tasa de paso (veh/h) per state_reader. Suma directa.
`queue` toma el `max` (no la suma) porque ambas direcciones del eje
comparten la fase verde — la "cola crítica" del eje es la mayor.
"""
from __future__ import annotations

from .state_reader import DirectionState


SATURATION_VPH = {"NS": 10800.0, "EW": 7200.0}


def aggregate_to_phase_flows(
    states: dict[str, DirectionState],
    has_pedestrian: bool = True,
) -> list[dict]:
    """Retorna la lista [PhaseFlow_NS, PhaseFlow_EW] como dicts JSON-serializables
    listos para el body del POST /control/recommend.
    """
    flow_ns = states["N"].flow_vph + states["S"].flow_vph
    flow_ew = states["E"].flow_vph + states["W"].flow_vph
    queue_ns = max(states["N"].queue_vehicles, states["S"].queue_vehicles)
    queue_ew = max(states["E"].queue_vehicles, states["W"].queue_vehicles)
    return [
        {
            "phase_id": "NS",
            "flow": flow_ns,
            "saturation_flow": SATURATION_VPH["NS"],
            "queue": int(queue_ns),
            "has_pedestrian": has_pedestrian,
        },
        {
            "phase_id": "EW",
            "flow": flow_ew,
            "saturation_flow": SATURATION_VPH["EW"],
            "queue": int(queue_ew),
            "has_pedestrian": has_pedestrian,
        },
    ]
