"""Lee estado por dirección desde TraCI live (ventana de 30 s).

CRÍTICO (Catch C de Cesar): ``flow`` debe ser **tasa de paso**
(veh/h cruzando), NO ocupación instantánea (``getLastStepVehicleNumber``).
Underestimación rompe el routing del engine (debería rutear a
max_pressure bajo pico).

Estrategia: tracker mantiene IDs de vehículos vistos en cada edge de
aproche. Cada vez que ``commit_window()`` se llama (cada 30 s sim),
los IDs nuevos respecto al snapshot anterior se cuentan como
arrivals y se convierte a veh/h.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import traci


# Edges de aproche por dirección.
EDGE_BY_DIR = {"N": "N_in", "S": "S_in", "E": "E_in", "W": "W_in"}
QUEUE_VEH_LENGTH_M = 7.5  # Vehículo promedio + gap (HCM estándar)


@dataclass
class DirectionState:
    direction: str
    flow_vph: float           # Tasa de paso veh/h (Catch C)
    queue_vehicles: int       # Cola en vehículos (halting number agregado)
    queue_length_m: float     # Cola en metros (estimada)
    mean_speed_mps: float     # Velocidad media instantánea


@dataclass
class StateTracker:
    """Tracker que cuenta arrivals únicos por dirección en una ventana de N segundos."""

    window_s: float = 30.0
    _seen_per_dir: dict[str, set[str]] = field(default_factory=dict)
    _last_commit_time: float = 0.0

    def __post_init__(self) -> None:
        for d in EDGE_BY_DIR:
            self._seen_per_dir[d] = set()

    def observe(self, sim_time: float) -> None:
        """Agrega IDs de vehículos actualmente en cada edge de aproche al set de vistos."""
        for d, edge_id in EDGE_BY_DIR.items():
            current = set(traci.edge.getLastStepVehicleIDs(edge_id))
            self._seen_per_dir[d] |= current

    def commit_window(self, sim_time: float) -> dict[str, DirectionState]:
        """Cierra la ventana, devuelve DirectionState por dirección y resetea."""
        window_actual = max(1.0, sim_time - self._last_commit_time)
        states: dict[str, DirectionState] = {}
        for d, edge_id in EDGE_BY_DIR.items():
            n_arrivals = len(self._seen_per_dir[d])
            flow_vph = n_arrivals * (3600.0 / window_actual)
            queue_n = int(traci.edge.getLastStepHaltingNumber(edge_id))
            queue_m = queue_n * QUEUE_VEH_LENGTH_M
            mean_speed = float(traci.edge.getLastStepMeanSpeed(edge_id))
            states[d] = DirectionState(
                direction=d,
                flow_vph=flow_vph,
                queue_vehicles=queue_n,
                queue_length_m=queue_m,
                mean_speed_mps=mean_speed,
            )
            self._seen_per_dir[d].clear()
        self._last_commit_time = sim_time
        return states
