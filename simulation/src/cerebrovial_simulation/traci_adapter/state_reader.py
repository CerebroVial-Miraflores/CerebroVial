"""Lee estado por dirección desde TraCI live (ventana de 30 s).

CRÍTICO (Catch C de Cesar): ``flow`` debe ser **tasa de paso**
(veh/h cruzando), NO ocupación instantánea (``getLastStepVehicleNumber``).
Underestimación rompe el routing del engine (debería rutear a
max_pressure bajo pico).

Estrategia (medición representativa, no instantánea):
- **Cola** (presión Max Pressure): se muestrea ``getLastStepHaltingNumber``
  por edge en CADA step de la ventana; ``commit_window()`` devuelve el
  PROMEDIO. Un único snapshot aliasea con la fase del semáforo (cola sube
  en rojo, baja en verde) → presión sesgada → split invertido → gridlock.
  El promedio integra la ventana y elimina esa inversión.
- **Flujo** (tasa de paso): se cuentan vehículos que SALIERON del aproche
  entre steps (presentes en el step previo, ausentes en el actual =
  cruzaron la línea de pare), excluyendo teleports. NO la unión de IDs
  presentes, que cuenta la cola parada como flujo y lo infla.
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
    queue_vehicles: int       # Cola en vehículos (promedio de halting en la ventana)
    queue_length_m: float     # Cola en metros (estimada)
    mean_speed_mps: float     # Velocidad media instantánea


@dataclass
class StateTracker:
    """Muestrea cola y cruces por dirección en una ventana de N segundos.

    - Cola (presión Max Pressure): se muestrea getLastStepHaltingNumber por edge
      en CADA step; commit_window() devuelve el PROMEDIO de la ventana — presión
      sostenida, de-aliasada respecto de la fase del semáforo. NO un snapshot.
    - Flujo (tasa de paso): se cuentan vehículos que SALIERON del aproche entre
      steps (presentes en el step previo, ausentes en el actual = cruzaron la
      línea de pare), excluyendo teleports. NO presencia/unión de IDs.
    """

    window_s: float = 30.0
    _prev_ids: dict[str, set[str]] = field(default_factory=dict)
    _crossed: dict[str, int] = field(default_factory=dict)
    _halt_samples: dict[str, list[int]] = field(default_factory=dict)
    _last_commit_time: float = 0.0

    def __post_init__(self) -> None:
        for d in EDGE_BY_DIR:
            self._prev_ids[d] = set()
            self._crossed[d] = 0
            self._halt_samples[d] = []

    def observe(self, sim_time: float) -> None:
        """Por step: cuenta cruces (salidas del aproche) y muestrea la cola."""
        teleporting = set(traci.simulation.getStartingTeleportIDList())
        for d, edge_id in EDGE_BY_DIR.items():
            current = set(traci.edge.getLastStepVehicleIDs(edge_id))
            # Presentes antes y ausentes ahora = cruzaron; se excluyen teleports.
            departed = (self._prev_ids[d] - current) - teleporting
            self._crossed[d] += len(departed)
            self._prev_ids[d] = current
            self._halt_samples[d].append(
                int(traci.edge.getLastStepHaltingNumber(edge_id))
            )

    def commit_window(self, sim_time: float) -> dict[str, DirectionState]:
        """Cierra la ventana: flow = cruces/h, queue = promedio de la ventana."""
        window_actual = max(1.0, sim_time - self._last_commit_time)
        states: dict[str, DirectionState] = {}
        for d, edge_id in EDGE_BY_DIR.items():
            flow_vph = self._crossed[d] * (3600.0 / window_actual)
            samples = self._halt_samples[d]
            queue_n = int(round(sum(samples) / len(samples))) if samples else 0
            queue_m = queue_n * QUEUE_VEH_LENGTH_M
            mean_speed = float(traci.edge.getLastStepMeanSpeed(edge_id))
            states[d] = DirectionState(
                direction=d,
                flow_vph=flow_vph,
                queue_vehicles=queue_n,
                queue_length_m=queue_m,
                mean_speed_mps=mean_speed,
            )
            # Reset de acumuladores de ventana. _prev_ids NO se resetea: un auto
            # presente en el último step de esta ventana y ausente en el primero
            # de la próxima cruzó en la próxima ventana.
            self._crossed[d] = 0
            self._halt_samples[d].clear()
        self._last_commit_time = sim_time
        return states
