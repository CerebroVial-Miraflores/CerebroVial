"""Interfaz de feed de congestión por arista (CT-12.3).

Este es el punto de desacople de la FUENTE de congestión. Define el contrato que
todo emisor de congestión por arista cumple — hoy el replay SUMO
(``infrastructure.sumo_replay_adapter.SumoReplayAdapter``), mañana SUMO en vivo
vía TraCI o un ingestor de Waze real. Sustituir la fuente NO toca a los
consumidores (repositorio, endpoint, vista) mientras respete esta interfaz.

La derivación del nivel 0-5 (escala Waze, D-009) vive DETRÁS de la interfaz, en el
adaptador concreto, reusando ``ratio_to_jam_level`` / ``sumo_to_jam_level`` —
nunca se reescribe el mapeo en este nivel.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Protocol, runtime_checkable


@dataclass(frozen=True)
class EdgeCongestion:
    """Estado de congestión de UNA arista en UN instante.

    Es el dato que cruza la interfaz: por ``edge_id``, un ``congestion_level``
    0-5 (escala Waze, D-009) con su marca de tiempo. Atravesar la interfaz como
    dataclass plano evita filtrar ORM/Parquet hacia los consumidores.
    """
    edge_id: str
    congestion_level: int          # 0-5 (0 = flujo libre; CT-12.3 / D-009)
    snapshot_timestamp: datetime


@runtime_checkable
class CongestionFeed(Protocol):
    """Contrato de una fuente de congestión por arista (CT-12.3).

    Un feed expone, para un instante (``timestep`` del día reproducido), el estado
    de congestión de todas las aristas conocidas. La implementación decide de dónde
    sale (replay de dataset, TraCI en vivo, API de Waze) y cómo se deriva el nivel.
    """

    def timesteps(self) -> list[int]:
        """Los instantes (segundos simulados del día) que el feed puede emitir, ascendente."""
        ...

    def levels_at(self, timestep: int) -> list[EdgeCongestion]:
        """Estado de congestión de cada arista en ``timestep`` (uno por arista)."""
        ...

    def snapshots(self) -> Iterable[EdgeCongestion]:
        """Itera TODOS los (arista × instante) del día — usado por la pre-siembra batch."""
        ...
