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
from datetime import date, datetime
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


@dataclass(frozen=True)
class EdgeSeries:
    """Serie de niveles de congestión de UNA arista a lo largo de un día (Formato B).

    ``levels[i]`` es el nivel 0-5 (D-009) en el instante ``t0 + i * step_s`` de la
    serie del día (ver ``DayCongestionSeries``). El consumidor (HU-23) indexa en O(1).
    """
    edge_id: str
    levels: list[int]


@dataclass(frozen=True)
class DayCongestionSeries:
    """Serie de congestión de toda la red para un día (TTH-13 / CT-13.1, Formato B).

    Compacta: por arista, un array ``levels`` de niveles 0-5 muestreados a paso
    constante ``step_s`` desde ``t0`` (primer instante con datos) hasta
    ``coverage_end`` (último instante con datos, límite del control temporal HU-23
    CA-23.2). Día sin datos → ``edges`` vacío y campos temporales en ``None`` (señal
    de ausencia, CA-23.7), no error.
    """
    day: date
    t0: datetime | None
    step_s: int | None
    coverage_end: datetime | None
    edges: list[EdgeSeries]


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
