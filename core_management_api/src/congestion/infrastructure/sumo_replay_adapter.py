"""Adaptador de fuente: replay de un día de dataset SUMO pre-generado (CT-12.4).

Implementa la interfaz ``CongestionFeed`` (CT-12.3): reproduce en streaming un día
del dataset por arista (Parquet, granularidad 60 s, 1440 pasos) y emite el nivel de
congestión por arista. La fuente está DETRÁS de la interfaz, de modo que cambiarla
(SUMO en vivo, Waze real) no toca a los consumidores.

NO requiere un proceso SUMO corriendo: reproduce datos ya generados.

Dependencias de herramienta (NO del runtime de la API): ``pyarrow`` (lee el Parquet,
gitignored/regenerable) y ``cerebrovial_simulation.jam_level`` (la derivación 0-5,
reusada NO reescrita). Ambas se importan de forma perezosa para no contaminar el
import-graph de la API servida.

--- REGLA CRÍTICA DEL NaN (no repetir el matiz del perfilado) ---
``ratio_to_jam_level`` con un NaN crudo devolvería 4 (los ``<`` con NaN dan False y
cae al ``return 4``), lo cual es INCORRECTO para un bucket sin observación. En el
dataset, un bucket vacío sale con ``sampledSeconds=0`` y SIN atributo ``speed`` (ver
convención en ``simulation/.../dataset/generate.py``, commit 5af501b0), por lo que
``speed``/``speedRelative`` quedan NaN en el Parquet. Este adaptador intercepta ese
caso ANTES de llamar a la función: NaN → nivel 0 (flujo libre), ``speed_mps`` = la
velocidad de flujo libre de la arista; ``delay_seconds`` = 0. La función real solo
ve ratios válidos.
"""
from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Iterator

from ..application.feed import CongestionFeed, EdgeCongestion
from .repositories import WazeJamRow, WazeJamsRepo

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_PARQUET_DIR = _REPO_ROOT / "simulation/data/datasets/miraflores_laborable_60d"
_DEFAULT_NET = _REPO_ROOT / "simulation/conf/network/miraflores.net.xml"
_DEFAULT_MAPPING = _REPO_ROOT / "ia_prediction_service/src/data/artifacts/miraflores_graph_lcc_mapping.json"

# Epoch base del día reproducido: lunes laborable, naive (la columna es timestamp
# without time zone). El timestep (s del día simulado, 0..86340) se suma a esta base.
DAY_EPOCH = datetime(2025, 1, 6, 0, 0, 0)  # lunes
# Namespace fijo para event_uuid determinista (idempotencia de la pre-siembra).
_UUID_NS = uuid.UUID("8f6c1d2e-0a3b-5c7d-9e10-aa11bb22cc33")

# Centinelas (CT-12.4 / mapeo): sin fuente en el feed SUMO V1 (ver DHU-028).
SENTINEL_JAM_LENGTH_M = -1
SENTINEL_ROAD_TYPE = 0


def _event_uuid(edge_id: str, ts: datetime) -> str:
    return str(uuid.uuid5(_UUID_NS, f"{edge_id}|{ts.isoformat()}"))


class SumoReplayAdapter(CongestionFeed):
    """Replay de un día del dataset por arista, detrás de la interfaz de feed."""

    def __init__(
        self,
        day: str = "seed062",
        *,
        parquet_dir: Path | str = _DEFAULT_PARQUET_DIR,
        net_path: Path | str = _DEFAULT_NET,
        mapping_path: Path | str = _DEFAULT_MAPPING,
    ) -> None:
        self.day = day
        self.parquet_dir = Path(parquet_dir)
        self.net_path = Path(net_path)
        self.mapping_path = Path(mapping_path)
        self._loaded = False
        # poblados por _load():
        self._universe: list[str] = []
        self._freeflow: dict[str, float] = {}
        self._edge_ids = None   # np.ndarray[str]
        self._timestep = None   # np.ndarray[int]
        self._speed = None      # np.ndarray[float] (NaN = sin observación)
        self._speed_rel = None  # np.ndarray[float] (NaN = sin observación)
        self._timeloss = None   # np.ndarray[float]

    # --- carga perezosa (pyarrow + net.xml + mapping) ---

    def _load(self) -> None:
        if self._loaded:
            return
        import json
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._universe = [n["sumo_edge"] for n in json.loads(self.mapping_path.read_text())["nodes"]]
        uni_set = set(self._universe)
        self._freeflow = self._read_freeflow(uni_set)

        path = self.parquet_dir / f"day_{self.day}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Parquet del día no encontrado: {path}")
        t = pq.read_table(path, columns=["edge_id", "timestep", "speed", "timeLoss", "speedRelative"])
        t = t.filter(pa.compute.is_in(t.column("edge_id"), value_set=pa.array(list(uni_set))))
        self._edge_ids = t.column("edge_id").to_pylist()
        self._timestep = np.asarray(t.column("timestep"))
        self._speed = np.asarray(t.column("speed"), dtype=float)
        self._speed_rel = np.asarray(t.column("speedRelative"), dtype=float)
        self._timeloss = np.asarray(t.column("timeLoss"), dtype=float)
        self._loaded = True

    def _read_freeflow(self, universe: set[str]) -> dict[str, float]:
        """Velocidad de flujo libre por arista = speed del lane 0 en el .net.xml."""
        import xml.etree.ElementTree as ET
        ff: dict[str, float] = {}
        for _ev, el in ET.iterparse(str(self.net_path), events=("end",)):
            if el.tag == "edge":
                eid = el.get("id")
                if eid in universe:
                    lanes = el.findall("lane")
                    if lanes and lanes[0].get("speed") is not None:
                        ff[eid] = float(lanes[0].get("speed"))
                el.clear()
        missing = universe - ff.keys()
        if missing:
            raise RuntimeError(f"{len(missing)} aristas sin speed de flujo libre: {list(missing)[:5]}")
        return ff

    # --- derivación (reusa la función real; intercepta NaN) ---

    def _level_speed_delay(self, sr: float, sp: float, tl: float, edge_id: str) -> tuple[int, float, int]:
        """Devuelve (congestion_level 0-5, speed_mps, delay_seconds) para un bucket."""
        from cerebrovial_simulation.jam_level import ratio_to_jam_level  # reuso, no reescritura
        if math.isnan(sr):  # bucket sin observación -> flujo libre (NO llamar a la función)
            return 0, self._freeflow[edge_id], 0
        level = ratio_to_jam_level(float(sr))
        speed_mps = self._freeflow[edge_id] if math.isnan(sp) else float(sp)
        delay = 0 if math.isnan(tl) else int(round(float(tl)))
        return level, speed_mps, delay

    def _row_at(self, i: int) -> WazeJamRow:
        edge_id = self._edge_ids[i]
        ts = DAY_EPOCH + timedelta(seconds=int(self._timestep[i]))
        level, speed_mps, delay = self._level_speed_delay(
            self._speed_rel[i], self._speed[i], self._timeloss[i], edge_id
        )
        return WazeJamRow(
            event_uuid=_event_uuid(edge_id, ts),
            snapshot_timestamp=ts,
            edge_id=edge_id,
            speed_mps=speed_mps,
            delay_seconds=delay,
            congestion_level=level,
            jam_length_m=SENTINEL_JAM_LENGTH_M,
            road_type=SENTINEL_ROAD_TYPE,
        )

    # --- interfaz CongestionFeed (vista del consumidor) ---

    def timesteps(self) -> list[int]:
        self._load()
        import numpy as np
        return [int(t) for t in np.unique(self._timestep)]

    def levels_at(self, timestep: int) -> list[EdgeCongestion]:
        self._load()
        import numpy as np
        idx = np.where(self._timestep == timestep)[0]
        out = []
        for i in idx:
            r = self._row_at(int(i))
            out.append(EdgeCongestion(r.edge_id, r.congestion_level, r.snapshot_timestamp))
        return out

    def snapshots(self) -> Iterable[EdgeCongestion]:
        self._load()
        for i in range(len(self._edge_ids)):
            r = self._row_at(i)
            yield EdgeCongestion(r.edge_id, r.congestion_level, r.snapshot_timestamp)

    # --- persistencia ---

    def _rows(self) -> Iterator[WazeJamRow]:
        self._load()
        for i in range(len(self._edge_ids)):
            yield self._row_at(i)

    def preseed(self, repo: WazeJamsRepo, *, chunk: int = 10_000) -> dict:
        """Modo PRE-SIEMBRA: deriva los 375×1440 snapshots e inserta en lotes.

        Idempotente: event_uuid determinista + ON CONFLICT DO NOTHING. Tras insertar,
        puebla geom desde graph_edges (UPDATE-join). Devuelve métricas de la corrida.
        """
        self._load()
        n_rows = repo.batch_insert(self._rows(), chunk=chunk)
        n_geom = repo.populate_geom_from_edges()
        return {"day": self.day, "rows_processed": n_rows, "geom_updated": n_geom,
                "edges": len(self._universe), "timesteps": len(self.timesteps())}

    def live_replay(
        self, repo: WazeJamsRepo, *, speedup: float = 1.0, commit_each_step: bool = True,
        max_steps: int | None = None, wake: "Callable[[], None] | None" = None,
    ) -> int:
        """Modo REPLAY EN VIVO: gotea los snapshots paso a paso a cadencia configurable.

        Comparte TODA la derivación con ``preseed``; difiere solo en el ritmo de
        escritura: tras cada paso duerme ``60 / speedup`` segundos (60 s simulados por
        paso). ``speedup`` > 1 acelera; ``speedup`` real-time = 1.

        ``wake`` (CT-12.6): callback invocado tras escribir+commitear cada paso —
        ahí se publica el wake-up del broadcaster SSE de red. Se inyecta para NO
        acoplar el adaptador al broadcaster (el wiring lo hace quien orquesta el
        replay in-process). La pre-siembra NO emite wakes (es carga batch).
        Devuelve el nº de pasos emitidos.
        """
        steps = self.timesteps()
        if max_steps is not None:
            steps = steps[:max_steps]
        period = 60.0 / speedup if speedup > 0 else 0.0
        for n, ts in enumerate(steps):
            for ec in self._rows_at_step(ts):
                repo.insert_one(ec)
            if commit_each_step:
                repo.session.commit()
            repo.populate_geom_from_edges()
            if commit_each_step:
                repo.session.commit()
            if wake is not None:
                wake()  # CT-12.6: dispara el wake-up de red tras escribir el paso
            if period and n < len(steps) - 1:
                time.sleep(period)
        return len(steps)

    def _rows_at_step(self, timestep: int) -> list[WazeJamRow]:
        self._load()
        import numpy as np
        return [self._row_at(int(i)) for i in np.where(self._timestep == timestep)[0]]
