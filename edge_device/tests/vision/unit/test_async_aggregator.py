"""Tests del `AsyncTrafficAggregator` enfocados en §11 / DHU-026.

Tres familias:
- Constructor + lifecycle (force_flush + stop limpio + counters como properties).
- §11.1: errores de `save` → counter `aggregation_errors`, worker continúa,
  sin retry.
- §11.2: output queue llena → DROP-NEWEST + counter `data_dropped`.
- DHU-026 refinamiento: save y push a output queue son paths INDEPENDIENTES
  para el mismo TrafficData (un fallo de save NO impide el push; un drop
  de la output queue NO revierte el save).
"""
import threading

import pytest

from src.vision.application.aggregators.async_aggregator import (
    AsyncTrafficAggregator,
)
from src.vision.domain.entities import (
    DetectedVehicle,
    FrameAnalysis,
    TrafficData,
    ZoneVehicleCount,
)
from src.vision.domain.value_objects import CameraId, VehicleId, ZoneId


_CAM = CameraId("cam_test")
_ZONE_A = ZoneId("zone_a")
_ZONE_B = ZoneId("zone_b")


class _FakeRepository:
    """Implementa `TrafficRepository`. Permite fallar el `save` en demanda."""

    def __init__(self, fail_on_save: bool = False) -> None:
        self.saved: list[TrafficData] = []
        self.fail_on_save = fail_on_save
        self.save_call_count = 0
        self._lock = threading.Lock()

    def save(self, data: TrafficData) -> None:
        with self._lock:
            self.save_call_count += 1
            if self.fail_on_save:
                raise RuntimeError("simulated DB failure (§11.1)")
            self.saved.append(data)


def _veh(vid: str, vtype: str = "car") -> DetectedVehicle:
    return DetectedVehicle(
        id=VehicleId(vid),
        type=vtype,
        confidence=0.9,
        bbox=(0, 0, 10, 10),
        timestamp=0.0,
    )


def _zvc(zone: ZoneId, ids: list[str]) -> ZoneVehicleCount:
    return ZoneVehicleCount(
        zone_id=zone,
        count=len(ids),
        vehicle_ids=[VehicleId(i) for i in ids],
        occupancy=0.1,
    )


def _frame_one_zone(fid: int, ids: list[str]) -> FrameAnalysis:
    vehicles = [_veh(i) for i in ids]
    return FrameAnalysis(
        frame_id=fid,
        timestamp=float(fid),
        vehicles=vehicles,
        unique_vehicles=len(ids),
        zones={_ZONE_A: _zvc(_ZONE_A, ids)},
    )


def _frame_two_zones(fid: int, ids_a: list[str], ids_b: list[str]) -> FrameAnalysis:
    vehicles = [_veh(i) for i in {*ids_a, *ids_b}]
    return FrameAnalysis(
        frame_id=fid,
        timestamp=float(fid),
        vehicles=vehicles,
        unique_vehicles=len({v.id for v in vehicles}),
        zones={
            _ZONE_A: _zvc(_ZONE_A, ids_a),
            _ZONE_B: _zvc(_ZONE_B, ids_b),
        },
    )


def _aggregator(
    repo: _FakeRepository,
    *,
    output_queue_maxsize: int = 256,
    window_duration_s: float = 3600.0,
) -> AsyncTrafficAggregator:
    return AsyncTrafficAggregator(
        repository=repo,
        camera_id=_CAM,
        window_duration_s=window_duration_s,
        output_queue_maxsize=output_queue_maxsize,
    )


# ---------------------------------------------------------------------------
# Constructor + lifecycle + counters como properties.
# ---------------------------------------------------------------------------


def test_constructor_rejects_non_positive_window_duration():
    repo = _FakeRepository()
    with pytest.raises(ValueError, match="window_duration_s"):
        AsyncTrafficAggregator(
            repository=repo, camera_id=_CAM, window_duration_s=0.0
        )
    with pytest.raises(ValueError, match="window_duration_s"):
        AsyncTrafficAggregator(
            repository=repo, camera_id=_CAM, window_duration_s=-1.0
        )


def test_force_flush_returns_computed_traffic_data_and_persists():
    """Happy path: force_flush retorna TrafficData y el repo recibió el save."""
    repo = _FakeRepository()
    aggregator = _aggregator(repo)
    try:
        aggregator.add(_frame_one_zone(1, ["v1"]))
        result = aggregator.force_flush()
        assert len(result) == 1
        assert len(repo.saved) == 1
        assert result[0] is repo.saved[0]
        # Counters limpios cuando no hay errores ni drops.
        assert aggregator.aggregation_errors == 0
        assert aggregator.data_dropped == 0
    finally:
        aggregator.stop()


def test_stop_idempotent_and_force_flush_after_stop_returns_empty():
    repo = _FakeRepository()
    aggregator = _aggregator(repo)
    aggregator.add(_frame_one_zone(1, ["v1"]))
    aggregator.force_flush()
    aggregator.stop()
    aggregator.stop()  # segunda llamada no rompe
    # Post-stop, force_flush devuelve lo que quedara en la queue (vacía acá).
    assert aggregator.force_flush() == []


# ---------------------------------------------------------------------------
# §11.1 — errores de save.
# ---------------------------------------------------------------------------


def test_save_error_increments_counter_and_worker_continues():
    """§11.1: save falla → logger.exception + aggregation_errors + continúa loop."""
    repo = _FakeRepository(fail_on_save=True)
    aggregator = _aggregator(repo)
    try:
        # Primera ventana: el save falla, el worker no muere.
        aggregator.add(_frame_one_zone(1, ["v1"]))
        aggregator.force_flush()
        assert aggregator.aggregation_errors == 1

        # Segunda ventana: el worker procesa normal a pesar del fallo previo.
        aggregator.add(_frame_one_zone(2, ["v2"]))
        aggregator.force_flush()
        assert aggregator.aggregation_errors == 2
        assert repo.save_call_count == 2
    finally:
        aggregator.stop()


def test_save_error_does_not_block_output_queue_push_DHU_026():
    """DHU-026: save falla, pero el TrafficData igual entra a la output queue.

    Es el invariante clave de la independencia save vs output-queue:
    "Un fallo de save NO saca el item de la output queue — la operación en
    vivo sobrevive, exactamente lo que §11.1 promete."
    """
    repo = _FakeRepository(fail_on_save=True)
    aggregator = _aggregator(repo)
    try:
        aggregator.add(_frame_one_zone(1, ["v1"]))
        result = aggregator.force_flush()
        # Save falló (counter incrementado, repo.saved vacío).
        assert aggregator.aggregation_errors == 1
        assert repo.saved == []
        # PERO el TrafficData igual está disponible en la output queue.
        assert len(result) == 1
        assert result[0].zone_id == _ZONE_A
    finally:
        aggregator.stop()


# ---------------------------------------------------------------------------
# §11.2 — output queue full → DROP-NEWEST.
# ---------------------------------------------------------------------------


def test_output_queue_full_drops_newest_and_increments_counter():
    """§11.2: con maxsize=1 y 2 zonas en la ventana, el 2do push se dropea."""
    repo = _FakeRepository()
    aggregator = _aggregator(repo, output_queue_maxsize=1)
    try:
        aggregator.add(_frame_two_zones(1, ["a"], ["b"]))
        result = aggregator.force_flush()
        # 1 zona pasó a la output queue, 1 se dropeó.
        assert len(result) == 1
        assert aggregator.data_dropped == 1
    finally:
        aggregator.stop()


def test_output_queue_drop_does_not_rollback_save_DHU_026():
    """DHU-026: queue full dropea, pero ambos saves al repo ocurrieron.

    Es la independencia inversa: si la output queue está llena y un push
    falla, el save ya ocurrió antes — no se revierte.
    """
    repo = _FakeRepository()
    aggregator = _aggregator(repo, output_queue_maxsize=1)
    try:
        aggregator.add(_frame_two_zones(1, ["a"], ["b"]))
        result = aggregator.force_flush()

        # Output queue: 1 entró, 1 fue dropeada.
        assert len(result) == 1
        assert aggregator.data_dropped == 1
        # Repo: ambos saves ocurrieron (drop NO revierte save).
        assert len(repo.saved) == 2
        assert repo.save_call_count == 2
        assert aggregator.aggregation_errors == 0
    finally:
        aggregator.stop()


# ---------------------------------------------------------------------------
# flush() vs force_flush() y orden con stop().
# ---------------------------------------------------------------------------


def test_flush_is_non_blocking_and_returns_already_computed():
    """flush() retorna lo ya computado y disponible — no dispara cómputo."""
    repo = _FakeRepository()
    aggregator = _aggregator(repo)
    try:
        # add no procesa ni computa nada (timer del worker no disparó).
        aggregator.add(_frame_one_zone(1, ["v1"]))
        # flush() sin force previo: la output queue está vacía.
        assert aggregator.flush() == []
        # force_flush dispara cómputo y retorna lo computado.
        result = aggregator.force_flush()
        assert len(result) == 1
        # flush() posterior está vacío (force_flush ya drenó).
        assert aggregator.flush() == []
    finally:
        aggregator.stop()
