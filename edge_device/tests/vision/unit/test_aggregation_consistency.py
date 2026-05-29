"""Consistencia del cómputo del AsyncTrafficAggregator en escenarios concretos.

Migrado de la versión Sesión 1 (sync aggregator + entity shape viejo) al shape
canónico de §5.4 + AsyncTrafficAggregator (DHU-026). Preserva la intención
original: verificar que el cómputo a través de una ventana de frames produce
TrafficData correctos en casos típicos (voto mayoritario, múltiples vehículos
estables).

Estos tests usan `force_flush()` para sincronizar con el worker; los tests
específicos de §11 / DHU-026 (DROP-NEWEST, save errors, independencia save vs
output-queue) viven en `test_async_aggregator.py`.
"""
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


class _FakeRepository:
    """Implementa TrafficRepository en memoria para los tests."""

    def __init__(self) -> None:
        self.saved: list[TrafficData] = []

    def save(self, data: TrafficData) -> None:
        self.saved.append(data)


_CAM = CameraId("cam1")
_ZONE = ZoneId("zone1")


def _veh(vid: str, vtype: str) -> DetectedVehicle:
    return DetectedVehicle(
        id=VehicleId(vid),
        type=vtype,
        confidence=0.9,
        bbox=(0, 0, 10, 10),
        timestamp=0.0,
    )


def _zvc(ids: list[str]) -> ZoneVehicleCount:
    return ZoneVehicleCount(
        zone_id=_ZONE,
        count=len(ids),
        vehicle_ids=[VehicleId(i) for i in ids],
        occupancy=0.1,
    )


def _frame(fid: int, vehicles: list[DetectedVehicle]) -> FrameAnalysis:
    return FrameAnalysis(
        frame_id=fid,
        timestamp=float(fid),
        vehicles=vehicles,
        unique_vehicles=len({v.id for v in vehicles}),
        zones={_ZONE: _zvc([v.id.value for v in vehicles])},
    )


def _new_aggregator(repo: _FakeRepository) -> AsyncTrafficAggregator:
    # window_duration_s grande: el worker no cierra ventanas por timeout
    # durante el test; el test cierra explícitamente con force_flush().
    return AsyncTrafficAggregator(
        repository=repo,
        camera_id=_CAM,
        window_duration_s=3600.0,
    )


def test_aggregation_consistency_majority_vote_resolves_conflicting_types():
    """Vehículo cambiando car→truck→car a través de 3 frames: voto = car.

    Migrado del test homónimo de Sesión 1. La intención (voto mayoritario por
    vehículo y suma de tipos = unique_vehicles) sobrevive contra el schema
    nuevo §5.4.
    """
    repo = _FakeRepository()
    aggregator = _new_aggregator(repo)
    try:
        aggregator.add(_frame(1, [_veh("1", "car")]))
        aggregator.add(_frame(2, [_veh("1", "truck")]))
        aggregator.add(_frame(3, [_veh("1", "car")]))

        result = aggregator.force_flush()

        # 1 zona ventana → 1 TrafficData.
        assert len(result) == 1
        td = result[0]
        # El repo recibió el mismo TrafficData (DHU-026: worker persiste +
        # push a output queue son paths independientes; con repo sano y queue
        # con espacio, ambos paths se completan).
        assert len(repo.saved) == 1
        assert repo.saved[0] is td

        # Voto mayoritario: car=2, truck=1 → car.
        assert td.unique_vehicles == 1
        assert td.vehicles_by_type == {"car": 1}
        # Validación canónica §5.4: sum(vehicles_by_type) == unique_vehicles
        # (ya garantizado por __post_init__ de TrafficData; lo verificamos
        # explícito acá para preservar el espíritu del test original).
        assert sum(td.vehicles_by_type.values()) == td.unique_vehicles
    finally:
        aggregator.stop()


def test_aggregation_consistency_multiple_stable_vehicles():
    """Dos vehículos estables car + truck en un frame: ambos sobreviven al voto."""
    repo = _FakeRepository()
    aggregator = _new_aggregator(repo)
    try:
        aggregator.add(_frame(1, [_veh("1", "car"), _veh("2", "truck")]))

        result = aggregator.force_flush()

        assert len(result) == 1
        td = result[0]
        assert len(repo.saved) == 1
        assert td.unique_vehicles == 2
        assert td.vehicles_by_type == {"car": 1, "truck": 1}
        assert sum(td.vehicles_by_type.values()) == td.unique_vehicles
    finally:
        aggregator.stop()
