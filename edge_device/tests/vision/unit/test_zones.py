from src.vision.domain.entities import DetectedVehicle
from src.vision.domain.value_objects import VehicleId, ZoneId
from src.vision.infrastructure.zones.zone_counter import ZoneCounter

_SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_zone_manager_initialization():
    counter = ZoneCounter({ZoneId("zone1"): _SQUARE})
    assert ZoneId("zone1") in counter.count(detections=[], frame_id=1)


def test_zone_manager_update():
    counter = ZoneCounter({ZoneId("zone1"): _SQUARE})
    # Vehículo dentro de la zona
    v1 = DetectedVehicle(id=VehicleId("1"), type="car", confidence=0.9, bbox=(10, 10, 50, 50), timestamp=0.0)
    # Vehículo fuera de la zona
    v2 = DetectedVehicle(id=VehicleId("2"), type="car", confidence=0.9, bbox=(150, 150, 190, 190), timestamp=0.0)

    result = counter.count(detections=[v1, v2], frame_id=1)

    assert result[ZoneId("zone1")].count == 1
    assert VehicleId("1") in result[ZoneId("zone1")].vehicle_ids


def test_zone_manager_empty_detections():
    counter = ZoneCounter({ZoneId("zone1"): _SQUARE})
    assert counter.count(detections=[], frame_id=1)[ZoneId("zone1")].count == 0
