"""Test de aceptación §6.1 — gate del primer commit funcional de Fase 4.

Contrato literal §2.3: un vehículo cuyo bbox cae dentro del polígono cuenta
EXACTAMENTE 1 (§2.4), y la entrada preserva la identidad del vehículo.
"""
from src.vision.domain.entities import DetectedVehicle, ZoneVehicleCount
from src.vision.domain.value_objects import VehicleId, ZoneId
from src.vision.infrastructure.zones.zone_counter import ZoneCounter


def test_zone_counter_counts_vehicle_inside_polygon():
    counter = ZoneCounter({
        ZoneId("zone_test"): [(100, 100), (200, 100), (200, 200), (100, 200)],
    })
    detection = DetectedVehicle(
        id=VehicleId("vehicle_test"),
        type="car",
        confidence=0.95,
        bbox=(140, 140, 160, 160),  # centroide (150, 150) claramente dentro
        timestamp=0.0,
    )

    result = counter.count(detections=[detection], frame_id=1)

    assert ZoneId("zone_test") in result
    zvc = result[ZoneId("zone_test")]
    assert isinstance(zvc, ZoneVehicleCount)
    assert zvc.count == 1                                  # §2.4 — exacto, no >= 1
    assert VehicleId("vehicle_test") in zvc.vehicle_ids    # preservación de identidad
