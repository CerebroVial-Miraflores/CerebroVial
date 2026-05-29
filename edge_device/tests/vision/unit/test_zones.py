from src.vision.domain.entities import DetectedVehicle
from src.vision.domain.value_objects import VehicleId, ZoneId
from src.vision.infrastructure.zones.zone_counter import ZoneCounter

_SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def _vehicle(vid: str, bbox: tuple[int, int, int, int]) -> DetectedVehicle:
    return DetectedVehicle(
        id=VehicleId(vid), type="car", confidence=0.9, bbox=bbox, timestamp=0.0
    )


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


# ---------------------------------------------------------------------------
# Occupancy (DHU-025) — 4 casos: fuera, dentro, parcial, solapados-no-double-cuenta.
# Validan la elección de UNIÓN (no suma + clip) de §5.4.
# ---------------------------------------------------------------------------


def test_occupancy_bbox_completely_outside_is_zero():
    counter = ZoneCounter({ZoneId("z"): _SQUARE})
    # bbox lejos del polígono (0,0)-(100,100).
    v = _vehicle("v", bbox=(200, 200, 250, 250))
    zvc = counter.count(detections=[v], frame_id=1)[ZoneId("z")]
    assert zvc.occupancy == 0.0


def test_occupancy_bbox_fully_inside_equals_area_ratio():
    counter = ZoneCounter({ZoneId("z"): _SQUARE})
    # bbox 40x40 completamente dentro del polígono 100x100.
    v = _vehicle("v", bbox=(10, 10, 50, 50))
    zvc = counter.count(detections=[v], frame_id=1)[ZoneId("z")]
    # Ratio exacto: bbox_area / polygon_area.
    # Sin asumir el área literal que cv2.fillPoly produce (puede tener bordes
    # off-by-one), verificamos que occupancy esté cerca de 1600/10000 = 0.16.
    assert 0.14 <= zvc.occupancy <= 0.18


def test_occupancy_bbox_partial_overlap_in_range():
    counter = ZoneCounter({ZoneId("z"): _SQUARE})
    # bbox que cruza el borde derecho del polígono: 50% dentro.
    v = _vehicle("v", bbox=(80, 30, 130, 70))  # 50x40 = 2000; mitad dentro = 20*40 = 800
    zvc = counter.count(detections=[v], frame_id=1)[ZoneId("z")]
    # Esperado: ~800/10000 = 0.08.
    assert 0.0 < zvc.occupancy < 0.5
    assert 0.06 <= zvc.occupancy <= 0.10


def test_occupancy_overlapping_bboxes_use_union_not_sum():
    """Test clave de DHU-025: dos bboxes solapados NO double-cuentan.

    bbox A = (10, 10, 60, 60) — 50x50 = 2500 px.
    bbox B = (30, 30, 80, 80) — 50x50 = 2500 px.
    Overlap (interno A ∩ B) = (30, 30, 60, 60) = 30x30 = 900 px.
    Unión A ∪ B = 2500 + 2500 - 900 = 4100 px.
    Polígono = 100x100 = 10000 px.

    Suma + clip: 2500/10000 + 2500/10000 = 0.50.
    Unión (lo correcto, DHU-025): 4100/10000 = 0.41.

    El gate: occupancy_union DEBE estar cerca de 0.41, NO de 0.50.
    """
    counter = ZoneCounter({ZoneId("z"): _SQUARE})
    a = _vehicle("a", bbox=(10, 10, 60, 60))
    b = _vehicle("b", bbox=(30, 30, 80, 80))

    # Occupancy individual de cada bbox como baseline.
    occ_a = counter.count(detections=[a], frame_id=1)[ZoneId("z")].occupancy
    occ_b = counter.count(detections=[b], frame_id=1)[ZoneId("z")].occupancy
    occ_union = counter.count(detections=[a, b], frame_id=1)[ZoneId("z")].occupancy

    # Sanity: las occupancies individuales son ~0.25 cada una.
    assert 0.23 <= occ_a <= 0.27
    assert 0.23 <= occ_b <= 0.27

    # Gate DHU-025: la unión es estrictamente menor que la suma (no double-cuenta).
    assert occ_union < occ_a + occ_b - 0.05, (
        f"unión ({occ_union}) debe ser estrictamente menor que suma "
        f"({occ_a + occ_b}); si está cerca de la suma, el cómputo está sumando "
        f"con clip en vez de unir (viola DHU-025)."
    )

    # La unión incluye ambos, así que debe ser >= max individual.
    assert occ_union >= max(occ_a, occ_b)

    # Esperado por geometría: 4100/10000 = 0.41 (con tolerancia para fillPoly).
    assert 0.39 <= occ_union <= 0.43


def test_zone_vehicle_count_rejects_invalid_occupancy():
    """Validación de la entity (DHU-025): occupancy fuera de [0, 1] es error."""
    import pytest
    from src.vision.domain.entities import ZoneVehicleCount

    with pytest.raises(ValueError, match="occupancy"):
        ZoneVehicleCount(zone_id=ZoneId("z"), count=0, vehicle_ids=[], occupancy=1.5)
    with pytest.raises(ValueError, match="occupancy"):
        ZoneVehicleCount(zone_id=ZoneId("z"), count=0, vehicle_ids=[], occupancy=-0.1)
    # Los bordes [0.0, 1.0] son válidos.
    ZoneVehicleCount(zone_id=ZoneId("z"), count=0, vehicle_ids=[], occupancy=0.0)
    ZoneVehicleCount(zone_id=ZoneId("z"), count=0, vehicle_ids=[], occupancy=1.0)
