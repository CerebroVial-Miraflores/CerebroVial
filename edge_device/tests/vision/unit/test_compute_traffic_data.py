"""Tests unitarios de `compute_traffic_data` (función pura de Fase 5).

Schema canónico §5.4 de tth-08-fase1-diseno.md.
Interpretación de mean_occupancy: DHU-025 (unión).
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.vision.application.aggregators._compute import compute_traffic_data
from src.vision.domain.entities import (
    DetectedVehicle,
    FrameAnalysis,
    ZoneVehicleCount,
)
from src.vision.domain.value_objects import CameraId, VehicleId, ZoneId

_CAM = CameraId("cam_test")
_ZONE_A = ZoneId("zone_a")
_ZONE_B = ZoneId("zone_b")
_WS = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
_WE = _WS + timedelta(seconds=60)


def _veh(vid: str, vtype: str, speed: float | None = None) -> DetectedVehicle:
    return DetectedVehicle(
        id=VehicleId(vid),
        type=vtype,
        confidence=0.9,
        bbox=(0, 0, 10, 10),
        timestamp=_WS.timestamp(),
        speed=speed,
    )


def _zvc(zone: ZoneId, ids: list[str], occupancy: float = 0.0) -> ZoneVehicleCount:
    return ZoneVehicleCount(
        zone_id=zone,
        count=len(ids),
        vehicle_ids=[VehicleId(i) for i in ids],
        occupancy=occupancy,
    )


def _frame(
    fid: int,
    vehicles: list[DetectedVehicle],
    zones: dict[ZoneId, ZoneVehicleCount],
) -> FrameAnalysis:
    return FrameAnalysis(
        frame_id=fid,
        timestamp=_WS.timestamp() + fid * 0.1,
        vehicles=vehicles,
        unique_vehicles=len({v.id for v in vehicles}),
        zones=zones,
    )


# ---------------------------------------------------------------------------
# Casos base
# ---------------------------------------------------------------------------


def test_empty_analyses_returns_empty_list():
    assert compute_traffic_data([], _CAM, _WS, _WE, {}) == []


def test_window_end_not_after_start_raises():
    with pytest.raises(ValueError, match="window_end"):
        compute_traffic_data([], _CAM, _WS, _WS, {})


# ---------------------------------------------------------------------------
# Una zona con un vehículo: shape canónico completo.
# ---------------------------------------------------------------------------


def test_single_zone_single_vehicle_no_speed_no_segment():
    v = _veh("v1", "car", speed=None)
    frame = _frame(1, [v], {_ZONE_A: _zvc(_ZONE_A, ["v1"], occupancy=0.25)})

    result = compute_traffic_data([frame], _CAM, _WS, _WE, {_ZONE_A: None})

    assert len(result) == 1
    td = result[0]
    assert td.camera_id == _CAM
    assert td.zone_id == _ZONE_A
    assert td.window_start == _WS
    assert td.window_end == _WE
    assert td.window_duration_seconds == 60.0
    assert td.unique_vehicles == 1
    assert td.vehicles_by_type == {"car": 1}
    # Sin velocidad válida → None (§5.4).
    assert td.mean_speed_kmh is None
    # Sin calibración de segment_length → None (§5.4 + §5.5).
    assert td.density_vehicles_per_km is None
    # flow = 1 / 60s * 3600 = 60 veh/h.
    assert td.flow_vehicles_per_hour == pytest.approx(60.0)
    # mean_occupancy = único frame con 0.25.
    assert td.mean_occupancy == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Múltiples zonas: cada zona produce su propio TrafficData.
# ---------------------------------------------------------------------------


def test_multiple_zones_produce_one_traffic_data_each():
    va = _veh("va", "car")
    vb = _veh("vb", "truck")
    frame = _frame(
        1,
        [va, vb],
        {
            _ZONE_A: _zvc(_ZONE_A, ["va"], occupancy=0.1),
            _ZONE_B: _zvc(_ZONE_B, ["vb"], occupancy=0.4),
        },
    )

    result = compute_traffic_data([frame], _CAM, _WS, _WE, {_ZONE_A: None, _ZONE_B: None})

    assert [td.zone_id for td in result] == [_ZONE_A, _ZONE_B]  # ordenado por value
    assert result[0].vehicles_by_type == {"car": 1}
    assert result[1].vehicles_by_type == {"truck": 1}
    assert result[0].mean_occupancy == pytest.approx(0.1)
    assert result[1].mean_occupancy == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Ausencia de calibración (§5.4 + §5.5).
# ---------------------------------------------------------------------------


def test_density_is_none_when_segment_length_missing():
    frame = _frame(
        1, [_veh("v1", "car")], {_ZONE_A: _zvc(_ZONE_A, ["v1"], occupancy=0.1)}
    )
    result = compute_traffic_data([frame], _CAM, _WS, _WE, {_ZONE_A: None})
    assert result[0].density_vehicles_per_km is None


def test_density_computed_when_segment_length_present():
    frames = [
        _frame(i, [_veh(f"v{i}", "car")], {_ZONE_A: _zvc(_ZONE_A, [f"v{i}"], occupancy=0.05)})
        for i in range(5)
    ]
    # 5 vehículos únicos, segment_length=50m → density = 5/50*1000 = 100 veh/km.
    result = compute_traffic_data(frames, _CAM, _WS, _WE, {_ZONE_A: 50.0})
    assert result[0].density_vehicles_per_km == pytest.approx(100.0)


def test_mean_speed_is_none_when_no_valid_speeds():
    # Speeds None y 0 son excluidos.
    frames = [
        _frame(1, [_veh("v1", "car", speed=None)], {_ZONE_A: _zvc(_ZONE_A, ["v1"], occupancy=0.1)}),
        _frame(2, [_veh("v2", "car", speed=0.0)], {_ZONE_A: _zvc(_ZONE_A, ["v2"], occupancy=0.1)}),
    ]
    result = compute_traffic_data(frames, _CAM, _WS, _WE, {_ZONE_A: None})
    assert result[0].mean_speed_kmh is None


# ---------------------------------------------------------------------------
# mean_speed_kmh count-weighted (§5.4).
# ---------------------------------------------------------------------------


def test_mean_speed_is_count_weighted_not_simple_average():
    # Frame 1: 2 vehículos en zona a 20 km/h.
    # Frame 2: 5 vehículos en zona a 10 km/h.
    # Promedio simple: 15. Ponderado por count: (2*20 + 5*10) / (2+5) = 90/7 ≈ 12.857.
    f1_vehs = [_veh(f"a{i}", "car", speed=20.0) for i in range(2)]
    f2_vehs = [_veh(f"b{i}", "car", speed=10.0) for i in range(5)]
    f1 = _frame(1, f1_vehs, {_ZONE_A: _zvc(_ZONE_A, [v.id.value for v in f1_vehs], occupancy=0.1)})
    f2 = _frame(2, f2_vehs, {_ZONE_A: _zvc(_ZONE_A, [v.id.value for v in f2_vehs], occupancy=0.1)})

    result = compute_traffic_data([f1, f2], _CAM, _WS, _WE, {_ZONE_A: None})
    assert result[0].mean_speed_kmh == pytest.approx(90.0 / 7.0)


# ---------------------------------------------------------------------------
# Voto mayoritario por vehículo (§5.4).
# ---------------------------------------------------------------------------


def test_majority_vote_resolves_conflicting_types():
    # Mismo VehicleId aparece en 3 frames como ["car", "car", "truck"] → "car".
    frames = [
        _frame(1, [_veh("v1", "car")], {_ZONE_A: _zvc(_ZONE_A, ["v1"], occupancy=0.1)}),
        _frame(2, [_veh("v1", "car")], {_ZONE_A: _zvc(_ZONE_A, ["v1"], occupancy=0.1)}),
        _frame(3, [_veh("v1", "truck")], {_ZONE_A: _zvc(_ZONE_A, ["v1"], occupancy=0.1)}),
    ]
    result = compute_traffic_data(frames, _CAM, _WS, _WE, {_ZONE_A: None})
    assert result[0].unique_vehicles == 1
    assert result[0].vehicles_by_type == {"car": 1}


# ---------------------------------------------------------------------------
# flow_vehicles_per_hour: unidad correcta (no veh/min).
# ---------------------------------------------------------------------------


def test_flow_is_vehicles_per_hour_not_per_minute():
    # 10 vehículos únicos en ventana de 60s = 1 minuto.
    # flow = 10 / 60s * 3600 = 600 veh/h. (Si fuera veh/min sería 10.)
    frames = [
        _frame(
            i,
            [_veh(f"v{i}", "car")],
            {_ZONE_A: _zvc(_ZONE_A, [f"v{i}"], occupancy=0.0)},
        )
        for i in range(10)
    ]
    result = compute_traffic_data(frames, _CAM, _WS, _WE, {_ZONE_A: None})
    assert result[0].flow_vehicles_per_hour == pytest.approx(600.0)


# ---------------------------------------------------------------------------
# Zone con count = 0 + occupancy > 0: bbox cruza la zona pero centroide fuera.
# ---------------------------------------------------------------------------


def test_zone_with_zero_count_but_positive_occupancy_is_reported():
    # Caso: un frame con la zona registrada pero sin vehículos por centroide
    # (count=0, vehicle_ids=[]). La occupancy puede ser >0 (bbox cruza la zona
    # sin que el centroide caiga dentro). TrafficData se emite con 0s y la
    # occupancy promediada.
    frame = _frame(1, [], {_ZONE_A: _zvc(_ZONE_A, [], occupancy=0.07)})
    result = compute_traffic_data([frame], _CAM, _WS, _WE, {_ZONE_A: None})
    assert len(result) == 1
    td = result[0]
    assert td.unique_vehicles == 0
    assert td.vehicles_by_type == {}
    assert td.flow_vehicles_per_hour == 0.0
    assert td.mean_speed_kmh is None
    assert td.mean_occupancy == pytest.approx(0.07)


# ---------------------------------------------------------------------------
# Fase 3 — fallback zone-less (zones == {} en todos los frames).
# ---------------------------------------------------------------------------


def test_zoneless_cuenta_sobre_analysis_vehicles():
    # Sin zonas en NINGÚN frame → un solo TrafficData (zone_id="all") contado
    # sobre analysis.vehicles, la misma fuente que el overlay (mata "0 con boxes").
    # Dedup por id a nivel ventana: v1 aparece en los dos frames = 1 único.
    f0 = _frame(0, [_veh("v1", "car"), _veh("v2", "bus")], {})
    f1 = _frame(1, [_veh("v1", "car"), _veh("v3", "truck")], {})
    result = compute_traffic_data([f0, f1], _CAM, _WS, _WE, {})

    assert len(result) == 1
    td = result[0]
    assert td.zone_id == ZoneId("all")
    assert td.unique_vehicles == 3  # v1, v2, v3
    assert td.vehicles_by_type == {"car": 1, "bus": 1, "truck": 1}
    # flujo = 3 / 60s * 3600
    assert td.flow_vehicles_per_hour == pytest.approx(3 / 60 * 3600)


def test_zoneless_ocupacion_cero_densidad_none():
    # El fallback NO computa ocupación (relativa a polígono) ni densidad (sin
    # segment_length): occupancy=0.0, density=None — límite honesto.
    f0 = _frame(0, [_veh("v1", "car")], {})
    td = compute_traffic_data([f0], _CAM, _WS, _WE, {})[0]
    assert td.mean_occupancy == 0.0
    assert td.density_vehicles_per_km is None


def test_zoneless_velocidad_count_weighted():
    # mean_speed_kmh count-weighted sobre frames con velocidades válidas.
    f0 = _frame(0, [_veh("v1", "car", speed=40.0), _veh("v2", "car", speed=20.0)], {})
    td = compute_traffic_data([f0], _CAM, _WS, _WE, {})[0]
    # un frame, 2 vehículos: avg (40+20)/2 = 30
    assert td.mean_speed_kmh == pytest.approx(30.0)


def test_zoneless_sin_vehiculos_emite_cero_vivo():
    # Frames sin vehículos y sin zonas → un TrafficData con 0 (estado vivo "0
    # ahora"), no [] (que dejaría el panel sin payload/congelado).
    f0 = _frame(0, [], {})
    result = compute_traffic_data([f0], _CAM, _WS, _WE, {})
    assert len(result) == 1
    assert result[0].unique_vehicles == 0
    assert result[0].zone_id == ZoneId("all")


def test_fallback_cede_ante_zonas():
    # Si AL MENOS un frame tiene zonas calibradas, el zone-less NO aplica: gana el
    # camino por-zona (no aparece la zona sentinel "all").
    f0 = _frame(0, [_veh("v1", "car")], {_ZONE_A: _zvc(_ZONE_A, ["v1"], occupancy=0.3)})
    result = compute_traffic_data([f0], _CAM, _WS, _WE, {_ZONE_A: None})

    zone_ids = {td.zone_id for td in result}
    assert ZoneId("all") not in zone_ids
    assert zone_ids == {_ZONE_A}
