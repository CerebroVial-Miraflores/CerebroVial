"""Tests del serializador del tap de detecciones (Fase 4 Mitad A, D-019).

Check determinístico (sin stack vivo) de lo que el Commit 1 puede verificar:
- bbox normalizado correctamente a [0, 1] (división por dims del frame);
- clamp a [0, 1] de cajas que asoman fuera del frame;
- timestamps presentes y correctos (frame vs server, ambos en reloj del edge);
- forma del JSON esperada (claves y tipos);
- campos por-vehículo (id, type, confidence) preservados;
- payload vacío para cámara sin detecciones.
La verificación por curl contra el stack vivo (HLS real) la corre Cesar post-commit.
"""
from src.vision.domain.entities import DetectedVehicle, FrameAnalysis
from src.vision.domain.value_objects import VehicleId
from src.vision.presentation.serializers import (
    empty_detections_payload,
    serialize_detections,
)


def _vehicle(vid="v1", vtype="car", conf=0.9, bbox=(0, 0, 0, 0)):
    return DetectedVehicle(
        id=VehicleId(vid),
        type=vtype,
        confidence=conf,
        bbox=bbox,
        timestamp=1.0,
    )


def _analysis(vehicles, timestamp=123.456, detection_ran=True):
    return FrameAnalysis(
        frame_id=1,
        timestamp=timestamp,
        vehicles=vehicles,
        unique_vehicles=len(vehicles),
        zones={},
        detection_ran=detection_ran,
    )


def test_bbox_se_normaliza_a_0_1():
    # (640, 360, 1280, 720) sobre frame 1280x720 → (0.5, 0.5, 1.0, 1.0) exacto.
    analysis = _analysis([_vehicle(bbox=(640, 360, 1280, 720))])
    payload = serialize_detections(analysis, 1280, 720, server_timestamp=200.0, camera_id="camX")

    assert payload["detections"][0]["bbox"] == [0.5, 0.5, 1.0, 1.0]


def test_bbox_en_origen_es_cero():
    analysis = _analysis([_vehicle(bbox=(0, 0, 64, 72))])
    payload = serialize_detections(analysis, 1280, 720, server_timestamp=1.0, camera_id="camX")

    assert payload["detections"][0]["bbox"] == [0.0, 0.0, 0.05, 0.1]


def test_bbox_se_clampea_a_0_1():
    # Caja que asoma fuera del frame (negativo y > dim) → clamp a [0, 1].
    analysis = _analysis([_vehicle(bbox=(-50, -10, 1400, 800))])
    payload = serialize_detections(analysis, 1280, 720, server_timestamp=1.0, camera_id="camX")

    assert payload["detections"][0]["bbox"] == [0.0, 0.0, 1.0, 1.0]


def test_timestamps_presentes_y_correctos():
    analysis = _analysis([], timestamp=123.456)
    payload = serialize_detections(analysis, 1280, 720, server_timestamp=200.0, camera_id="camX")

    # frame_timestamp = instante de la detección; server_timestamp = "ahora" del edge.
    assert payload["frame_timestamp"] == 123.456
    assert payload["server_timestamp"] == 200.0
    # Edad de la caja (lo que el overlay usa para la frescura) = server - frame.
    assert payload["server_timestamp"] - payload["frame_timestamp"] == 76.544


def test_forma_del_json_y_campos_por_vehiculo():
    analysis = _analysis(
        [
            _vehicle(vid="v1", vtype="car", conf=0.91, bbox=(0, 0, 128, 72)),
            _vehicle(vid="v2", vtype="bus", conf=0.8, bbox=(640, 0, 1280, 360)),
        ]
    )
    payload = serialize_detections(analysis, 1280, 720, server_timestamp=5.0, camera_id="cam_larco")

    assert set(payload.keys()) == {
        "camera_id",
        "frame",
        "frame_timestamp",
        "server_timestamp",
        "detection_ran",
        "detections",
    }
    assert payload["camera_id"] == "cam_larco"
    assert payload["frame"] == {"width": 1280, "height": 720}
    assert payload["detection_ran"] is True
    assert len(payload["detections"]) == 2

    first = payload["detections"][0]
    assert set(first.keys()) == {"id", "type", "confidence", "bbox"}
    assert first["id"] == "v1"
    assert first["type"] == "car"
    assert first["confidence"] == 0.91
    second = payload["detections"][1]
    assert second["type"] == "bus"
    assert second["bbox"] == [0.5, 0.0, 1.0, 0.5]


def test_detection_ran_false_se_preserva():
    analysis = _analysis([], detection_ran=False)
    payload = serialize_detections(analysis, 1280, 720, server_timestamp=1.0, camera_id="camX")

    assert payload["detection_ran"] is False
    assert payload["detections"] == []


def test_dims_no_positivas_no_revientan():
    # Defensivo: 0/None no debe dividir por cero (cae a 1.0; el overlay clampea).
    analysis = _analysis([_vehicle(bbox=(10, 10, 20, 20))])
    payload = serialize_detections(analysis, 0, 0, server_timestamp=1.0, camera_id="camX")

    bbox = payload["detections"][0]["bbox"]
    assert all(0.0 <= c <= 1.0 for c in bbox)


def test_empty_payload_para_camara_sin_detecciones():
    payload = empty_detections_payload("camX", server_timestamp=10.0)

    assert payload["camera_id"] == "camX"
    assert payload["frame"] is None
    assert payload["frame_timestamp"] is None
    assert payload["server_timestamp"] == 10.0
    assert payload["detection_ran"] is False
    assert payload["detections"] == []
