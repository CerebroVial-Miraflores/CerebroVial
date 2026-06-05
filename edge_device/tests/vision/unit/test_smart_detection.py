"""Tests del SmartDetectionProcessor — skip-cada-N puro (Fase 5d).

Los dos xfails C1.7 previos (`test_interpolation_logic`, `test_trajectory_update`)
testeaban funcionalidad que el refactor de Sesión 1 ya había marcado como
movida al tracker (ByteTrack con `lost_track_buffer=60`) y que Sesión 3
declaró fuera del scope de este processor (decisión #5d del plan de Fase 5).
Verificación del tracker hecha en el paso 0 de 5d.

Esos dos tests se eliminan en lugar de "resolverse" — no quedan en el set
xfails como deuda muerta.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.vision.application.processors.smart_detection import (
    SmartDetectionProcessor,
)
from src.vision.domain.entities import DetectedVehicle, Frame
from src.vision.domain.value_objects import VehicleId


def _img():
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _frame(fid: int) -> Frame:
    return Frame(id=fid, timestamp=float(fid) * 0.01, image=_img())


def test_detection_frequency():
    """Detector llamado en frames 0, 3, 6, ...; el resto son skip con FrameAnalysis vacío."""
    detector = MagicMock()
    metrics = MagicMock()
    detector.detect.return_value = [
        DetectedVehicle(
            id=VehicleId("1"),
            type="car",
            confidence=0.9,
            bbox=(0, 0, 10, 10),
            timestamp=0.0,
        )
    ]
    processor = SmartDetectionProcessor(
        detector=detector,
        detect_every_n=3,
        metrics_collector=metrics,
    )

    # Frame 0: detecta (primer frame, sentinel _last_detection_frame=-1).
    a0 = processor._process(_frame(0), None)
    assert detector.detect.call_count == 1
    assert a0 is not None
    assert a0.frame_id == 0
    assert len(a0.vehicles) == 1
    assert a0.unique_vehicles == 1
    assert a0.zones == {}
    assert a0.detection_ran is True  # C1: corrió inferencia
    metrics.record_detection.assert_called_once()

    # Frame 1: skip.
    a1 = processor._process(_frame(1), None)
    assert detector.detect.call_count == 1
    assert a1.vehicles == []
    assert a1.unique_vehicles == 0
    assert a1.zones == {}
    assert a1.detection_ran is False  # C1: skip → visualización persiste boxes

    # Frame 2: skip.
    processor._process(_frame(2), None)
    assert detector.detect.call_count == 1

    # Frame 3: detecta (>= 3 frames desde la última detección).
    processor._process(_frame(3), None)
    assert detector.detect.call_count == 2


def test_smart_detection_rejects_invalid_n():
    detector = MagicMock()
    with pytest.raises(ValueError, match="detect_every_n"):
        SmartDetectionProcessor(detector=detector, detect_every_n=0)
    with pytest.raises(ValueError, match="detect_every_n"):
        SmartDetectionProcessor(detector=detector, detect_every_n=-1)


def test_detect_every_n_one_runs_detector_every_frame():
    """N=1: caso degenerado, detección en cada frame, sin skip."""
    detector = MagicMock()
    detector.detect.return_value = []
    processor = SmartDetectionProcessor(detector=detector, detect_every_n=1)

    for fid in range(5):
        processor._process(_frame(fid), None)

    assert detector.detect.call_count == 5
