"""Unit tests for SimpleSpeedEstimator (CT-08.2 — velocidad / CT-08.11c).

Pure-python derivation test: no heavy deps, no mocks. Also pins the
immutability contract — DetectedVehicle is frozen, so estimate() must return
new instances rather than mutating its inputs.
"""
import pytest

from src.vision.domain.entities import DetectedVehicle
from src.vision.infrastructure.tracking.speed_estimator import SimpleSpeedEstimator


def _vehicle(vid, y2, timestamp, speed=None):
    return DetectedVehicle(
        id=vid, type="car", confidence=0.9, bbox=(0, 0, 10, y2), timestamp=timestamp, speed=speed
    )


def test_single_frame_yields_no_speed():
    est = SimpleSpeedEstimator(pixels_per_meter=10.0)
    out = est.estimate([_vehicle("v1", 100, 0.0)])
    assert out[0].speed is None


def test_speed_derived_from_two_frames():
    est = SimpleSpeedEstimator(pixels_per_meter=10.0)
    est.estimate([_vehicle("v1", 100, 0.0)])
    out = est.estimate([_vehicle("v1", 150, 0.5)])
    # 50 px / 10 px-per-m = 5 m over 0.5 s = 10 m/s = 36 km/h
    assert out[0].speed == pytest.approx(36.0)


def test_frozen_input_is_not_mutated():
    est = SimpleSpeedEstimator(pixels_per_meter=10.0)
    est.estimate([_vehicle("v1", 100, 0.0)])
    second = _vehicle("v1", 150, 0.5)
    out = est.estimate([second])

    assert out[0].speed == pytest.approx(36.0)  # speed lives on the returned copy
    assert second.speed is None                 # original untouched


def test_vehicle_without_id_passes_through_unchanged():
    est = SimpleSpeedEstimator()
    v = _vehicle("", 100, 0.0)
    out = est.estimate([v])
    assert out == [v]
    assert out[0] is v
