"""
Simple speed estimator.
"""
from dataclasses import replace
from typing import Dict, List, Tuple

from ...domain.entities import DetectedVehicle
from ...domain.protocols import SpeedEstimator

_WINDOW_SECONDS = 1.0
_MIN_TIME_DIFF = 0.1


class SimpleSpeedEstimator(SpeedEstimator):
    """Estimates speed from the pixel distance the bbox bottom travels over time."""

    def __init__(self, pixels_per_meter: float = 10.0, fps: float = 30.0):
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.history: Dict[str, List[Tuple[float, float]]] = {}  # id -> [(timestamp, center_y)]

    def estimate(self, vehicles: List[DetectedVehicle]) -> List[DetectedVehicle]:
        current_time = vehicles[0].timestamp if vehicles else 0

        results: List[DetectedVehicle] = []
        for vehicle in vehicles:
            if not vehicle.id:
                results.append(vehicle)
                continue

            # Bottom-center is a better proxy for the ground plane.
            center_y = vehicle.bbox[3]

            track = self.history.setdefault(vehicle.id, [])
            track.append((current_time, center_y))
            # Keep only the recent window.
            track = [(t, y) for t, y in track if current_time - t < _WINDOW_SECONDS]
            self.history[vehicle.id] = track

            speed_kmh = self._speed_from_track(track)
            if speed_kmh is not None:
                # DetectedVehicle is frozen: produce a new instance instead of mutating.
                results.append(replace(vehicle, speed=speed_kmh))
            else:
                results.append(vehicle)

        return results

    def _speed_from_track(self, track: List[Tuple[float, float]]):
        if len(track) < 2:
            return None
        t_first, y_first = track[0]
        t_last, y_last = track[-1]
        time_diff = t_last - t_first
        if time_diff <= _MIN_TIME_DIFF:
            return None
        dist_meters = abs(y_last - y_first) / self.pixels_per_meter
        return dist_meters / time_diff * 3.6
