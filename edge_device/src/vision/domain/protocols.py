"""
Domain protocols for the Computer Vision module.
"""
from typing import Optional, Protocol

import numpy as np

from .entities import (
    DetectedVehicle,
    Frame,
    FrameAnalysis,
    TrafficData,
    ZoneVehicleCount,
)
from .value_objects import ZoneId


class VehicleDetector(Protocol):
    """Protocol for vehicle detection in a single frame."""
    def detect(self, frame: np.ndarray, frame_id: int) -> list[DetectedVehicle]: ...


class VehicleTracker(Protocol):
    """Protocol for assigning stable identities across frames."""
    def update(self, detections: list[DetectedVehicle]) -> list[DetectedVehicle]: ...


class SpeedEstimator(Protocol):
    """Protocol for speed estimation of tracked vehicles."""
    def estimate(self, vehicles: list[DetectedVehicle]) -> list[DetectedVehicle]: ...


class FrameProducer(Protocol):
    """Protocol for a source of frames (file, webcam, stream)."""
    def read(self) -> Optional[Frame]: ...
    def release(self) -> None: ...


class ZoneCounter(Protocol):
    """Protocol for counting vehicles per configured zone (polygon)."""
    def count(
        self,
        detections: list[DetectedVehicle],
        frame_id: int,
    ) -> dict[ZoneId, ZoneVehicleCount]: ...


class AsyncAggregator(Protocol):
    """Protocol for asynchronous aggregation using a worker thread + queue.

    Same compute contract as `SyncAggregator` (`add` + `flush`) plus
    `force_flush` (synchronous drain) and `stop` (worker lifecycle).

    Semantics of the four methods:

    - `add(analysis)`: non-blocking enqueue. The worker thread picks the
      analysis from its input queue asynchronously.
    - `flush() -> list[TrafficData]`: non-blocking. Returns the
      `TrafficData` items already computed and waiting in the output queue
      at call time. Items still being processed by the worker are NOT
      included in this call's return. Use case: periodic telemetry pull
      from the pipeline thread without pausing the worker.
    - `force_flush() -> list[TrafficData]`: blocking. Forces the worker to
      drain its input queue, computes everything pending, and returns the
      full result. Use case: clean shutdown before `stop()`, or test
      synchronization.
    - `stop() -> None`: signals the worker to terminate. Does NOT return
      data; callers that need final data should invoke `force_flush()`
      first.

    The caller (typically a use case in `application/`) is responsible for
    persistence via the `TrafficRepository` Protocol.
    """
    def add(self, analysis: FrameAnalysis) -> None: ...
    def flush(self) -> list[TrafficData]: ...
    def force_flush(self) -> list[TrafficData]: ...
    def stop(self) -> None: ...


class Broadcaster(Protocol):
    """Protocol for publishing traffic state to real-time consumers."""
    async def publish(self, data: TrafficData) -> None: ...
    def subscriber_count(self) -> int: ...
    def is_subscribed(self, subscriber_id: str) -> bool: ...


class FrameRenderer(Protocol):
    """Protocol for visually annotating frames (bbox, labels, ROI overlays).

    Returns annotated frame as `np.ndarray` of shape (H, W, 3), dtype uint8,
    BGR channel order (OpenCV convention), same H/W as input frame.
    """
    def render(self, frame: Frame, analysis: FrameAnalysis) -> np.ndarray: ...
