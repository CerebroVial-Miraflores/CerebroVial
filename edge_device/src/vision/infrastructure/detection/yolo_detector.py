"""
YOLO-based vehicle detector.
"""
import logging
import time

import numpy as np

from ...domain.entities import DetectedVehicle
from ...domain.protocols import VehicleDetector
from cerebrovial_shared.logging import setup_logger, log_execution_time
from cerebrovial_shared.exceptions import DetectionError
from cerebrovial_shared.lfs_check import assert_real_binary

# COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
_TARGET_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class YoloDetector(VehicleDetector):
    """Implementation of VehicleDetector using YOLO."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.5,
        model=None,
    ):
        self.conf_threshold = conf_threshold
        self.target_classes = dict(_TARGET_CLASSES)
        self.logger = setup_logger(__name__)

        if model is None:
            # Real path: heavy deps imported lazily so the module collects
            # in environments without ultralytics/torch installed.
            import torch
            from ultralytics import YOLO

            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            self.logger.info(f"Using inference device: {device}")

            assert_real_binary(model_path)
            self._model = YOLO(model_path)
            self._model.to(device)
        else:
            # Injected model (tests): no weights on disk, so assert_real_binary
            # must NOT run here.
            self._model = model

    @log_execution_time(logging.getLogger(__name__))
    def detect(self, frame: np.ndarray, frame_id: int = 0) -> list[DetectedVehicle]:
        """Detect vehicles in a single frame, returning domain entities."""
        try:
            results = self._model(frame, verbose=False, conf=self.conf_threshold)[0]

            vehicles: list[DetectedVehicle] = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                if class_id not in self.target_classes:
                    continue

                confidence = float(box.conf[0])
                if confidence < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicles.append(
                    DetectedVehicle(
                        id=f"{frame_id}_{len(vehicles)}",  # temporary; tracking assigns the stable id
                        type=self.target_classes[class_id],
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        timestamp=time.time(),
                    )
                )

            return vehicles
        except Exception as e:
            self.logger.error(f"Detection failed on frame {frame_id}: {e}")
            raise DetectionError(f"YOLO inference failed: {e}") from e
