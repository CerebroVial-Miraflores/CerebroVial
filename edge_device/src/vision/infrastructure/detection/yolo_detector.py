"""
YOLO-based vehicle detector.
"""
import logging
import time
from typing import Optional

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

        # Device de inferencia; None en el path inyectado (tests, sin torch).
        # Se conserva para que release() limpie la caché del device correcto.
        self._device: str | None = None

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
            self._device = device
            self.logger.info(f"Using inference device: {device}")

            assert_real_binary(model_path)
            self._model = YOLO(model_path)
            self._model.to(device)
        else:
            # Injected model (tests): no weights on disk, so assert_real_binary
            # must NOT run here.
            self._model = model

    def release(self) -> None:
        """Libera el modelo YOLO y la caché del device (C1).

        On-demand exige que tras la baja de una cámara su modelo NO quede en
        memoria. Soltamos la referencia al modelo, forzamos un `gc.collect()` y
        vaciamos la caché del acelerador (cuda/mps) si aplica. Idempotente:
        llamarla dos veces no rompe. En el path inyectado (`_device is None`)
        no se toca torch.
        """
        import gc

        self._model = None
        gc.collect()

        if self._device in ("cuda", "mps"):
            try:
                import torch

                if self._device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif self._device == "mps" and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:  # noqa: BLE001 - liberar es best-effort
                self.logger.warning("empty_cache falló al liberar el modelo", exc_info=True)

    @log_execution_time(logging.getLogger(__name__))
    def detect(
        self, frame: np.ndarray, frame_id: int = 0, imgsz: Optional[int] = None
    ) -> list[DetectedVehicle]:
        """Detect vehicles in a single frame, returning domain entities.

        `imgsz` (B1 1c): resolución de inferencia por-llamada. `None` = no se pasa
        a ultralytics → corre su nativo (no escribimos un literal de política acá).
        Cuando viene un valor (el scheduler lo aplica), se pasa a `predict`.
        """
        try:
            kwargs = {"verbose": False, "conf": self.conf_threshold}
            if imgsz is not None:
                kwargs["imgsz"] = imgsz
            results = self._model(frame, **kwargs)[0]

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
