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
        device: str | None = None,
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

            if device is not None:
                # Device resuelto en el arranque del server (`select_device()`):
                # el banner ya se imprimió al boot, acá no se re-detecta ni se
                # vuelve a anunciar.
                pass
            elif torch.cuda.is_available():
                # Fallback para callers que no inyectan device (path legacy de
                # `src/main.py`, construcción directa): misma política
                # cuda→mps→cpu, sin banner. La fuente de verdad del banner es
                # `select_device()` en el arranque.
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
            result = self._model(frame, **kwargs)[0]
            return self._parse_result(result, frame_id)
        except Exception as e:
            self.logger.error(f"Detection failed on frame {frame_id}: {e}")
            raise DetectionError(f"YOLO inference failed: {e}") from e

    @log_execution_time(logging.getLogger(__name__))
    def detect_batch(
        self,
        frames: list[np.ndarray],
        frame_ids: Optional[list[int]] = None,
        imgsz: Optional[int] = None,
    ) -> list[list[DetectedVehicle]]:
        """Detect vehicles en un LOTE de frames → una lista de detecciones por frame.

        Topología B (15Hz): el batch worker central junta frames de varias cámaras y
        los infiere de una sola pasada en GPU. ultralytics acepta `model([f1, f2, ...])`
        y devuelve un result por frame; cada uno se parsea con la MISMA lógica que
        `detect` (misma fuente de verdad: `_parse_result`), así que
        `detect_batch([f])[0] == detect(f)` salvo el `timestamp` (wall-clock).

        `frame_ids` (opcional): ids por frame para los ids temporales de detección
        (el tracker asigna el id estable downstream). Si falta, se usa el índice.
        """
        if not frames:
            return []
        if frame_ids is not None and len(frame_ids) != len(frames):
            raise ValueError("frame_ids debe tener el mismo largo que frames")
        try:
            kwargs = {"verbose": False, "conf": self.conf_threshold}
            if imgsz is not None:
                kwargs["imgsz"] = imgsz
            results = self._model(frames, **kwargs)
            ids = frame_ids if frame_ids is not None else list(range(len(frames)))
            return [
                self._parse_result(result, fid)
                for result, fid in zip(results, ids)
            ]
        except Exception as e:
            self.logger.error(f"Batch detection failed ({len(frames)} frames): {e}")
            raise DetectionError(f"YOLO batch inference failed: {e}") from e

    def _parse_result(self, result, frame_id: int) -> list[DetectedVehicle]:
        """Mapea un result de ultralytics → entidades de dominio (filtro clase + conf).

        Fuente de verdad ÚNICA del parseo, compartida por `detect` (1 frame) y
        `detect_batch` (lote). El id es temporal (`{frame_id}_{i}`); el tracker
        asigna el id estable.
        """
        vehicles: list[DetectedVehicle] = []
        for box in result.boxes:
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
