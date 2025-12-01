
import logging
import cv2
import time
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple
from ...domain.entities import FrameAnalysis, DetectedVehicle
from ...domain.protocols import VehicleDetector
from ....common.logging import setup_logger, log_execution_time
from ....common.exceptions import DetectionError

class YoloDetector(VehicleDetector):
    """
    Implementation of VehicleDetector using YOLO.
    """
    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.5):
        # Dynamic device selection
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
            
        print(f"[INFO] Using inference device: {device}")
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
        self.target_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.logger = setup_logger(__name__)

    @log_execution_time(logging.getLogger(__name__))
    def detect(self, frame: np.ndarray, frame_id: int = 0) -> FrameAnalysis:
        """
        Detects vehicles in the given frame.
        """
        try:
            # Run inference
            results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
            
            vehicles = []
            
            # Process detections
            for box in results.boxes:
                class_id = int(box.cls[0])
                
                if class_id in self.target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    vehicle = DetectedVehicle(
                        id=f"{frame_id}_{len(vehicles)}", # Temporary ID, tracking will assign real ID
                        type=self.target_classes[class_id],
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        timestamp=time.time()
                    )
                    vehicles.append(vehicle)
            
            # Debug: Print raw detection count
            # print(f"[DEBUG] Frame {frame_id}: Raw detections: {len(vehicles)}")
            
            return FrameAnalysis(
                frame_id=frame_id,
                timestamp=time.time(),
                vehicles=vehicles,
                total_count=len(vehicles)
            )
        except Exception as e:
            self.logger.error(f"Detection failed on frame {frame_id}: {e}")
            raise DetectionError(f"YOLO inference failed: {e}") from e

