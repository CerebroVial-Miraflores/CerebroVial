import time
from typing import List
import numpy as np
from ultralytics import YOLO
from ..domain import VehicleDetector, FrameAnalysis, DetectedVehicle

class YoloDetector(VehicleDetector):
    """
    Implementation of VehicleDetector using YOLO.
    """
    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # COCO class mapping: {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.target_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> FrameAnalysis:
        """
        Detect vehicles in a single frame.
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
        
        detected_vehicles: List[DetectedVehicle] = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                vehicle = DetectedVehicle(
                    id="", # ID is assigned by tracker, not detector
                    type=self.target_classes[cls_id],
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    timestamp=start_time
                )
                detected_vehicles.append(vehicle)
                
        return FrameAnalysis(
            frame_id=frame_id,
            timestamp=start_time,
            vehicles=detected_vehicles,
            total_count=len(detected_vehicles)
        )
