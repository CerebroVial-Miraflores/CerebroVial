import cv2
import numpy as np
from typing import List, Tuple
from ...domain.entities import Frame, FrameAnalysis, DetectedVehicle, ZoneVehicleCount

class OpenCVVisualizer:
    """
    Handles visualization of detection results using OpenCV.
    """
    def __init__(self, zones_config: dict = None):
        self.zones_config = zones_config
        
    def draw(self, frame: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
        """
        Draws bounding boxes, labels, and zones on the frame.
        """
        # Draw zones if present in analysis
        if analysis and analysis.zones and self.zones_config:
            for zone_status in analysis.zones:
                zone_id = zone_status.zone_id
                points = self.zones_config.get(zone_id)
                if points:
                    pts = np.array(points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    # Draw polygon
                    cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
                    # Draw count
                    # Find center for text
                    M = cv2.moments(pts)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(frame, f"{zone_id}: {zone_status.vehicle_count}", (cX - 20, cY), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if not analysis:
            return frame
            
        for vehicle in analysis.vehicles:
            x1, y1, x2, y2 = vehicle.bbox
            label = f"{vehicle.type} {vehicle.id}"
            if vehicle.speed is not None:
                label += f" {vehicle.speed:.0f}km/h"
            else:
                label += f" {vehicle.confidence:.2f}"
            
            # Color based on type
            color = (0, 255, 0) # Green default
            if vehicle.type == 'car': color = (0, 255, 0)
            elif vehicle.type == 'bus': color = (0, 165, 255) # Orange
            elif vehicle.type == 'truck': color = (0, 0, 255) # Red
            elif vehicle.type == 'motorcycle': color = (255, 255, 0) # Cyan
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw total count
        cv2.putText(frame, f"Vehicles: {analysis.total_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return frame
