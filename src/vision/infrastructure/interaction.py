import cv2
import numpy as np
from typing import List, Tuple

class PointCollector:
    """
    Handles the logic of collecting points for a polygon.
    """
    def __init__(self):
        self.points: List[List[int]] = []

    def add_point(self, x: int, y: int):
        self.points.append([x, y])

    def remove_last_point(self):
        if self.points:
            self.points.pop()

    def clear(self):
        self.points = []

    def get_points(self) -> List[List[int]]:
        return self.points

    def is_valid_polygon(self) -> bool:
        return len(self.points) >= 3


class InteractiveZoneSelector:
    """
    Handles the UI interaction for selecting a zone.
    """
    def __init__(self, window_name: str, collector: PointCollector = None):
        self.window_name = window_name
        self.collector = collector or PointCollector()
        self.done = False

    def select_zone(self, frame: np.ndarray) -> List[List[int]]:
        """
        Opens an interactive session to select points on the frame.
        """
        self.collector.clear()
        self.done = False
        
        # Clone frame to draw on without modifying original immediately
        display_frame = frame.copy()
        
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n--- ROI Selection Mode ---")
        print("Left Click: Add point")
        print("Right Click: Remove last point")
        print("Enter: Finish selection")
        print("Esc: Cancel")
        
        while not self.done:
            # Draw current polygon
            temp_frame = display_frame.copy()
            points = self.collector.get_points()
            
            if len(points) > 0:
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(temp_frame, [pts], False, (0, 255, 255), 2)
                
                for pt in points:
                    cv2.circle(temp_frame, tuple(pt), 4, (0, 0, 255), -1)
            
            cv2.imshow(self.window_name, temp_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13: # Enter
                if self.collector.is_valid_polygon():
                    self.done = True
                else:
                    print("Need at least 3 points for a polygon.")
            elif key == 27: # Esc
                self.collector.clear()
                self.done = True
                
        # Cleanup callback
        cv2.setMouseCallback(self.window_name, lambda *args: None)
        return self.collector.get_points()

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.collector.add_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.collector.remove_last_point()

# Legacy alias for backward compatibility if needed, but better to update usage
class ZoneSelector(InteractiveZoneSelector):
    def __init__(self, window_name: str):
        super().__init__(window_name)
