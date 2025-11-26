import cv2
import numpy as np
from typing import List, Tuple

class ZoneSelector:
    """
    Handles interactive selection of ROI polygons.
    """
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.points = []
        self.done = False

    def select_zone(self, frame: np.ndarray) -> List[List[int]]:
        """
        Opens an interactive session to select points on the frame.
        """
        self.points = []
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
            if len(self.points) > 0:
                pts = np.array(self.points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(temp_frame, [pts], False, (0, 255, 255), 2)
                
                for pt in self.points:
                    cv2.circle(temp_frame, tuple(pt), 4, (0, 0, 255), -1)
            
            cv2.imshow(self.window_name, temp_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13: # Enter
                if len(self.points) >= 3:
                    self.done = True
                else:
                    print("Need at least 3 points for a polygon.")
            elif key == 27: # Esc
                self.points = []
                self.done = True
                
        # Cleanup callback
        cv2.setMouseCallback(self.window_name, lambda *args: None)
        return self.points

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
