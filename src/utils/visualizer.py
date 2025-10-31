import cv2

class Visualizer:
    def __init__(self, line_y: int, text_color=(255, 255, 255), line_color=(0, 255, 0)):
        self.line_y = line_y
        self.text_color = text_color
        self.line_color = line_color

    def draw_line(self, frame):
        h, w, _ = frame.shape
        cv2.line(frame, (0, self.line_y), (w, self.line_y), self.line_color, 2)

    def draw_count(self, frame, count):
        cv2.putText(frame, f'Conteo Total: {count}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 2)

    def draw_detections(self, frame, results):
        # El método plot() de YOLO es una forma rápida de dibujar
        return results[0].plot()
    
    def draw_crossing_indicator(self, frame, box):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        center_x = int((x1 + x2) / 2)
        center_y = int(y2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)