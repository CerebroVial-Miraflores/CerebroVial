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
        text = f'Conteo Total: {count}'
        font_scale = 1
        font_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Obtener el tamaño del texto para posicionarlo correctamente
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Obtener el ancho del frame
        frame_h, frame_w, _ = frame.shape

        # Posicionar en la esquina superior derecha con un padding de 20px
        position = (frame_w - text_w - 20, text_h + 20)
        cv2.putText(frame, text, position, font, font_scale, self.text_color, font_thickness)

    def draw_detections(self, frame, results):
        # El método plot() de YOLO es una forma rápida de dibujar
        return results[0].plot()
    
    def draw_crossing_indicator(self, frame, box):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        center_x = int((x1 + x2) / 2)
        center_y = int(y2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)