import cv2
import numpy as np
from collections import defaultdict, deque

class Visualizer:
    def __init__(self, line_y: int, text_color=(255, 255, 255), line_color=(0, 255, 0), 
                 show_trajectories=True, trajectory_length=30, speed_line_color=(255, 0, 0)):
        self.line_y = line_y
        self.text_color = text_color
        self.line_color = line_color
        self.show_trajectories = show_trajectories
        self.trajectory_length = trajectory_length

        # Propiedades para las líneas de velocidad
        self.speed_line_top_y = None
        self.speed_line_bottom_y = None
        self.speed_line_color = speed_line_color
        
        # Almacenar trayectorias de cada vehículo
        # {track_id: deque([(x, y), (x, y), ...])}
        self.trajectories = defaultdict(lambda: deque(maxlen=trajectory_length))
        
        # Colores por tipo de vehículo
        self.class_colors = {
            2: (255, 100, 100),  # Autos - Azul claro
            3: (100, 255, 100),  # Motos - Verde claro
            5: (100, 100, 255),  # Buses - Rojo claro
            7: (255, 255, 100)   # Camiones - Cyan
        }

    def set_speed_measurement_lines(self, line_top_y: int, line_bottom_y: int):
        """
        Establece las coordenadas para las líneas de medición de velocidad.
        """
        self.speed_line_top_y = line_top_y
        self.speed_line_bottom_y = line_bottom_y

    def draw_lines(self, frame):
        """Dibuja todas las líneas configuradas (conteo y velocidad)."""
        h, w, _ = frame.shape
        # Dibujar línea de conteo (Verde)
        cv2.line(frame, (0, self.line_y), (w, self.line_y), self.line_color, 3)

        # Dibujar líneas de velocidad (Azul/Rojo por defecto)
        if self.speed_line_top_y is not None:
            cv2.line(frame, (0, self.speed_line_top_y), (w, self.speed_line_top_y), self.speed_line_color, 2)
        if self.speed_line_bottom_y is not None:
            cv2.line(frame, (0, self.speed_line_bottom_y), (w, self.speed_line_bottom_y), self.speed_line_color, 2)

    def draw_count(self, frame, count):
        """
        Dibuja el conteo total en la esquina superior derecha.
        
        :param frame: Frame donde dibujar
        :param count: Conteo total
        """
        text = f'Total: {count}'
        font_scale = 1
        font_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        frame_h, frame_w, _ = frame.shape

        position = (frame_w - text_w - 20, text_h + 20)
        cv2.putText(frame, text, position, font, font_scale, self.text_color, font_thickness)
    
    def draw_counts_breakdown(self, frame, counts_summary):
        """
        Dibuja el desglose de conteo por tipo de vehículo.
        
        :param frame: Frame donde dibujar
        :param counts_summary: Diccionario {'Autos': count, 'Motos': count, ...}
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        line_height = 35
        padding = 20
        
        # Posición inicial (esquina superior izquierda)
        x = padding
        y = padding + 25
        
        # Fondo semitransparente para mejor legibilidad
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 30), (x + 250, y + len(counts_summary) * line_height + 10), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Dibujar cada tipo de vehículo
        for i, (vehicle_type, count) in enumerate(counts_summary.items()):
            text = f'{vehicle_type}: {count}'
            position = (x, y + i * line_height)
            cv2.putText(frame, text, position, font, font_scale, self.text_color, font_thickness)

    def draw_speed_stats(self, frame, avg_speed: float):
        """
        Dibuja la velocidad promedio en la esquina superior derecha, debajo del conteo.
        """
        text = f'Vel. Promedio: {avg_speed:.2f} km/h'
        font_scale = 0.8
        font_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        frame_h, frame_w, _ = frame.shape

        # Posición debajo del conteo total
        position = (frame_w - text_w - 20, (text_h + 20) * 2)
        cv2.putText(frame, text, position, font, font_scale, self.text_color, font_thickness)

    def draw_speed_on_box(self, frame, box, speed_kmh: float):
        """
        Dibuja la velocidad individual sobre la caja del vehículo.
        """
        x1, y1, _, _ = [int(coord) for coord in box]
        text = f"{speed_kmh:.1f} km/h"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def draw_detections(self, frame, results):
        """
        Dibuja las detecciones usando el método plot() de YOLO.
        
        :param frame: Frame original
        :param results: Resultados de YOLO
        :return: Frame anotado
        """
        return results[0].plot()
    
    def draw_crossing_indicator(self, frame, box):
        """
        Dibuja un indicador visual cuando un vehículo cruza la línea.
        
        :param frame: Frame donde dibujar
        :param box: Caja delimitadora [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = [int(coord) for coord in box]
        center_x = int((x1 + x2) / 2)
        center_y = int(y2)
        
        # Círculo rojo brillante
        cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)
        # Borde blanco para mejor visibilidad
        cv2.circle(frame, (center_x, center_y), 8, (255, 255, 255), 2)

    def update_trajectories(self, results):
        """
        Actualiza las trayectorias con las nuevas posiciones de los vehículos.
        
        :param results: Resultados del tracker de YOLO.
        """
        if not self.show_trajectories or results[0].boxes is None or results[0].boxes.id is None:
            return

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            self.trajectories[int(track_id)].append((center_x, center_y))

    def draw_trajectories(self, frame, results):
        """
        Dibuja las trayectorias de los vehículos en el frame.
        
        :param frame: Frame donde dibujar.
        :param results: Resultados del tracker de YOLO para obtener el color por clase.
        """
        if not self.show_trajectories:
            return

        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            id_to_class = {int(tid): int(cid) for tid, cid in zip(track_ids, class_ids)}

            for track_id, path in self.trajectories.items():
                if len(path) > 1:
                    class_id = id_to_class.get(track_id)
                    color = self.class_colors.get(class_id, (200, 200, 200)) # Gris por defecto
                    
                    points = np.array(path, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

    def clear_old_trajectories(self, active_track_ids: set):
        """Elimina trayectorias de IDs que ya no están activos."""
        inactive_ids = set(self.trajectories.keys()) - active_track_ids
        for inactive_id in inactive_ids:
            del self.trajectories[inactive_id]