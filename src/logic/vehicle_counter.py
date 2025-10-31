from collections import defaultdict

class VehicleCounter:
    def __init__(self, line_y: int):
        """
        Inicializa el contador de vehículos.
        :param line_y: La coordenada 'y' de la línea de conteo.
        """
        self.line_y = line_y
        self.track_history = defaultdict(list)
        self.counted_ids = set()
        self.vehicle_count = 0

    def update(self, track_id: int, box: tuple) -> bool:
        """
        Actualiza el estado de un vehículo y determina si debe ser contado.
        :param track_id: El ID del vehículo.
        :param box: La tupla (x1, y1, x2, y2) de la caja detectada.
        :return: True si el vehículo acaba de ser contado, False en caso contrario.
        """
        x1, y1, x2, y2 = box
        center_y = int(y2) # Usamos el borde inferior

        # Guardar historial de posiciones
        prev_y = self.track_history[track_id][-1] if self.track_history[track_id] else center_y
        self.track_history[track_id].append(center_y)

        # Lógica de cruce de línea (de arriba hacia abajo)
        if prev_y < self.line_y and center_y >= self.line_y and track_id not in self.counted_ids:
            self.vehicle_count += 1
            self.counted_ids.add(track_id)
            return True
        
        return False

    def get_count(self) -> int:
        return self.vehicle_count