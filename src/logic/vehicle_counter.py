# /Users/rasec/Documents/github/Proyecto de Tesis/CerebroVial/src/logic/vehicle_counter.py

class VehicleCounter:
    """
    Gestiona la lógica de conteo de vehículos que cruzan una línea horizontal.
    """
    def __init__(self, line_y: int, direction: str = 'down'):
        """
        Inicializa el contador de vehículos.

        :param line_y: La coordenada 'y' de la línea de conteo.
        :param direction: La dirección del tráfico a contar ('down' o 'up').
        """
        self.line_y = line_y
        self.direction = direction.lower()
        self.track_history = {}
        self.counted_ids = set()
        self.vehicle_count = 0

        if self.direction not in ['down', 'up']:
            raise ValueError("La dirección debe ser 'down' o 'up'.")

    def update(self, track_id: int, box: list) -> bool:
        """
        Actualiza el estado de un vehículo rastreado y determina si debe contarse.

        :param track_id: El ID de seguimiento del objeto.
        :param box: La caja delimitadora [x1, y1, x2, y2] del objeto.
        :return: True si el vehículo acaba de ser contado, False en caso contrario.
        """
        # Usamos el borde inferior de la caja como punto de referencia
        center_y = int(box[3])

        # Si es un vehículo nuevo, registrar su posición y continuar
        # También verificamos si ya fue contado para evitar reprocesamiento innecesario.
        if track_id not in self.track_history:
            self.track_history[track_id] = center_y
            # Si el vehículo aparece por primera vez ya habiendo cruzado la línea,
            # lo añadimos a los contados para no registrarlo si retrocede y vuelve a cruzar.
            if (self.direction == 'down' and center_y > self.line_y) or \
               (self.direction == 'up' and center_y < self.line_y):
                self.counted_ids.add(track_id)
            return False

        # Si el vehículo ya fue contado, solo actualizamos su posición y salimos.
        if track_id in self.counted_ids:
            self.track_history[track_id] = center_y
            return False

        prev_y = self.track_history[track_id]
        
        crossed = False
        if self.direction == 'down':
            # Cruce de arriba hacia abajo
            if prev_y < self.line_y and center_y >= self.line_y:
                crossed = True
        elif self.direction == 'up':
            # Cruce de abajo hacia arriba
            if prev_y > self.line_y and center_y <= self.line_y:
                crossed = True

        # Actualizar el historial de posición
        self.track_history[track_id] = center_y

        if crossed:
            self.vehicle_count += 1
            self.counted_ids.add(track_id)
            return True

        return False

    def get_count(self) -> int:
        """
        Devuelve el conteo total de vehículos.
        """
        return self.vehicle_count
