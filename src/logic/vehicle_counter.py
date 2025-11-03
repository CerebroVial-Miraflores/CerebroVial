class VehicleCounter:
    """
    Gestiona la lógica de conteo de vehículos que cruzan una línea horizontal.
    Cuenta por tipo de vehículo (car, motorcycle, bus, truck).
    """
    
    # Mapeo de IDs de COCO a nombres legibles
    CLASS_NAMES = {
        2: 'Autos',
        3: 'Motos',
        5: 'Buses',
        7: 'Camiones'
    }
    
    def __init__(self, line_y: int, direction: str = 'down'):
        """
        Inicializa el contador de vehículos.

        :param line_y: La coordenada 'y' de la línea de conteo.
        :param direction: La dirección del tráfico a contar ('down' o 'up').
        """
        self.line_y = line_y
        self.direction = direction.lower()
        self.track_history = {}  # {track_id: {'y': pos_y, 'class': class_id}}
        self.counted_ids = set()
        
        # Conteo separado por tipo de vehículo
        self.counts_by_class = {
            2: 0,  # cars
            3: 0,  # motorcycles
            5: 0,  # buses
            7: 0   # trucks
        }
        
        if self.direction not in ['down', 'up']:
            raise ValueError("La dirección debe ser 'down' o 'up'.")

    def update(self, track_id: int, box: list, class_id: int) -> bool:
        """
        Actualiza el estado de un vehículo rastreado y determina si debe contarse.

        :param track_id: El ID de seguimiento del objeto.
        :param box: La caja delimitadora [x1, y1, x2, y2] del objeto.
        :param class_id: El ID de clase del vehículo (2=car, 3=motorcycle, 5=bus, 7=truck).
        :return: True si el vehículo acaba de ser contado, False en caso contrario.
        """
        # Usamos el borde inferior de la caja como punto de referencia
        center_y = int(box[3])

        # Si el vehículo ya fue contado, no hacer nada más
        if track_id in self.counted_ids:
            return False

        # Si es un vehículo nuevo, registrar su posición y clase
        if track_id not in self.track_history:
            self.track_history[track_id] = {
                'y': center_y,
                'class': class_id
            }
            return False

        prev_y = self.track_history[track_id]['y']
        
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
        self.track_history[track_id]['y'] = center_y

        if crossed:
            # Incrementar contador específico de la clase
            if class_id in self.counts_by_class:
                self.counts_by_class[class_id] += 1
            self.counted_ids.add(track_id)
            return True

        return False

    def get_count(self) -> int:
        """
        Devuelve el conteo total de vehículos (suma de todas las clases).
        """
        return sum(self.counts_by_class.values())
    
    def get_counts_by_class(self) -> dict:
        """
        Devuelve el conteo separado por tipo de vehículo.
        
        :return: Diccionario {class_id: count}
        """
        return self.counts_by_class.copy()
    
    def get_counts_summary(self) -> dict:
        """
        Devuelve un resumen legible del conteo por tipo.
        
        :return: Diccionario {'Autos': count, 'Motos': count, ...}
        """
        return {
            self.CLASS_NAMES[class_id]: count 
            for class_id, count in self.counts_by_class.items()
        }
    
    def reset(self):
        """
        Reinicia todos los contadores y el historial.
        Útil para análisis por intervalos de tiempo.
        """
        self.track_history.clear()
        self.counted_ids.clear()
        self.counts_by_class = {k: 0 for k in self.counts_by_class}