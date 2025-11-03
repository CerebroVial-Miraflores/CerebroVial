"""
Contiene la clase SpeedCalculator para estimar la velocidad de los vehículos.
"""

class SpeedCalculator:
    """
    Estima la velocidad de los objetos rastreados que cruzan dos líneas horizontales.
    Maneja el cálculo de velocidad para ambas direcciones ('up' y 'down').
    """
    def __init__(self, line_a_y: int, line_b_y: int, real_world_distance_m: float, fps: float, direction: str = 'down'):
        """
        Inicializa el calculador de velocidad.

        :param line_a_y: Coordenada 'y' de la línea de ENTRADA.
        :param line_b_y: Coordenada 'y' de la línea de SALIDA.
        :param real_world_distance_m: Distancia en el mundo real (en metros) entre las dos líneas.
        :param fps: Frames por segundo del video.
        :param direction: Dirección del tráfico a medir ('up' o 'down').
        """
        self.line_a_y = line_a_y # Línea de entrada
        self.line_b_y = line_b_y # Línea de salida
        self.real_world_distance_m = real_world_distance_m
        self.fps = fps
        self.direction = direction.lower()

        if self.direction not in ['down', 'up']:
            raise ValueError("La dirección debe ser 'down' o 'up'.")

        # Almacena el frame de entrada de un track_id: {track_id: frame_num}
        self.track_entry_frames = {}
        
        # Almacena la velocidad calculada para un track_id: {track_id: speed_kmh}
        self.speed_data = {}
        
        # Lista de velocidades calculadas en el intervalo actual
        self.all_speeds_list = []

    def update(self, track_id: int, box: list, current_frame_num: int) -> float:
        """
        Actualiza el estado de un vehículo y calcula su velocidad si cruza la zona.

        :param track_id: El ID de seguimiento del objeto.
        :param box: La caja delimitadora [x1, y1, x2, y2] del objeto.
        :param current_frame_num: El número de frame actual del video.
        :return: La velocidad calculada (km/h) si se completa la medición, None en caso contrario.
        """
        center_y = int(box[3]) # Usamos el borde inferior de la caja
        
        # --- LÓGICA DE CÁLCULO BASADA EN LA DIRECCIÓN ---
        
        entered_zone = False
        exited_zone = False

        if self.direction == 'down':
            # --- Dirección: ABAJO (y aumenta) ---
            # 1. Vehículo entra en la zona (cruza línea A)
            if track_id not in self.track_entry_frames and center_y >= self.line_a_y and center_y < self.line_b_y:
                self.track_entry_frames[track_id] = current_frame_num
                entered_zone = True
            
            # 2. Vehículo sale de la zona (cruza línea B)
            elif track_id in self.track_entry_frames and center_y >= self.line_b_y:
                exited_zone = True
        
        elif self.direction == 'up':
            # --- Dirección: ARRIBA (y disminuye) ---
            # 1. Vehículo entra en la zona (cruza línea A)
            if track_id not in self.track_entry_frames and center_y <= self.line_a_y and center_y > self.line_b_y:
                self.track_entry_frames[track_id] = current_frame_num
                entered_zone = True
            
            # 2. Vehículo sale de la zona (cruza línea B)
            elif track_id in self.track_entry_frames and center_y <= self.line_b_y:
                exited_zone = True

        # --- 3. Si acaba de salir de la zona, calcular velocidad ---
        if exited_zone:
            start_frame = self.track_entry_frames.pop(track_id)
            end_frame = current_frame_num
            
            frame_delta = end_frame - start_frame
            if frame_delta == 0: # Evitar división por cero
                return None

            time_sec = frame_delta / self.fps
            speed_ms = self.real_world_distance_m / time_sec
            speed_kmh = speed_ms * 3.6 # Convertir de m/s a km/h

            self.speed_data[track_id] = speed_kmh
            self.all_speeds_list.append(speed_kmh)
            
            return speed_kmh

        return None

    def get_speed(self, track_id: int) -> float:
        """
        Obtiene la última velocidad calculada para un track_id.
        """
        return self.speed_data.get(track_id, None)

    def get_average_speed(self) -> float:
        """
        Calcula la velocidad promedio de todos los vehículos medidos en el intervalo.
        """
        if not self.all_speeds_list:
            return 0.0
        
        return sum(self.all_speeds_list) / len(self.all_speeds_list)

    def reset_interval(self):
        """
        Reinicia la lista de velocidades para el próximo intervalo de 5 segundos.
        """
        self.all_speeds_list.clear()
        self.speed_data.clear()
        self.track_entry_frames.clear()