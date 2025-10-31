import cv2
from ultralytics import YOLO
import os

# --- Configuración Inicial ---

# Cargar el modelo YOLOv8 (usamos 'n' por ser rápido, puedes usar 's' o 'm' para más precisión)
# El modelo se descargará automáticamente si no lo tienes.
model = YOLO('yolov8n.pt') 

# Obtener la ruta del directorio donde se encuentra este script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definir la ruta al video de entrada
video_in_path = os.path.join(script_dir, '..', 'data', 'traffic_test.mp4')
# Definir la ruta al video de salida
video_out_path = os.path.join(script_dir, '..', 'data', 'traffic_test_output.mp4')

# IDs de las clases del dataset COCO que queremos detectar y contar
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# --- Procesamiento del Video ---

# Abrir el archivo de video
cap = cv2.VideoCapture(video_in_path)
if not cap.isOpened():
    print(f"Error: No se pudo abrir el video en {video_in_path}")
    exit()

# Obtener propiedades del video (ancho, alto, fps)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Definir el codec y crear el objeto VideoWriter para guardar el video de salida
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec para .mp4
out = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))

# --- Lógica de Conteo ---

# Definir la línea de conteo (horizontal, en el medio de la pantalla)
# Puedes ajustar este valor (0.5) para subir o bajar la línea
LINE_Y = int(h * 0.5) 

# Diccionario para guardar la última posición 'y' de cada vehículo rastreado
track_history = {}
# Set para guardar los IDs de los vehículos que ya han sido contados
counted_ids = set()
# Contador total
vehicle_count = 0

print(f"Procesando video... Presiona 'q' en la ventana del video para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Fin del video o error al leer frame.")
        break

    # Ejecutar el rastreo (tracking) de YOLOv8 en el frame
    # persist=True: Mantiene los IDs de rastreo entre frames
    # classes=VEHICLE_CLASSES: Filtra para detectar solo las clases que nos interesan
    results = model.track(frame, persist=True, classes=VEHICLE_CLASSES)

    # Obtener las cajas (bounding boxes) y los IDs de rastreo
    # Asegurarnos de que 'boxes' no esté vacío
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()

        # Dibujar las detecciones en el frame
        annotated_frame = results[0].plot()

        # Dibujar la línea de conteo en el frame
        cv2.line(annotated_frame, (0, LINE_Y), (w, LINE_Y), (0, 255, 0), 2) # Línea verde

        # Iterar sobre las detecciones
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            
            # Calcular el punto central inferior de la caja (más estable para contar)
            center_x = int((x1 + x2) / 2)
            center_y = int(y2) # Usamos el borde inferior

            # --- Lógica de cruce de línea ---
            
            # 1. Verificar si el vehículo es nuevo en el historial
            if track_id not in track_history:
                track_history[track_id] = center_y
                continue # Saltar a la siguiente iteración

            # 2. Obtener la posición 'y' anterior
            prev_y = track_history[track_id]
            
            # 3. Verificar si ha cruzado la línea (en este caso, de arriba hacia abajo)
            # y si AÚN NO ha sido contado
            if prev_y < LINE_Y and center_y >= LINE_Y and track_id not in counted_ids:
                vehicle_count += 1
                counted_ids.add(track_id)
                
                # Opcional: Dibujar un círculo rojo cuando cruza
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # 4. Actualizar la historial de posición del vehículo
            track_history[track_id] = center_y
            
    else:
        # Si no hay detecciones, 'annotated_frame' es solo el frame original
        annotated_frame = frame.copy()
        # Dibujar la línea de conteo de todas formas
        cv2.line(annotated_frame, (0, LINE_Y), (w, LINE_Y), (0, 255, 0), 2)


    # Mostrar el contador en el video
    cv2.putText(
        annotated_frame, 
        f'Conteo Total: {vehicle_count}', 
        (50, 50), # Posición del texto
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, # Tamaño de fuente
        (255, 255, 255), # Color (blanco)
        2 # Grosor
    )

    # Guardar el frame procesado en el video de salida
    out.write(annotated_frame)

    # Mostrar el video en una ventana (opcional, pero útil para depurar)
    cv2.imshow("CerebroVial - Conteo de Vehiculos", annotated_frame)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Limpieza ---
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Procesamiento finalizado. Video guardado en {video_out_path}")
print(f"Conteo final: {vehicle_count} vehículos.")