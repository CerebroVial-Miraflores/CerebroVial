import cv2
from ultralytics import YOLO
import os
import csv

# --- Configuración Inicial ---

# Cargar el modelo YOLOv8
model = YOLO('yolov8n.pt') 

# Obtener la ruta del directorio donde se encuentra este script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definir la ruta al video de entrada
video_in_path = os.path.join(script_dir, '..', 'data', 'traffic_test.mp4')
# Definir la ruta al video de salida (opcional, pero útil para verificar)
video_out_path = os.path.join(script_dir, '..', 'data', 'traffic_test_timeseries_output.mp4')
# Definir la ruta al archivo CSV de salida
csv_out_path = os.path.join(script_dir, '..', 'data', 'vehicle_counts_timeseries.csv')

# IDs de las clases del dataset COCO que queremos contar
VEHICLE_CLASSES = [2, 3, 5, 7] # car, motorcycle, bus, truck

# --- Configuración de la Serie Temporal ---
# Definir el intervalo de tiempo en segundos para agrupar los conteos
INTERVAL_SEC = 10 

# --- Procesamiento del Video ---

cap = cv2.VideoCapture(video_in_path)
if not cap.isOpened():
    print(f"Error: No se pudo abrir el video en {video_in_path}")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Calcular cuántos frames hay en cada intervalo
frames_per_interval = int(fps * INTERVAL_SEC)
print(f"Info del Video: {w}x{h} @ {fps:.2f} FPS. Contando en intervalos de {INTERVAL_SEC} seg ({frames_per_interval} frames).")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))

# --- Lógica de Conteo ---

LINE_Y = int(h * 0.5) 
track_history = {}
counted_ids = set()
total_vehicle_count = 0

# --- Nuevas variables para la serie temporal ---
time_series_data = [] # Lista para guardar los datos (timestamp, conteo)
current_frame_num = 0
interval_count = 0
current_interval_num = 1

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Fin del video.")
        break
    
    # Incrementar el contador de frames
    current_frame_num += 1

    results = model.track(frame, persist=True, classes=VEHICLE_CLASSES)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        annotated_frame = results[0].plot()
        cv2.line(annotated_frame, (0, LINE_Y), (w, LINE_Y), (0, 255, 0), 2)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            center_y = int(y2) 

            if track_id not in track_history:
                track_history[track_id] = center_y
                continue 

            prev_y = track_history[track_id]
            
            # Si cruza la línea Y NO ha sido contado en este intervalo
            if prev_y < LINE_Y and center_y >= LINE_Y and track_id not in counted_ids:
                interval_count += 1
                total_vehicle_count += 1
                counted_ids.add(track_id) # Añadir al set para evitar doble conteo
                cv2.circle(annotated_frame, (int((x1+x2)/2), center_y), 5, (0, 0, 255), -1)

            track_history[track_id] = center_y
            
    else:
        annotated_frame = frame.copy()
        cv2.line(annotated_frame, (0, LINE_Y), (w, LINE_Y), (0, 255, 0), 2)

    # --- Lógica para guardar el intervalo ---
    
    # Comprobar si hemos completado un intervalo
    if current_frame_num == frames_per_interval * current_interval_num:
        # Calcular el timestamp (en segundos)
        timestamp_sec = current_interval_num * INTERVAL_SEC
        
        # Guardar los datos del intervalo
        time_series_data.append({
            'timestamp_sec': timestamp_sec,
            'vehicle_count': interval_count
        })
        
        print(f"Intervalo {current_interval_num} ({timestamp_sec}s): {interval_count} vehículos.")
        
        # Resetear para el siguiente intervalo
        interval_count = 0
        current_interval_num += 1
        counted_ids.clear() # Limpiar los IDs contados para el nuevo intervalo

    # Mostrar contadores en el video
    cv2.putText(annotated_frame, f'Conteo Total: {total_vehicle_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f'Intervalo Actual ({current_interval_num}): {interval_count}', (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out.write(annotated_frame)
    cv2.imshow("CerebroVial - Serie Temporal", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Limpieza y Guardado Final ---
cap.release()
out.release()
cv2.destroyAllWindows()

# Guardar los datos de la serie temporal en un archivo CSV
try:
    with open(csv_out_path, 'w', newline='') as csvfile:
        fieldnames = ['timestamp_sec', 'vehicle_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(time_series_data)
        
    print(f"\nProcesamiento finalizado. Video de salida guardado en {video_out_path}")
    print(f"¡Éxito! Datos de serie temporal guardados en {csv_out_path}")
    print(f"Conteo final total: {total_vehicle_count} vehículos.")

except Exception as e:
    print(f"\nError al guardar el archivo CSV: {e}")