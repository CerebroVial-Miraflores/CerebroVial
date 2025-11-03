import cv2
import threading
import time 
import yt_dlp
from flask import Flask, Response, render_template

from components.object_detector import ObjectDetector
from components.object_tracker import ObjectTracker
from logic.vehicle_counter import VehicleCounter
from logic.speed_calculator import SpeedCalculator 
from utils.visualizer import Visualizer

# --- CONFIGURACIÓN ---
VIDEO_SOURCE = "https://youtu.be/qMYlpMsWsBE"
MODEL_PATH = 'yolov8n.pt'
VEHICLE_CLASSES = [2, 3, 5, 7] # car, motorcycle, bus, truck

# --- 1. LÍNEAS MOVIDAS AL "SUELO" ---
# Los autos se mueven de abajo (cerca) hacia arriba (lejos) -> direction='up'

# Línea de conteo (Verde)
COUNT_LINE_Y_RATIO = 0.85 # 85% hacia abajo

# Zona de medición de velocidad (Azules)
SPEED_LINE_ENTRY_RATIO = 0.80 # Línea de ENTRADA (más cercana)
SPEED_LINE_EXIT_RATIO = 0.50  # Línea de SALIDA (más alejada)

# ¡¡IMPORTANTE!! Mide esta distancia en el mundo real.
REAL_WORLD_DISTANCE_M = 15.0 # Metros (Estimación)

# Intervalo de actualización de la velocidad promedio
AVG_SPEED_UPDATE_INTERVAL_SEC = 3 # Cada 3 segundos

# --- VARIABLES GLOBALES PARA EL STREAMING ---
output_frame = None
lock = threading.Lock()

app = Flask(_name_)

def get_video_stream_url(url):
    """Usa yt-dlp para obtener la URL directa del stream de mejor calidad."""
    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]/best[height<=720]',
        'quiet': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get('url')
    except Exception as e:
        print(f"Error al obtener el stream con yt-dlp: {e}")
        return None

def process_video_stream():
    """
    Función que se ejecuta en un hilo separado para procesar el video.
    """
    global output_frame, lock

    stream_url = get_video_stream_url(VIDEO_SOURCE)
    if not stream_url:
        print("No se pudo obtener la URL del stream. Saliendo...")
        return

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: OpenCV no pudo abrir el stream de video desde {stream_url}")
        return

    # --- INICIALIZACIÓN DE COMPONENTES ---
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Advertencia: FPS es 0. Usando 25.0 como valor por defecto.")
        fps = 25.0 

    # 2. CALCULAR COORDENADAS 'Y' DE LAS LÍNEAS
    count_line_y = int(h * COUNT_LINE_Y_RATIO)
    speed_line_a_y = int(h * SPEED_LINE_ENTRY_RATIO) # Entrada
    speed_line_b_y = int(h * SPEED_LINE_EXIT_RATIO)  # Salida

    # Instanciar todos los componentes
    detector = ObjectDetector(model_path=MODEL_PATH)
    tracker = ObjectTracker(detector)
    
    # 3. PASAR LA DIRECCIÓN CORRECTA A AMBOS
    # La dirección 'up' es crucial para que la lógica funcione
    counter = VehicleCounter(line_y=count_line_y, direction='up')
    
    speed_calc = SpeedCalculator(
        line_a_y=speed_line_a_y, 
        line_b_y=speed_line_b_y, 
        real_world_distance_m=REAL_WORLD_DISTANCE_M, 
        fps=fps,
        direction='up' # <-- ¡AÑADIDO!
    )
    
    visualizer = Visualizer(line_y=count_line_y)
    # 4. PASAR AMBAS LÍNEAS AL VISUALIZER
    visualizer.set_speed_measurement_lines(speed_line_a_y, speed_line_b_y)

    print("Hilo de procesamiento iniciado. Procesando frames...")
    
    frame_counter = 0
    current_avg_speed = 0.0
    frames_in_interval = int(fps * AVG_SPEED_UPDATE_INTERVAL_SEC)

    while True:
        success, frame = cap.read()
        if not success:
            print("Stream finalizado o error al leer frame.")
            break
        
        frame_counter += 1

        results = tracker.track_objects(frame, classes_to_track=VEHICLE_CLASSES)
        annotated_frame = results[0].plot() 

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                # Actualizar contador
                if counter.update(track_id, box):
                    visualizer.draw_crossing_indicator(annotated_frame, box)
                
                # Actualizar calculador de velocidad
                speed_calc.update(track_id, box, frame_counter)
                
                speed = speed_calc.get_speed(track_id)
                if speed is not None:
                    visualizer.draw_speed_on_box(annotated_frame, box, speed)
        
        if frame_counter % frames_in_interval == 0:
            current_avg_speed = speed_calc.get_average_speed()
            speed_calc.reset_interval() 
            print(f"--- ACTUALIZACION ({AVG_SPEED_UPDATE_INTERVAL_SEC}s) --- Vel. Promedio: {current_avg_speed:.2f} km/h")

        
        visualizer.draw_lines(annotated_frame) 
        visualizer.draw_stats(annotated_frame, counter.get_count(), current_avg_speed)

        with lock:
            output_frame = annotated_frame.copy()
    
    cap.release()

def generate_frames():
    """Generador que sirve el stream 'multipart/x-mixed-replace'."""
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')

@app.route("/")
def index():
    """Página principal que muestra el video."""
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """Ruta que sirve el stream de video."""
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if _name_ == '_main_':
    processing_thread = threading.Thread(target=process_video_stream)
    processing_thread.daemon = True
    processing_thread.start()
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True, use_reloader=False)