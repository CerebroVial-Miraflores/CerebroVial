import cv2
import threading
import yt_dlp
from flask import Flask, Response, render_template

from components.object_detector import ObjectDetector
from components.object_tracker import ObjectTracker
from logic.vehicle_counter import VehicleCounter
from utils.visualizer import Visualizer

# --- CONFIGURACIÓN ---
VIDEO_SOURCE = "https://youtu.be/qMYlpMsWsBE"
MODEL_PATH = 'yolov8n.pt'
VEHICLE_CLASSES = [2, 3, 5, 7] # car, motorcycle, bus, truck
LINE_Y_RATIO = 0.75

# --- VARIABLES GLOBALES PARA EL STREAMING ---
# Frame de salida que se servirá a través de Flask
output_frame = None
# Lock para asegurar que el frame no se lea y escriba al mismo tiempo
lock = threading.Lock()

# Inicializar la aplicación Flask
app = Flask(__name__)

def get_video_stream_url(url):
    """Usa yt-dlp para obtener la URL directa del stream de mejor calidad."""
    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]/best[height<=720]', # Prioriza mp4 hasta 720p
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
    Lee frames, los procesa y actualiza la variable global 'output_frame'.
    """
    global output_frame, lock

    stream_url = get_video_stream_url(VIDEO_SOURCE)
    if not stream_url:
        print("No se pudo obtener la URL del stream. Saliendo del hilo de procesamiento.")
        return

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: OpenCV no pudo abrir el stream de video desde {stream_url}")
        return

    # Inicialización de componentes
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(h * LINE_Y_RATIO)

    detector = ObjectDetector(model_path=MODEL_PATH)
    tracker = ObjectTracker(detector)
    counter = VehicleCounter(line_y=line_y, direction='up')
    visualizer = Visualizer(line_y=line_y)

    print("Hilo de procesamiento iniciado. Procesando frames...")

    while True:
        success, frame = cap.read()
        if not success:
            print("Stream finalizado o error al leer frame.")
            break

        # Procesamiento del frame
        results = tracker.track_objects(frame, classes_to_track=VEHICLE_CLASSES)
        annotated_frame = results[0].plot()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                if counter.update(track_id, box):
                    visualizer.draw_crossing_indicator(annotated_frame, box)
        
        visualizer.draw_line(annotated_frame)
        visualizer.draw_count(annotated_frame, counter.get_count())

        # Actualizar el frame de salida de forma segura
        with lock:
            output_frame = annotated_frame.copy()
    
    cap.release()

def generate_frames():
    """
    Generador que codifica el 'output_frame' a JPEG y lo sirve como
    un stream 'multipart/x-mixed-replace'.
    """
    global output_frame, lock

    while True:
        with lock:
            if output_frame is None:
                continue
            
            # Codificar el frame a formato JPEG
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)

            if not flag:
                continue

        # Devolver el frame como parte de la respuesta multipart
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

if __name__ == '__main__':
    # Iniciar el hilo de procesamiento de video en segundo plano
    processing_thread = threading.Thread(target=process_video_stream)
    processing_thread.daemon = True
    processing_thread.start()

    # Iniciar el servidor Flask
    # Usa host='0.0.0.0' para que sea accesible desde otros dispositivos en tu red.
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True, use_reloader=False)
