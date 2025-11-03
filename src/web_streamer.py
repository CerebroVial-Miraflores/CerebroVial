import cv2
import threading
import yt_dlp
from flask import Flask, Response, render_template, jsonify
from datetime import datetime
import time

from components.object_detector import ObjectDetector
from components.object_tracker import ObjectTracker
from logic.vehicle_counter import VehicleCounter
from utils.visualizer import Visualizer
from utils.logger import setup_logger
from utils.config_loader import load_config

# Cargar configuración
config = load_config()
logger = setup_logger("CerebroVial.WebStreamer", log_level=config.get('logging.level', 'INFO'))

# --- CONFIGURACIÓN ---
VIDEO_SOURCE = config.get('stream.url', 'https://youtu.be/qMYlpMsWsBE')
MODEL_PATH = config.get('model.path', 'yolov8n.pt')
VEHICLE_CLASSES = config.get_vehicle_classes()

# --- VARIABLES GLOBALES ---
output_frame = None
lock = threading.Lock()

# Estadísticas globales
stats = {
    'counter': None,
    'start_time': None,
    'frames_processed': 0,
    'fps': 0,
    'last_fps_update': time.time(),
    'frame_times': []
}

# Inicializar Flask
app = Flask(__name__)


def get_video_stream_url(url):
    """Usa yt-dlp para obtener la URL directa del stream."""
    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]/best[height<=720]',
        'quiet': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get('url')
    except Exception as e:
        logger.error(f"Error obteniendo stream: {e}")
        return None


def calculate_fps():
    """Calcula FPS actual basado en frames recientes."""
    global stats
    
    current_time = time.time()
    stats['frame_times'].append(current_time)
    
    # Mantener solo últimos 30 frames
    if len(stats['frame_times']) > 30:
        stats['frame_times'].pop(0)
    
    # Calcular FPS
    if len(stats['frame_times']) > 1:
        time_diff = stats['frame_times'][-1] - stats['frame_times'][0]
        if time_diff > 0:
            stats['fps'] = (len(stats['frame_times']) - 1) / time_diff


def process_video_stream():
    """Función que procesa el video en un hilo separado."""
    global output_frame, lock, stats
    
    logger.info("Iniciando hilo de procesamiento de video...")
    
    stream_url = get_video_stream_url(VIDEO_SOURCE)
    if not stream_url:
        logger.error("No se pudo obtener URL del stream")
        return
    
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        logger.error(f"No se pudo abrir stream: {stream_url}")
        return
    
    # Inicialización
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = config.get_line_y(h)
    
    detector = ObjectDetector(
        model_path=MODEL_PATH,
        device=config.get('model.device', 'auto'),
        half_precision=config.get('model.half_precision', True),
        confidence=config.get('model.confidence', 0.25),
        iou=config.get('model.iou', 0.45)
    )
    
    tracker = ObjectTracker(detector)
    counter = VehicleCounter(line_y=line_y, direction=config.get('counter.direction', 'up'))
    
    colors = config.get_visualization_colors()
    visualizer = Visualizer(
        line_y=line_y,
        text_color=colors['text'],
        line_color=colors['line'],
        show_trajectories=config.get('visualization.show_trajectories', True),
        trajectory_length=config.get('visualization.trajectory_length', 30)
    )
    
    # Guardar referencia al contador para API
    stats['counter'] = counter
    stats['start_time'] = datetime.now()
    
    logger.info("Procesamiento de video iniciado")
    
    while True:
        success, frame = cap.read()
        if not success:
            logger.warning("Error leyendo frame")
            break
        
        # Procesamiento
        results = tracker.track_objects(frame, classes_to_track=VEHICLE_CLASSES)
        annotated_frame = results[0].plot()
        
        # Actualizar trayectorias
        visualizer.update_trajectories(results)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            
            active_ids = set(int(tid) for tid in track_ids)
            visualizer.clear_old_trajectories(active_ids)
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if counter.update(track_id, box, int(class_id)):
                    visualizer.draw_crossing_indicator(annotated_frame, box)
        
        # Dibujar visualizaciones
        visualizer.draw_trajectories(annotated_frame, results)
        visualizer.draw_line(annotated_frame)
        visualizer.draw_count(annotated_frame, counter.get_count())
        visualizer.draw_counts_breakdown(annotated_frame, counter.get_counts_summary())
        
        # Dibujar FPS
        if config.get('visualization.show_fps', True):
            cv2.putText(
                annotated_frame,
                f'FPS: {stats["fps"]:.1f}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        # Actualizar estadísticas
        stats['frames_processed'] += 1
        calculate_fps()
        
        # Actualizar frame de salida
        with lock:
            output_frame = annotated_frame.copy()
    
    cap.release()
    logger.info("Procesamiento de video finalizado")


def generate_frames():
    """Generador que sirve frames como stream multipart."""
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
    """Página principal."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Endpoint que sirve el stream de video."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/stats")
def api_stats():
    """
    API endpoint que retorna estadísticas en tiempo real.
    
    Ejemplo: GET /api/stats
    Retorna:
        {
            "total_count": 42,
            "counts_by_type": {"Autos": 30, "Motos": 10, ...},
            "fps": 28.5,
            "uptime_seconds": 120,
            "frames_processed": 3600
        }
    """
    if stats['counter'] is None or stats['start_time'] is None:
        return jsonify({"error": "Sistema no inicializado"}), 503
    
    uptime = (datetime.now() - stats['start_time']).total_seconds()
    
    return jsonify({
        "total_count": stats['counter'].get_count(),
        "counts_by_type": stats['counter'].get_counts_summary(),
        "counts_by_class_id": stats['counter'].get_counts_by_class(),
        "fps": round(stats['fps'], 2),
        "uptime_seconds": round(uptime, 2),
        "frames_processed": stats['frames_processed'],
        "start_time": stats['start_time'].isoformat(),
        "line_y": stats['counter'].line_y,
        "direction": stats['counter'].direction
    })


@app.route("/api/health")
def api_health():
    """
    Health check endpoint.
    
    Ejemplo: GET /api/health
    Retorna: {"status": "ok", "timestamp": "2024-..."}
    """
    return jsonify({
        "status": "ok" if stats['counter'] is not None else "initializing",
        "timestamp": datetime.now().isoformat()
    })


if __name__ == '__main__':
    logger.info("=== Iniciando CerebroVial Web Streamer ===")
    
    # Iniciar hilo de procesamiento
    processing_thread = threading.Thread(target=process_video_stream)
    processing_thread.daemon = True
    processing_thread.start()
    
    # Dar tiempo para inicialización
    time.sleep(2)
    
    # Iniciar servidor Flask
    port = config.get('web.port', 8080)
    host = config.get('web.host', '0.0.0.0')
    
    logger.info(f"Servidor web iniciando en http://{host}:{port}")
    logger.info("Endpoints disponibles:")
    logger.info("  - /                 : Interfaz web")
    logger.info("  - /video_feed       : Stream de video")
    logger.info("  - /api/stats        : Estadísticas JSON")
    logger.info("  - /api/health       : Health check")
    
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)