import cv2
import threading
import yt_dlp
from flask import Flask, Response, render_template, jsonify, request
from flask_cors import CORS # Importar CORS
from flask import request # Importar request para manejar peticiones POST
import numpy as np # Para operaciones numéricas y predicciones
import tensorflow as tf # Para cargar el modelo Keras
import joblib # Para cargar el StandardScaler
from datetime import datetime
import time

from components.object_detector import ObjectDetector
from components.object_tracker import ObjectTracker
from logic.vehicle_counter import VehicleCounter
from logic.speed_calculator import SpeedCalculator
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

# Variables globales para el modelo de IA y el escalador
traffic_prediction_model = None
traffic_scaler = None


# Estadísticas globales
stats = {
    'counter': None,
    'speed_calculator': None,
    'start_time': None,
    'frames_processed': 0,
    'fps': 0,
    'last_fps_update': time.time(),
    'frame_times': []
}

# Mapeo de tráfico (necesario para post-procesar predicciones)
# Debe ser el mismo que se usó en el entrenamiento
traffic_mapping = {'low': 0, 'normal': 1, 'high': 2, 'heavy' : 3}
inverse_traffic_mapping = {v: k for k, v in traffic_mapping.items()}

# Inicializar Flask
app = Flask(__name__)

# --- HABILITAR CORS ---
# Esto permitirá que tu frontend en localhost:3000 (o cualquier otro) se comunique con tu API.
CORS(app)


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

def load_prediction_model_and_scaler():
    """Carga el modelo de predicción de tráfico y el escalador."""
    global traffic_prediction_model, traffic_scaler
    
    model_path = config.get('deployment.model_path')
    scaler_path = config.get('deployment.scaler_path')
    
    try:
        traffic_prediction_model = tf.keras.models.load_model(model_path)
        traffic_scaler = joblib.load(scaler_path)
        logger.info(f"Modelo de predicción cargado desde: {model_path}")
        logger.info(f"Escalador cargado desde: {scaler_path}")
    except Exception as e:
        logger.error(f"Error al cargar el modelo o escalador de predicción: {e}")
        traffic_prediction_model = None
        traffic_scaler = None

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
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logger.warning("FPS del stream es 0, usando 25.0 como valor por defecto.")
        fps = 25.0

    line_y = config.get_line_y(h)
    speed_line_entry_y, speed_line_exit_y = config.get_speed_line_y_coords(h)
    speed_calc_enabled = config.get('speed_calculator.enabled', False)
    
    detector = ObjectDetector(
        model_path=MODEL_PATH,
        device=config.get('model.device', 'auto'),
        half_precision=config.get('model.half_precision', True),
        confidence=config.get('model.confidence', 0.25),
        iou=config.get('model.iou', 0.45)
    )
    
    tracker = ObjectTracker(detector)
    counter = VehicleCounter(line_y=line_y, direction=config.get('counter.direction', 'up'))
    speed_calc = None
    
    if speed_calc_enabled:
        logger.info("Calculador de velocidad habilitado.")
        speed_calc = SpeedCalculator(
            line_a_y=speed_line_entry_y,
            line_b_y=speed_line_exit_y,
            real_world_distance_m=config.get('speed_calculator.real_world_distance_m', 15.0),
            fps=fps,
            direction=config.get('counter.direction', 'up') # Usar la misma dirección que el contador
        )
        # Guardar referencia para la API
        stats['speed_calculator'] = speed_calc
    else:
        logger.info("Calculador de velocidad deshabilitado.")
    
    colors = config.get_visualization_colors()
    visualizer = Visualizer(
        line_y=line_y,
        text_color=colors['text'],
        line_color=colors['line'],
        show_trajectories=config.get('visualization.show_trajectories', True),
        trajectory_length=config.get('visualization.trajectory_length', 30)
    )
    
    # Configurar líneas de velocidad en el visualizador si están habilitadas
    if speed_calc_enabled:
        visualizer.set_speed_measurement_lines(speed_line_entry_y, speed_line_exit_y)
    
    # Guardar referencia al contador para API
    stats['counter'] = counter
    stats['start_time'] = datetime.now()
    
    logger.info("Procesamiento de video iniciado")
    
    # Variables para el cálculo de velocidad promedio
    frame_counter = 0
    current_avg_speed = 0.0
    avg_speed_interval = int(fps * config.get('speed_calculator.avg_speed_update_interval_sec', 3))


    while True:
        success, frame = cap.read()
        if not success:
            logger.warning("Error leyendo frame")
            break
        
        # Procesamiento
        frame_counter += 1
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
                
                # Actualizar calculador de velocidad si está habilitado
                if speed_calc:
                    speed_calc.update(track_id, box, frame_counter)
                    speed = speed_calc.get_speed(track_id)
                    if speed is not None:
                        visualizer.draw_speed_on_box(annotated_frame, box, speed)
        
        # Actualizar y dibujar velocidad promedio periódicamente
        if speed_calc and frame_counter % avg_speed_interval == 0:
            current_avg_speed = speed_calc.get_average_speed()
            speed_calc.reset_interval()
        
        # Dibujar visualizaciones
        visualizer.draw_trajectories(annotated_frame, results)
        visualizer.draw_lines(annotated_frame) # Ahora dibuja todas las líneas
        visualizer.draw_count(annotated_frame, counter.get_count())
        visualizer.draw_counts_breakdown(annotated_frame, counter.get_counts_summary())
        
        # Dibujar estadísticas de velocidad si está habilitado
        if speed_calc:
            visualizer.draw_speed_stats(annotated_frame, current_avg_speed)

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
    
    response_data = {
        "total_count": stats['counter'].get_count(),
        "counts_by_type": stats['counter'].get_counts_summary(),
        "counts_by_class_id": stats['counter'].get_counts_by_class(),
        "fps": round(stats['fps'], 2),
        "uptime_seconds": round(uptime, 2),
        "frames_processed": stats['frames_processed'],
        "start_time": stats['start_time'].isoformat()
    }
    
    # Añadir datos de velocidad si está disponible
    if stats['speed_calculator']:
        response_data['average_speed_kmh'] = round(stats['speed_calculator'].get_average_speed(), 2)

    return jsonify(response_data)


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


@app.route("/api/predict_traffic", methods=['POST'])
def api_predict_traffic():
    """
    API endpoint para predecir la situación del tráfico futura (+15, +30, +45 min).
    
    Ejemplo: POST /api/predict_traffic
    Body:
        {
            "features": [0.0, 25, 2, 70, 8, 3, 22, 103, 1]
        }
    Retorna:
        {
            "predicted_15_min": "normal",
            "predicted_30_min": "high",
            "predicted_45_min": "heavy"
        }
    """
    if traffic_prediction_model is None or traffic_scaler is None:
        return jsonify({"error": "Modelo de predicción no cargado o no disponible"}), 503

    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({"error": "Formato de solicitud inválido. Se espera un JSON con 'features'."}), 400

    input_features = data['features']
    if not isinstance(input_features, list) or len(input_features) != 9:
        return jsonify({"error": "Las 'features' deben ser una lista de 9 valores numéricos."}), 400

    try:
        # Convertir a array de numpy y escalar
        input_array = np.array([input_features], dtype=np.float32)
        input_scaled = traffic_scaler.transform(input_array)

        # Realizar predicción
        predictions_prob = traffic_prediction_model.predict(input_scaled, verbose=0)

        # Obtener la clase predicha para cada salida
        predicted_15 = inverse_traffic_mapping[np.argmax(predictions_prob[0][0])]
        predicted_30 = inverse_traffic_mapping[np.argmax(predictions_prob[1][0])]
        predicted_45 = inverse_traffic_mapping[np.argmax(predictions_prob[2][0])]

        return jsonify({
            "predicted_15_min": predicted_15,
            "predicted_30_min": predicted_30,
            "predicted_45_min": predicted_45
        })
    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        return jsonify({"error": f"Error interno del servidor durante la predicción: {str(e)}"}), 500


if __name__ == '__main__':
    logger.info("=== Iniciando CerebroVial Web Streamer ===")
    
    # Iniciar hilo de procesamiento
    processing_thread = threading.Thread(target=process_video_stream)
    processing_thread.daemon = True
    processing_thread.start()
    
    # Cargar el modelo de predicción y el escalador
    load_prediction_model_and_scaler()
    
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
    logger.info("  - /api/predict_traffic: Predicción de tráfico (POST)")
    logger.info("  - /api/health       : Health check")
    
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)