import cv2
from components.object_detector import ObjectDetector
from components.object_tracker import ObjectTracker
from logic.vehicle_counter import VehicleCounter
from utils.visualizer import Visualizer
import streamlink

# --- CONFIGURACIÓN ---
# URL del stream de la cámara IP.
# Formato RTSP: "rtsp://usuario:contraseña@direccion_ip:puerto/ruta_del_stream"
# Formato HTTP: "http://direccion_ip/video.mjpg"
# Para probar con una webcam local, usa 0: VIDEO_SOURCE = 0
# VIDEO_SOURCE = "https://kamere.mup.gov.rs:4443/Horgos/horgos1.m3u8"
VIDEO_SOURCE = "https://youtu.be/qMYlpMsWsBE"

MODEL_PATH = 'yolov8n.pt'
VEHICLE_CLASSES = [2, 3, 5, 7] # car, motorcycle, bus, truck
LINE_Y_RATIO = 0.75 # Línea a la mitad de la altura

def get_video_stream(url):
    """Usa streamlink para obtener la URL del stream de mejor calidad."""
    try:
        streams = streamlink.streams(url)
        if not streams:
            print("No se encontraron streams en la URL.")
            return None
        return streams["best"].url
    except Exception as e:
        print(f"Error al obtener el stream con streamlink: {e}")
        return None

def main():
    # 1. Inicialización de componentes
    stream_url = get_video_stream(VIDEO_SOURCE)
    if not stream_url:
        return

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: OpenCV no pudo abrir el stream de video desde {stream_url}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    line_y = int(h * LINE_Y_RATIO)

    detector = ObjectDetector(model_path=MODEL_PATH)
    tracker = ObjectTracker(detector)
    counter = VehicleCounter(line_y=line_y, direction='up') # <-- CAMBIO CLAVE: Especificar dirección
    visualizer = Visualizer(line_y=line_y)

    print("Iniciando stream... Presiona 'q' en la ventana para salir.")

    # 2. Bucle principal de procesamiento
    while cap.isOpened():
        print("Leyendo frame del stream...")
        success, frame = cap.read()
        if not success:
            print("Stream finalizado o error al leer frame.")
            break

        # Realizar seguimiento de objetos
        results = tracker.track_objects(frame, classes_to_track=VEHICLE_CLASSES)
        
        # El método plot() de ultralytics es una forma rápida de dibujar las cajas y los IDs.
        # Lo usamos como base para nuestro frame anotado.
        annotated_frame = results[0].plot()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                # Actualizar el contador y verificar si se contó un nuevo vehículo
                if counter.update(track_id, box):
                    # Si se contó, dibujar un indicador visual en el frame ya anotado
                    visualizer.draw_crossing_indicator(annotated_frame, box)
        
        # Dibujar la línea y el contador en el frame que ya tiene las detecciones
        visualizer.draw_line(annotated_frame)
        visualizer.draw_count(annotated_frame, counter.get_count())

        # Mostrar el frame final con todas las anotaciones
        cv2.imshow("CerebroVial - Stream en Vivo", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 3. Limpieza
    cap.release()
    cv2.destroyAllWindows()
    print(f"Procesamiento finalizado. Conteo total: {counter.get_count()}")


if __name__ == '__main__':
    main()