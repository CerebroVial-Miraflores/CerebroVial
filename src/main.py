import cv2
import os
from components.object_detector import ObjectDetector
from components.object_tracker import ObjectTracker
from logic.vehicle_counter import VehicleCounter
from utils.visualizer import Visualizer

# --- CONFIGURACIÓN ---
# Idealmente, esto se cargaría desde configs/default_config.yaml
VIDEO_IN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'input', 'traffic_test.mp4')
VIDEO_OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'traffic_test_output.mp4')
MODEL_PATH = 'yolov8n.pt'
VEHICLE_CLASSES = [2, 3, 5, 7] # car, motorcycle, bus, truck
LINE_Y_RATIO = 0.5 # Línea a la mitad de la altura

def main():
    # 1. Inicialización de componentes
    if not os.path.exists(VIDEO_IN_PATH):
        print(f"Error: Video no encontrado en {VIDEO_IN_PATH}")
        return
        
    cap = cv2.VideoCapture(VIDEO_IN_PATH)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    line_y = int(h * LINE_Y_RATIO)

    # Asegurarse de que el directorio de salida exista
    os.makedirs(os.path.dirname(VIDEO_OUT_PATH), exist_ok=True)
    out = cv2.VideoWriter(VIDEO_OUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    detector = ObjectDetector(model_path=MODEL_PATH)
    tracker = ObjectTracker(detector)
    counter = VehicleCounter(line_y=line_y)
    visualizer = Visualizer(line_y=line_y)

    print("Procesando video... Presiona 'q' para salir.")

    # 2. Bucle principal de procesamiento
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = tracker.track_objects(frame, classes_to_track=VEHICLE_CLASSES)
        annotated_frame = visualizer.draw_detections(frame, results)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                if counter.update(track_id, box):
                    visualizer.draw_crossing_indicator(annotated_frame, box)
        
        visualizer.draw_line(annotated_frame)
        visualizer.draw_count(annotated_frame, counter.get_count())

        out.write(annotated_frame)
        cv2.imshow("CerebroVial", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 3. Limpieza
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Procesamiento finalizado. Conteo total: {counter.get_count()}")
    print(f"Video guardado en: {VIDEO_OUT_PATH}")

if __name__ == '__main__':
    main()
