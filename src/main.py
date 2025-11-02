import cv2
import os
from components.object_detector import ObjectDetector
from components.object_tracker import ObjectTracker
from logic.vehicle_counter import VehicleCounter
from utils.visualizer import Visualizer
from utils.logger import setup_logger

# Configurar logger
logger = setup_logger("CerebroVial.Main")

# --- CONFIGURACIÓN ---
VIDEO_IN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'input', 'traffic_test.mp4')
VIDEO_OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'output', 'traffic_test_output.mp4')
MODEL_PATH = 'yolov8n.pt'
VEHICLE_CLASSES = [2, 3, 5, 7] # car, motorcycle, bus, truck
LINE_Y_RATIO = 0.5

def main():
    logger.info("=== Iniciando CerebroVial ===")
    
    # 1. Validación de archivos
    if not os.path.exists(VIDEO_IN_PATH):
        logger.error(f"Video no encontrado en {VIDEO_IN_PATH}")
        return
    
    logger.info(f"Video de entrada: {VIDEO_IN_PATH}")
    
    try:
        # 2. Inicialización de video
        cap = cv2.VideoCapture(VIDEO_IN_PATH)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {VIDEO_IN_PATH}")
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        line_y = int(h * LINE_Y_RATIO)
        
        logger.info(f"Propiedades del video: {w}x{h} @ {fps:.2f} FPS, {total_frames} frames")
        logger.info(f"Línea de conteo en Y={line_y} (ratio: {LINE_Y_RATIO})")
        
        # 3. Asegurar directorio de salida
        os.makedirs(os.path.dirname(VIDEO_OUT_PATH), exist_ok=True)
        out = cv2.VideoWriter(VIDEO_OUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # 4. Inicialización de componentes
        logger.info(f"Cargando modelo YOLO: {MODEL_PATH}")
        detector = ObjectDetector(model_path=MODEL_PATH)
        tracker = ObjectTracker(detector)
        counter = VehicleCounter(line_y=line_y, direction='down')
        visualizer = Visualizer(line_y=line_y)
        
        logger.info("Todos los componentes inicializados correctamente")
        logger.info("Procesando video... Presiona 'q' en la ventana para salir.")
        
        # 5. Bucle principal de procesamiento
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logger.info("Fin del video alcanzado")
                break
            
            frame_count += 1
            
            # Log de progreso cada 10% del video
            if frame_count % (total_frames // 10) == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Progreso: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            # Procesamiento
            results = tracker.track_objects(frame, classes_to_track=VEHICLE_CLASSES)
            annotated_frame = visualizer.draw_detections(frame, results)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    if counter.update(track_id, box, int(class_id)):
                        visualizer.draw_crossing_indicator(annotated_frame, box)
                        vehicle_type = counter.CLASS_NAMES.get(int(class_id), 'Desconocido')
                        logger.debug(f"Vehículo {int(track_id)} ({vehicle_type}) contado en frame {frame_count}")
            
            visualizer.draw_line(annotated_frame)
            visualizer.draw_count(annotated_frame, counter.get_count())
            visualizer.draw_counts_breakdown(annotated_frame, counter.get_counts_summary())
            
            out.write(annotated_frame)
            cv2.imshow("CerebroVial", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.warning("Procesamiento interrumpido por el usuario")
                break
        
        # 6. Limpieza
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # 7. Resumen final
        logger.info("=== Procesamiento Finalizado ===")
        logger.info(f"Frames procesados: {frame_count}/{total_frames}")
        logger.info(f"Conteo total: {counter.get_count()} vehículos")
        logger.info(f"Video de salida guardado en: {VIDEO_OUT_PATH}")
        
    except Exception as e:
        logger.error(f"Error durante el procesamiento: {e}", exc_info=True)
        raise
    finally:
        # Asegurar limpieza incluso si hay error
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Programa interrumpido por el usuario (Ctrl+C)")
    except Exception as e:
        logger.critical(f"Error fatal: {e}", exc_info=True)