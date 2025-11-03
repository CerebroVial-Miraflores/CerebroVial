import cv2
import os
from components.object_detector import ObjectDetector
from components.object_tracker import ObjectTracker
from logic.vehicle_counter import VehicleCounter
from utils.visualizer import Visualizer
from utils.logger import setup_logger
from utils.data_persistence import DataPersistence, TimeSeriesTracker
from utils.config_loader import load_config
from datetime import datetime

# Cargar configuración
config = load_config()

# Configurar logger con nivel desde config
logger = setup_logger("CerebroVial.Main", log_level=config.get('logging.level', 'INFO'))

# --- CONFIGURACIÓN DESDE YAML ---
VIDEO_IN_PATH = config.get('paths.input_video')
VIDEO_OUT_PATH = config.get('paths.output_video')
MODEL_PATH = config.get('model.path', 'yolov8n.pt')
VEHICLE_CLASSES = config.get_vehicle_classes()
TIMESERIES_INTERVAL_SEC = config.get('timeseries.interval_seconds', 60)

def main():
    logger.info("=== Iniciando CerebroVial ===")
    processing_start = datetime.now()
    
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
        line_y = config.get_line_y(h)
        
        logger.info(f"Propiedades del video: {w}x{h} @ {fps:.2f} FPS, {total_frames} frames")
        logger.info(f"Línea de conteo en Y={line_y} (ratio: {config.get('counter.line_y_ratio')})")
        
        # 3. Asegurar directorio de salida
        os.makedirs(os.path.dirname(VIDEO_OUT_PATH), exist_ok=True)
        out = cv2.VideoWriter(VIDEO_OUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # 4. Inicialización de componentes
        logger.info(f"Cargando modelo YOLO: {MODEL_PATH}")
        detector = ObjectDetector(
            model_path=MODEL_PATH,
            device=config.get('model.device', 'auto'),
            half_precision=config.get('model.half_precision', True),
            confidence=config.get('model.confidence', 0.25),
            iou=config.get('model.iou', 0.45)
        )
        
        # Log de información del dispositivo
        device_info = detector.get_device_info()
        logger.info(f"Dispositivo: {device_info['device']}")
        if 'gpu_name' in device_info:
            logger.info(f"GPU: {device_info['gpu_name']} ({device_info['gpu_memory']})")
        
        tracker = ObjectTracker(detector)
        counter = VehicleCounter(line_y=line_y, direction=config.get('counter.direction', 'down'))
        
        # Usar colores desde config
        colors = config.get_visualization_colors()
        visualizer = Visualizer(
            line_y=line_y, 
            text_color=colors['text'], 
            line_color=colors['line'],
            show_trajectories=config.get('visualization.show_trajectories', True),
            trajectory_length=config.get('visualization.trajectory_length', 30)
        )
        
        # Inicializar persistencia de datos
        results_dir = config.get('paths.results_dir', 'data/results')
        persistence = DataPersistence(output_dir=results_dir)
        
        if config.get('timeseries.enabled', True):
            timeseries_tracker = TimeSeriesTracker(interval_seconds=TIMESERIES_INTERVAL_SEC, fps=fps)
        else:
            timeseries_tracker = None
        
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
            
            # Actualizar trayectorias
            visualizer.update_trajectories(results)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                
                # Limpiar trayectorias antiguas
                active_ids = set(int(tid) for tid in track_ids)
                visualizer.clear_old_trajectories(active_ids)
                
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    if counter.update(track_id, box, int(class_id)):
                        visualizer.draw_crossing_indicator(annotated_frame, box)
                        vehicle_type = counter.CLASS_NAMES.get(int(class_id), 'Desconocido')
                        logger.debug(f"Vehículo {int(track_id)} ({vehicle_type}) contado en frame {frame_count}")
            
            # Actualizar serie temporal
            if timeseries_tracker:
                timeseries_tracker.update(frame_count, counter)
            
            # Dibujar trayectorias
            visualizer.draw_trajectories(annotated_frame, results)
            
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
        processing_end = datetime.now()
        processing_duration = (processing_end - processing_start).total_seconds()
        
        logger.info("=== Procesamiento Finalizado ===")
        logger.info(f"Frames procesados: {frame_count}/{total_frames}")
        logger.info(f"Duración: {processing_duration:.2f} segundos")
        logger.info(f"Conteo total: {counter.get_count()} vehículos")
        
        # Desglose por tipo
        counts_summary = counter.get_counts_summary()
        logger.info("Desglose por tipo:")
        for vehicle_type, count in counts_summary.items():
            logger.info(f"  - {vehicle_type}: {count}")
        
        logger.info(f"Video de salida guardado en: {VIDEO_OUT_PATH}")
        
        # 8. Guardar resultados
        video_info = {
            "path": VIDEO_IN_PATH,
            "width": w,
            "height": h,
            "fps": fps,
            "total_frames": total_frames
        }
        
        processing_info = {
            "frames_processed": frame_count,
            "duration_sec": round(processing_duration, 2),
            "line_y": line_y,
            "direction": counter.direction,
            "model": MODEL_PATH
        }
        
        saved_files = persistence.save_results(
            counter=counter,
            video_info=video_info,
            processing_info=processing_info,
            time_series=timeseries_tracker.get_time_series() if timeseries_tracker else None
        )
        
        logger.info("=== Resultados Guardados ===")
        logger.info(f"JSON: {saved_files['json']}")
        logger.info(f"CSV Resumen: {saved_files['csv_summary']}")
        if saved_files['csv_timeseries']:
            logger.info(f"CSV Serie Temporal: {saved_files['csv_timeseries']}")
        
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