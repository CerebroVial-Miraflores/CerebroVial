import cv2
from components.object_detector import ObjectDetector
from components.object_tracker import ObjectTracker
from logic.vehicle_counter import VehicleCounter
from utils.visualizer import Visualizer
from utils.logger import setup_logger
import streamlink

# Configurar logger
logger = setup_logger("CerebroVial.Stream")

# --- CONFIGURACIÓN ---
VIDEO_SOURCE = "https://youtu.be/qMYlpMsWsBE"
MODEL_PATH = 'yolov8n.pt'
VEHICLE_CLASSES = [2, 3, 5, 7] # car, motorcycle, bus, truck
LINE_Y_RATIO = 0.75
MAX_RETRIES = 3  # Intentos de reconexión

def get_video_stream(url, retry_count=0):
    """
    Usa streamlink para obtener la URL del stream de mejor calidad.
    Incluye reintentos en caso de fallo.
    """
    try:
        logger.info(f"Obteniendo stream desde: {url}")
        streams = streamlink.streams(url)
        
        if not streams:
            logger.error("No se encontraron streams disponibles en la URL")
            return None
        
        stream_url = streams["best"].url
        logger.info(f"Stream obtenido exitosamente: calidad 'best'")
        return stream_url
        
    except streamlink.exceptions.NoPluginError:
        logger.error(f"Streamlink no puede manejar esta URL: {url}")
        return None
    except Exception as e:
        logger.error(f"Error al obtener stream (intento {retry_count + 1}/{MAX_RETRIES}): {e}")
        
        # Reintentar si no hemos alcanzado el límite
        if retry_count < MAX_RETRIES - 1:
            logger.info("Reintentando en 3 segundos...")
            import time
            time.sleep(3)
            return get_video_stream(url, retry_count + 1)
        
        return None

def main():
    logger.info("=== Iniciando CerebroVial Stream ===")
    
    # 1. Obtener stream URL
    stream_url = get_video_stream(VIDEO_SOURCE)
    if not stream_url:
        logger.critical("No se pudo obtener la URL del stream. Abortando.")
        return
    
    # 2. Inicializar captura de video
    try:
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise ValueError(f"OpenCV no pudo abrir el stream: {stream_url}")
        
        # Obtener propiedades del stream
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        line_y = int(h * LINE_Y_RATIO)
        
        logger.info(f"Stream iniciado: {w}x{h} @ {fps:.2f} FPS")
        logger.info(f"Línea de conteo en Y={line_y}")
        
        # 3. Inicializar componentes
        logger.info(f"Cargando modelo YOLO: {MODEL_PATH}")
        detector = ObjectDetector(model_path=MODEL_PATH)
        tracker = ObjectTracker(detector)
        counter = VehicleCounter(line_y=line_y, direction='up')
        visualizer = Visualizer(line_y=line_y)
        
        logger.info("Componentes inicializados. Presiona 'q' para salir.")
        
        # 4. Bucle principal
        frame_count = 0
        error_count = 0
        MAX_CONSECUTIVE_ERRORS = 30  # Salir después de 30 errores seguidos
        
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                error_count += 1
                logger.warning(f"Error al leer frame (errores consecutivos: {error_count})")
                
                if error_count >= MAX_CONSECUTIVE_ERRORS:
                    logger.error("Demasiados errores consecutivos. Stream finalizado.")
                    break
                continue
            
            # Resetear contador de errores si leímos bien
            error_count = 0
            frame_count += 1
            
            # Log cada 300 frames (~10 seg a 30fps)
            if frame_count % 300 == 0:
                logger.info(f"Frames procesados: {frame_count}, Conteo actual: {counter.get_count()}")
            
            # Procesamiento
            try:
                results = tracker.track_objects(frame, classes_to_track=VEHICLE_CLASSES)
                annotated_frame = results[0].plot()
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy()
                    
                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        if counter.update(track_id, box, int(class_id)):
                            visualizer.draw_crossing_indicator(annotated_frame, box)
                            vehicle_type = counter.CLASS_NAMES.get(int(class_id), 'Desconocido')
                            logger.debug(f"Vehículo {int(track_id)} ({vehicle_type}) contado")
                
                visualizer.draw_line(annotated_frame)
                visualizer.draw_count(annotated_frame, counter.get_count())
                visualizer.draw_counts_breakdown(annotated_frame, counter.get_counts_summary())
                
                cv2.imshow("CerebroVial - Stream en Vivo", annotated_frame)
                
            except Exception as e:
                logger.error(f"Error procesando frame {frame_count}: {e}")
                continue
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Salida solicitada por usuario")
                break
        
        # 5. Limpieza
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info("=== Stream Finalizado ===")
        logger.info(f"Total de frames procesados: {frame_count}")
        logger.info(f"Conteo final: {counter.get_count()} vehículos")
        
    except Exception as e:
        logger.critical(f"Error fatal en el stream: {e}", exc_info=True)
        raise
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Programa interrumpido (Ctrl+C)")
    except Exception as e:
        logger.critical(f"Error no manejado: {e}", exc_info=True)