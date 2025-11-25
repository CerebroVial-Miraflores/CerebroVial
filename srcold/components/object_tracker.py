from .object_detector import ObjectDetector

class ObjectTracker:
    """
    Rastreador de objetos usando el tracking integrado de YOLO.
    """
    
    def __init__(self, detector: ObjectDetector):
        """
        Inicializa el rastreador de objetos.
        :param detector: Una instancia de ObjectDetector.
        """
        self.detector = detector

    def track_objects(self, frame, classes_to_track=None):
        """
        Realiza el seguimiento de objetos en un frame.
        
        :param frame: El frame de video a procesar.
        :param classes_to_track: Lista de IDs de clases a seguir.
        :return: Los resultados del seguimiento de YOLO (con IDs).
        """
        # persist=True es clave para que el tracker recuerde los objetos entre frames
        return self.detector.model.track(
            frame, 
            persist=True, 
            classes=classes_to_track,
            conf=self.detector.confidence,
            iou=self.detector.iou,
            verbose=False,
            device=self.detector.device
        )