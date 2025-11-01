from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Inicializa el detector de objetos.
        :param model_path: Ruta al archivo del modelo YOLO.
        """
        self.model = YOLO(model_path)

    def detect(self, frame, classes_to_detect=None):
        """
        Realiza la detección de objetos en un frame.
        :param frame: El frame de video a procesar.
        :param classes_to_detect: Lista de IDs de clases a detectar.
        :return: Los resultados de la detección de YOLO.
        """
        return self.model(frame, classes=classes_to_detect, verbose=False)