from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', device='auto', half_precision=True, confidence=0.25, iou=0.45):
        """
        Inicializa el detector de objetos.
        :param model_path: Ruta al archivo del modelo YOLO.
        :param device: Dispositivo para ejecutar el modelo ('cpu', 'cuda', 'mps', 'auto').
        :param half_precision: Usar FP16/half precision para inferencia.
        :param confidence: Umbral de confianza para detección.
        :param iou: Umbral de IoU para Non-Maximum Suppression.
        """
        self.model = YOLO(model_path)
        
        # Resolver el dispositivo 'auto' a 'cpu' o 'cuda' para evitar errores en ultralytics
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.half = half_precision
        self.confidence = confidence
        self.iou = iou

    def detect(self, frame, classes_to_detect=None):
        """
        Realiza la detección de objetos en un frame.
        :param frame: El frame de video a procesar.
        :param classes_to_detect: Lista de IDs de clases a detectar.
        :return: Los resultados de la detección de YOLO.
        """
        return self.model(frame, classes=classes_to_detect, verbose=False, device=self.device, half=self.half, conf=self.confidence, iou=self.iou)

    def get_device_info(self) -> dict:
        """Retorna información sobre el dispositivo de cómputo utilizado."""
        info = {"device": self.device}
        if self.device.startswith('cuda') and torch.cuda.is_available():
            gpu_index = torch.cuda.current_device()
            info['gpu_name'] = torch.cuda.get_device_name(gpu_index)
            total_mem = torch.cuda.get_device_properties(gpu_index).total_memory
            info['gpu_memory'] = f"{total_mem / 1e9:.2f} GB"
        return info
        self.iou = iou

    def detect(self, frame, classes_to_detect=None):
        """
        Realiza la detección de objetos en un frame.
        :param frame: El frame de video a procesar.
        :param classes_to_detect: Lista de IDs de clases a detectar.
        :return: Los resultados de la detección de YOLO.
        """
        return self.model(frame, classes=classes_to_detect, verbose=False)
        return self.model(frame, classes=classes_to_detect, verbose=False, device=self.device, half=self.half, conf=self.confidence, iou=self.iou)

    def get_device_info(self) -> dict:
        """Retorna información sobre el dispositivo de cómputo utilizado."""
        if self.device == 'auto':
            resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            resolved_device = self.device
        
        info = {"device": resolved_device}
        if resolved_device.startswith('cuda') and torch.cuda.is_available():
            gpu_index = torch.cuda.current_device()
            info['gpu_name'] = torch.cuda.get_device_name(gpu_index)
            total_mem = torch.cuda.get_device_properties(gpu_index).total_memory
            info['gpu_memory'] = f"{total_mem / 1e9:.2f} GB"
        return info