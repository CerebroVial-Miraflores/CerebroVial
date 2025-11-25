from abc import ABC, abstractmethod
from typing import Any

class IVideoSource(ABC):
    """
    Puerto (Interfaz) para fuentes de video.
    Define CÓMO obtenemos video, sin importar de DÓNDE viene.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establece conexión con la fuente de video."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Cierra la conexión con la fuente de video."""
        pass

    @abstractmethod
    def get_frame(self) -> Any:
        """
        Obtiene el siguiente frame del video.
        Retorna el frame (tipo Any para no acoplar a numpy/opencv aquí) o None si no hay más.
        """
        pass
