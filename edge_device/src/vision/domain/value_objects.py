"""
Domain value objects for the Computer Vision module.
"""
from dataclasses import dataclass
from enum import Enum

_MAX_ID_LEN = 64


class SensorStatus(str, Enum):
    """Estado del sensor de una cámara en un ciclo del scheduler (B1 1b, D-018).

    Distingue "midiendo" de "no midiendo, con motivo" — la honestidad de datos
    del D-018 (NULL-con-motivo). `str`-Enum a propósito: **extensible** (agregar
    miembros en fases posteriores) y serializable directo al payload de health.

    Motivos básicos de 1b. El set fino de D-017 (separar "cámara caída" de "sin
    stream disponible", y `RECALIBRANDO`) es DEUDA de Fase 1/3: `RECALIBRANDO`
    depende de la calibración (que no existe aún), y el split caída/sin-stream
    requiere distinguir open-fail de read-fail. 1b colapsa ambos en
    `FUENTE_NO_DISPONIBLE` (simplificación consciente, declarada).
    """

    OK = "ok"                                # frame fresco → se infiere
    SIN_FRAME_FRESCO = "sin_frame_fresco"    # hay frame pero viejo (stream lento/caído/colgado)
    FUENTE_NO_DISPONIBLE = "fuente_no_disponible"  # nunca hubo frame / captura no-live (EOF/muerta)
    # DEUDA Fase 1/3: RECALIBRANDO = "recalibrando"  (depende de calibración)


@dataclass(frozen=True)
class VehicleId:
    """Identifier for a detected and tracked vehicle within a session."""
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("VehicleId no puede ser vacío")
        if len(self.value) > _MAX_ID_LEN:
            raise ValueError(f"VehicleId excede {_MAX_ID_LEN} caracteres")


@dataclass(frozen=True)
class ZoneId:
    """Identifier for a configured zone (polygon) in a camera frame."""
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("ZoneId no puede ser vacío")
        if len(self.value) > _MAX_ID_LEN:
            raise ValueError(f"ZoneId excede {_MAX_ID_LEN} caracteres")


@dataclass(frozen=True)
class CameraId:
    """Identifier for a camera (intersection access point)."""
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("CameraId no puede ser vacío")
        if len(self.value) > _MAX_ID_LEN:
            raise ValueError(f"CameraId excede {_MAX_ID_LEN} caracteres")
