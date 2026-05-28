"""
Domain value objects for the Computer Vision module.
"""
from dataclasses import dataclass

_MAX_ID_LEN = 64


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
