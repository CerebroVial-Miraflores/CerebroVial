from typing import Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict

class Interseccion(BaseModel):
    """
    Entidad de dominio que representa una intersección de tráfico.
    """
    id: str = Field(..., description="Identificador único de la intersección")
    nombre: str = Field(..., description="Nombre legible de la intersección")
    coordenadas: Optional[Tuple[float, float]] = Field(None, description="Latitud y Longitud")
    estado_actual: str = Field("desconocido", description="Estado actual del tráfico (ej. fluido, congestionado)")

    model_config = ConfigDict(frozen=True) # Inmutabilidad para entidades de dominio
