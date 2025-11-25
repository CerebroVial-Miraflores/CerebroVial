import pytest
from pydantic import ValidationError
from src.core.domain.traffic_models import Interseccion

def test_crear_interseccion_valida():
    """Verifica que se pueda crear una intersección con datos válidos."""
    interseccion = Interseccion(
        id="INT-001",
        nombre="Av. Principal y Calle 1",
        coordenadas=(-12.0464, -77.0428),
        estado_actual="fluido"
    )
    assert interseccion.id == "INT-001"
    assert interseccion.nombre == "Av. Principal y Calle 1"
    assert interseccion.coordenadas == (-12.0464, -77.0428)
    assert interseccion.estado_actual == "fluido"

def test_crear_interseccion_minima():
    """Verifica creación con campos obligatorios solamente."""
    interseccion = Interseccion(
        id="INT-002",
        nombre="Cruce Simple"
    )
    assert interseccion.id == "INT-002"
    assert interseccion.estado_actual == "desconocido" # Valor por defecto
    assert interseccion.coordenadas is None

def test_validacion_fallida_falta_id():
    """Verifica que falle si falta un campo obligatorio."""
    with pytest.raises(ValidationError):
        Interseccion(nombre="Sin ID")

def test_inmutabilidad():
    """Verifica que la entidad sea inmutable (frozen)."""
    interseccion = Interseccion(id="INT-003", nombre="Inmutable")
    with pytest.raises(ValidationError):
        interseccion.nombre = "Nuevo Nombre"
