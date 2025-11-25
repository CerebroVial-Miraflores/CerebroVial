# Progreso del Proyecto

## Logros Recientes
- **[Fecha Actual]**: Implementación del Dominio Core y Puertos.
  - Creada entidad `Interseccion` (Pydantic) en `src/core/domain`.
  - Definido puerto `IVideoSource` en `src/core/ports`.
  - Verificado con tests unitarios (`tests/unit/test_domain.py`).

## Estado Actual
- El núcleo del dominio está establecido siguiendo la Arquitectura Hexagonal.
- Próximos pasos: Implementar adaptadores para `IVideoSource` (ej. OpenCV).
