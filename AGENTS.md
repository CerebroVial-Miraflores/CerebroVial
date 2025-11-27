# CerebroVial - Contexto para Agentes de IA

> **Versión:** 2.0  
> **Última actualización:** 2025-11-26  
> **Propósito:** Guiar a agentes de IA en el desarrollo, mantenimiento y testing del proyecto CerebroVial

---

## 📋 Tabla de Contenidos

- [Resumen del Proyecto](#resumen-del-proyecto)
- [Arquitectura](#arquitectura)
- [Estándares de Código](#estándares-de-código)
- [Patrones de Diseño](#patrones-de-diseño)
- [Testing](#testing)
- [Flujo de Trabajo](#flujo-de-trabajo)
- [Comandos Comunes](#comandos-comunes)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Resumen del Proyecto

**CerebroVial** es un sistema inteligente de gestión de tráfico urbano que integra:
- **Visión por Computadora** (YOLO + Tracking)
- **Predicción de Congestión** (GNN + LSTM)
- **Control Adaptativo** (Optimización semafórica)

### Tecnologías Clave
- **Lenguaje:** Python 3.10+
- **ML/CV:** PyTorch, Ultralytics (YOLO), Supervision
- **Backend:** FastAPI, SQLAlchemy
- **DB:** PostgreSQL + TimescaleDB (PostGIS)
- **Config:** Hydra
- **Testing:** Pytest

### Estado Actual
✅ Módulo `vision` completamente funcional y production-ready  
🚧 Módulos `prediction` y `control` en desarrollo

---

## 🏗️ Arquitectura

### Principio: Monolito Modular

El proyecto sigue una arquitectura de **Monolito Modular** que permite:
- Desarrollo independiente de módulos
- Despliegue unificado
- Escalabilidad futura hacia microservicios

### Estructura de Directorios
```
cerebrovial/
├── src/
│   ├── common/              # Código compartido entre módulos
│   │   ├── config/          # Gestión de configuración
│   │   ├── database/        # Modelos de DB y conexión
│   │   ├── schemas/         # Pydantic models (validación)
│   │   ├── exceptions.py    # Excepciones del dominio
│   │   ├── logging.py       # Setup de logging
│   │   └── metrics.py       # Métricas de rendimiento
│   │
│   ├── vision/              # Módulo de Visión por Computadora
│   │   ├── domain.py        # Entidades del dominio (DetectedVehicle, Frame, etc)
│   │   ├── application/     # Lógica de aplicación
│   │   │   ├── builder.py   # Builder Pattern para pipeline
│   │   │   ├── pipeline.py  # Orquestador principal
│   │   │   ├── processors.py # Chain of Responsibility
│   │   │   └── aggregator.py
│   │   ├── infrastructure/  # Adaptadores e implementaciones
│   │   │   ├── sources.py   # Video sources (YouTube, webcam, file)
│   │   │   ├── yolo_detector.py
│   │   │   ├── tracking.py
│   │   │   ├── zones.py
│   │   │   └── repositories.py
│   │   └── presentation/    # API endpoints
│   │       └── api.py       # FastAPI app
│   │
│   ├── prediction/          # Módulo de Predicción [EN DESARROLLO]
│   │   └── domain.py
│   │
│   └── control/             # Módulo de Control [EN DESARROLLO]
│       └── domain.py
│
├── conf/                    # Configuración (Hydra)
│   ├── config.yaml          # Config principal
│   └── vision/              # Configs específicos de vision
│       ├── default.yaml
│       ├── balanced.yaml
│       ├── low_latency.yaml
│       └── vehicle_classes.yaml
│
├── tests/                   # Tests organizados por tipo
│   ├── unit/                # Tests unitarios puros
│   ├── integration/         # Tests con dependencias reales
│   ├── e2e/                 # Tests end-to-end
│   └── conftest.py          # Fixtures compartidos
│
├── scripts/                 # Scripts de ejecución
│   ├── run_vision.py        # Ejecutar visión en modo GUI
│   └── run_server.py        # Ejecutar API server
│
├── data/                    # Datos (gitignored)
│   ├── vision/
│   ├── prediction/
│   └── control/
│
└── docs/                    # Documentación
    └── specs/
```

### Capas de Arquitectura

1. **Domain Layer** (`domain.py`)
   - Entidades de negocio
   - Protocols (interfaces)
   - Sin dependencias externas

2. **Application Layer** (`application/`)
   - Casos de uso
   - Orquestación de lógica
   - Patrones: Builder, Chain of Responsibility

3. **Infrastructure Layer** (`infrastructure/`)
   - Implementaciones concretas
   - Adaptadores a librerías externas
   - Repositorios de datos

4. **Presentation Layer** (`presentation/`)
   - APIs REST
   - Endpoints HTTP

---

## 📐 Estándares de Código

### Principios SOLID

✅ **Aplicamos:**
- **S**ingle Responsibility: Cada clase tiene una responsabilidad única
- **O**pen/Closed: Extensible sin modificar (ej: SourceFactory)
- **L**iskov Substitution: Protocols permiten sustituibilidad
- **I**nterface Segregation: Protocols pequeños y específicos
- **D**ependency Inversion: Dependemos de abstracciones (Protocols)

### Convenciones de Nombres
```python
# Clases: PascalCase
class VehicleDetector(Protocol):
    pass

# Funciones/métodos: snake_case
def detect_vehicles(frame: np.ndarray) -> FrameAnalysis:
    pass

# Constantes: UPPER_SNAKE_CASE
MAX_BUFFER_SIZE = 10

# Variables privadas: _prefijo
def __init__(self):
    self._internal_state = None

# Type hints SIEMPRE
def process(frame: Frame, analysis: Optional[FrameAnalysis]) -> FrameAnalysis:
    pass
```

### Docstrings
```python
def complex_function(param1: str, param2: int) -> Dict[str, Any]:
    """
    Descripción breve de la función.

    Args:
        param1: Descripción del parámetro 1
        param2: Descripción del parámetro 2

    Returns:
        Diccionario con los resultados procesados

    Raises:
        ValueError: Si param2 es negativo
        DetectionError: Si la detección falla

    Example:
        >>> result = complex_function("test", 42)
        >>> print(result)
        {'status': 'ok'}
    """
    pass
```

### Imports
```python
# Standard library
import os
import time
from typing import List, Dict, Optional

# Third party
import numpy as np
import cv2
from pydantic import BaseModel

# Local
from ..domain import Frame, FrameAnalysis
from ...common.exceptions import DetectionError
```

---

## 🎨 Patrones de Diseño

### 1. Builder Pattern
**Ubicación:** `src/vision/application/builder.py`

**Cuándo usar:** Para construir objetos complejos paso a paso.
```python
# ✅ CORRECTO
builder = VisionApplicationBuilder(cfg)
pipeline = (
    builder
    .build_detector()
    .build_tracker()
    .build_source()
    .build_pipeline()
)

# ❌ INCORRECTO - No construir manualmente
detector = YoloDetector(...)
tracker = SupervisionTracker(...)
# ... construcción manual compleja
```

### 2. Chain of Responsibility
**Ubicación:** `src/vision/application/processors.py`

**Cuándo usar:** Para procesar datos a través de una cadena de procesadores.
```python
# ✅ CORRECTO
processor_chain = DetectionProcessor(detector)
processor_chain.set_next(TrackingProcessor(tracker))
processor_chain.set_next(SpeedEstimationProcessor(estimator))

# Los procesadores se encadenan automáticamente
analysis = processor_chain.process(frame, None)
```

### 3. Factory Pattern
**Ubicación:** `src/vision/infrastructure/sources.py`

**Cuándo usar:** Para crear objetos de diferentes tipos basados en condiciones.
```python
# ✅ CORRECTO - El factory decide qué clase instanciar
source = create_source("video.mp4")  # VideoFileSource
source = create_source("https://youtube.com/...")  # YouTubeSource
source = create_source(0)  # WebcamSource

# ❌ INCORRECTO - No instanciar directamente
source = VideoFileSource("video.mp4")  # Pierde flexibilidad
```

### 4. Dependency Injection
**Ubicación:** `src/vision/presentation/api.py`

**Cuándo usar:** Para inyectar dependencias en lugar de crearlas.
```python
# ✅ CORRECTO
@app.get("/metrics")
async def get_metrics(service: VisionService = Depends(get_vision_service)):
    return service.get_metrics()

# ❌ INCORRECTO - No usar variables globales directamente
@app.get("/metrics")
async def get_metrics():
    return _service.get_metrics()  # Acoplamiento fuerte
```

### 5. Repository Pattern
**Ubicación:** `src/vision/infrastructure/repositories.py`

**Cuándo usar:** Para abstraer el acceso a datos.
```python
# ✅ CORRECTO
class CSVTrafficRepository(TrafficRepository):
    def save(self, data: TrafficData):
        # Implementación específica de CSV
        pass

# Fácil de cambiar a DB sin modificar lógica de negocio
class PostgresTrafficRepository(TrafficRepository):
    def save(self, data: TrafficData):
        # Implementación específica de PostgreSQL
        pass
```

---

## 🧪 Testing

### Pirámide de Testing
```
        /\
       /  \  E2E (pocos, lentos, frágiles)
      /----\
     /      \ Integration (algunos, medianos)
    /--------\
   /          \ Unit (muchos, rápidos, confiables)
  /____________\
```

### Estructura de Tests
```
tests/
├── unit/                    # 70% de los tests
│   ├── vision/
│   │   ├── test_detector.py
│   │   ├── test_tracker.py
│   │   └── test_zones.py
│   └── common/
│       └── test_schemas.py
│
├── integration/             # 20% de los tests
│   └── vision/
│       └── test_builder.py  # Prueba componentes integrados
│
└── e2e/                     # 10% de los tests
    └── test_full_pipeline.py
```

### Convenciones de Testing

#### Nombres de Tests
```python
# Patrón: test_<componente>_<escenario>_<resultado_esperado>

def test_detector_valid_frame_returns_analysis():
    """Detector debe retornar FrameAnalysis con frame válido"""
    pass

def test_detector_empty_frame_returns_empty_analysis():
    """Detector debe retornar análisis vacío con frame sin vehículos"""
    pass

def test_detector_invalid_frame_raises_detection_error():
    """Detector debe lanzar DetectionError con frame inválido"""
    pass
```

#### Estructura AAA (Arrange-Act-Assert)
```python
def test_zone_counter_counts_vehicles_in_zone():
    # Arrange: Preparar datos de prueba
    config = {"zone1": [[0, 0], [100, 0], [100, 100], [0, 100]]}
    counter = ZoneCounter(config)
    vehicle = DetectedVehicle(id="1", type="car", bbox=(50, 50, 60, 60))
    
    # Act: Ejecutar la acción
    result = counter.count_vehicles_in_zones([vehicle])
    
    # Assert: Verificar resultado
    assert len(result) == 1
    assert result[0].vehicle_count == 1
    assert result[0].zone_id == "zone1"
```

#### Fixtures Compartidos

**Ubicación:** `tests/conftest.py`
```python
@pytest.fixture
def mock_frame():
    """Frame de prueba reutilizable"""
    return Frame(
        id=0,
        timestamp=1234567890.0,
        image=np.zeros((100, 100, 3), dtype=np.uint8)
    )

@pytest.fixture
def mock_detector():
    """Detector mockeado para tests rápidos"""
    detector = Mock(spec=VehicleDetector)
    detector.detect.return_value = FrameAnalysis(...)
    return detector
```

#### Mocking
```python
# ✅ CORRECTO - Mock solo dependencias externas
def test_yolo_detector():
    with patch('src.vision.infrastructure.yolo_detector.YOLO') as mock_yolo:
        mock_yolo.return_value.return_value = [mock_result]
        
        detector = YoloDetector()
        result = detector.detect(frame)
        
        assert result.total_count == 1

# ❌ INCORRECTO - No mockear tu propio código
def test_pipeline():
    with patch('src.vision.application.pipeline.VisionPipeline'):  # NO
        pass
```

### Generación Automática de Tests

**REGLA:** Al crear una nueva clase/función en `src/`, SIEMPRE genera tests correspondientes.

#### Template para Tests Unitarios
```python
# tests/unit/vision/test_<nombre_modulo>.py
import pytest
from unittest.mock import Mock, patch
from src.vision.infrastructure.<modulo> import <Clase>

class Test<Clase>:
    """Tests para <Clase>"""
    
    def test_initialization_with_valid_params(self):
        """Debe inicializarse correctamente con parámetros válidos"""
        # Arrange & Act
        instance = <Clase>(valid_param=value)
        
        # Assert
        assert instance.valid_param == value
    
    def test_initialization_with_invalid_params_raises_error(self):
        """Debe lanzar error con parámetros inválidos"""
        with pytest.raises(ValueError):
            <Clase>(invalid_param=value)
    
    def test_main_method_with_valid_input_returns_expected(self):
        """Método principal debe retornar resultado esperado"""
        # Arrange
        instance = <Clase>()
        valid_input = ...
        
        # Act
        result = instance.main_method(valid_input)
        
        # Assert
        assert result == expected_output
    
    def test_main_method_with_edge_case_handles_correctly(self):
        """Debe manejar casos límite correctamente"""
        pass
```

#### Template para Tests de Integración
```python
# tests/integration/test_<feature>.py
import pytest
from src.vision.application.builder import VisionApplicationBuilder

def test_<feature>_integration():
    """Test de integración completo para <feature>"""
    # Arrange: Setup real (o casi real)
    config = load_test_config()
    
    # Act: Ejecutar flujo completo
    result = execute_feature(config)
    
    # Assert: Verificar resultado final
    assert result.is_valid()
    assert result.meets_requirements()
```

### Coverage

**Objetivo:** ≥ 80% de cobertura
```bash
# Ejecutar tests con coverage
pytest --cov=src --cov-report=html --cov-report=term

# Ver reporte
open htmlcov/index.html
```

---

## 🔄 Flujo de Trabajo

### Agregar Nueva Funcionalidad

1. **Definir en Domain** (`domain.py`)
```python
   # Definir Protocol o Dataclass
   class NewFeature(Protocol):
       def process(self, data: Data) -> Result:
           ...
```

2. **Implementar en Infrastructure** (`infrastructure/`)
```python
   class ConcreteNewFeature(NewFeature):
       def process(self, data: Data) -> Result:
           # Implementación
           pass
```

3. **Integrar en Application** (`application/`)
```python
   # Agregar al Builder o Pipeline
   def build_new_feature(self):
       self.new_feature = ConcreteNewFeature(...)
       return self
```

4. **Escribir Tests**
```python
   # tests/unit/test_new_feature.py
   def test_new_feature_basic_functionality():
       pass
```

5. **Actualizar Configuración** (`conf/`)
```yaml
   new_feature:
     enabled: true
     param: value
```

### Modificar Código Existente

**ANTES de modificar:**
1. ✅ Leer el código actual completo
2. ✅ Entender los tests existentes
3. ✅ Verificar patrones usados
4. ✅ Identificar dependencias

**AL modificar:**
1. ✅ Mantener mismo patrón de diseño
2. ✅ Actualizar docstrings
3. ✅ Actualizar/agregar tests
4. ✅ Verificar que no rompes tests existentes

**DESPUÉS de modificar:**
1. ✅ Ejecutar todos los tests: `pytest`
2. ✅ Verificar type hints: `mypy src/`
3. ✅ Formatear código: `black src/`
4. ✅ Ordenar imports: `isort src/`

---

## 💻 Comandos Comunes

### Desarrollo
```bash
# Ejecutar visión con GUI
python scripts/run_vision.py

# Ejecutar con perfil específico
python scripts/run_vision.py --config-name=vision/low_latency

# Ejecutar API server
python scripts/run_server.py

# Ver métricas
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health
```

### Testing
```bash
# Todos los tests
pytest

# Solo tests unitarios
pytest tests/unit/

# Solo tests de un módulo
pytest tests/unit/vision/

# Con verbose
pytest -v

# Con coverage
pytest --cov=src --cov-report=html

# Ejecutar test específico
pytest tests/unit/vision/test_detector.py::test_yolo_detector_detect

# Modo watch (re-ejecuta al cambiar archivos)
ptw  # requiere pytest-watch
```

### Calidad de Código
```bash
# Type checking
mypy src/

# Formateo
black src/ tests/

# Ordenar imports
isort src/ tests/

# Linting
flake8 src/

# Todo en uno
black src/ && isort src/ && mypy src/ && pytest
```

### Base de Datos
```bash
# Inicializar DB
python -c "from src.common.database import init_db; init_db()"

# Ejecutar migrations (si usamos Alembic)
alembic upgrade head

# Crear nueva migration
alembic revision --autogenerate -m "description"
```

---

## 🐛 Troubleshooting

### Problemas Comunes

#### Error: "Could not open video source"
**Causa:** OpenCV no puede abrir la fuente
**Solución:**
```python
# Verificar que el archivo existe
assert os.path.exists("video.mp4")

# Para YouTube, verificar yt-dlp actualizado
pip install --upgrade yt-dlp

# Para webcam, verificar permisos
# macOS: System Preferences > Security & Privacy > Camera
```

#### Error: "YOLO model not found"
**Causa:** Modelo YOLO no descargado
**Solución:**
```bash
# El modelo se descarga automáticamente la primera vez
# Si falla, descargar manualmente:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
mv yolo11n.pt models/
```

#### Error: "ImportError: No module named 'src'"
**Causa:** Python no encuentra el módulo
**Solución:**
```bash
# Agregar directorio raíz al PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# O instalar en modo desarrollo
pip install -e .
```

#### Tests Fallan en CI pero Pasan Localmente
**Causa:** Diferencias de entorno
**Solución:**
```bash
# Usar mismas versiones de dependencias
pip install -r requirements.txt --no-cache-dir

# Ejecutar en ambiente limpio
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
pytest
```

---

## 🎯 Checklist para Agentes de IA

Antes de considerar una tarea completa, verificar:

### Para Código Nuevo
- [ ] Código sigue patrones del módulo existente
- [ ] Type hints en todas las funciones
- [ ] Docstrings en clases y funciones públicas
- [ ] Manejo de errores con excepciones del dominio
- [ ] Tests unitarios escritos (≥3 casos)
- [ ] Tests de integración si aplica
- [ ] Sin código duplicado (DRY)
- [ ] Configuración externalizada (no hardcoded)
- [ ] Logging apropiado

### Para Modificaciones
- [ ] Tests existentes aún pasan
- [ ] Nuevos tests para cambios
- [ ] Documentación actualizada
- [ ] Backward compatibility mantenida
- [ ] Performance no degradada

### Para Tests
- [ ] Nombres descriptivos (`test_<component>_<scenario>_<expected>`)
- [ ] Estructura AAA clara
- [ ] Mocks solo para dependencias externas
- [ ] Assertions específicas (no solo `assert result`)
- [ ] Edge cases cubiertos

---

## 📚 Referencias

### Documentación Interna
- `docs/specs/multimodal_data_architecture.md` - Arquitectura de datos
- `conf/vision/default.yaml` - Configuración ejemplo

### Librerías Clave
- [Ultralytics](https://docs.ultralytics.com/) - YOLO
- [Supervision](https://supervision.roboflow.com/) - Tracking
- [FastAPI](https://fastapi.tiangolo.com/) - API
- [Hydra](https://hydra.cc/) - Configuración
- [Pydantic](https://docs.pydantic.dev/) - Validación

### Patrones de Diseño
- [Refactoring Guru](https://refactoring.guru/design-patterns) - Patrones
- [Python Patterns](https://python-patterns.guide/) - Implementaciones en Python

---

## 🔐 Reglas de Seguridad

### NO incluir en commits:
- ❌ API keys o tokens
- ❌ Credenciales de DB
- ❌ Datos personales
- ❌ Archivos grandes (>10MB)
- ❌ Modelos entrenados (usar Git LFS o storage externo)

### Usar variables de entorno:
```python
# ✅ CORRECTO
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/db")

# ❌ INCORRECTO
DATABASE_URL = "postgresql://user:password@host/db"
```

---

## 📊 Métricas de Calidad

### Objetivos del Proyecto
| Métrica | Objetivo | Actual |
|---------|----------|--------|
| Test Coverage | ≥ 80% | 75% |
| Type Coverage | 100% | 95% |
| Cyclomatic Complexity | < 10 | 4 |
| Code Duplication | < 3% | 1% |
| Documentation Coverage | ≥ 80% | 70% |

### Monitoreo Continuo
```bash
# Generar reporte de métricas
pytest --cov=src --cov-report=term
mypy src/ | grep "Success"
radon cc src/ -a -nb
```

---

**Nota Final:** Este documento es la fuente de verdad para el desarrollo. Si encuentras inconsistencias entre este archivo y el código, prioriza las directrices de este archivo y actualiza el código.

**Última revisión:** Claude Sonnet 4 - 2025-11-26