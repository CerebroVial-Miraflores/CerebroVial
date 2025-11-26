# Contexto del Proyecto: CerebroVial (Monolito Modular)

Este documento describe la arquitectura y estructura del proyecto para guiar a los agentes de IA.

## Arquitectura: Monolito Modular

El proyecto sigue una arquitectura de Monolito Modular, dividiendo la lógica en tres dominios principales dentro de un único repositorio.

### Estructura de Directorios

- **`data/`**: Almacenamiento de datos, separado por dominio.
  - `vision/`: Imágenes y videos (raw, interim, processed).
  - `prediction/`: Datos de sensores y tráfico (raw, interim, processed).
  - `control/`: Logs de control semafórico.

- **`src/`**: Código fuente, estructurado como paquetes Python.
  - **`common/`**: Utilidades compartidas y feature engineering reutilizable.
  - **`vision/`**: Lógica de Visión por Computadora (detección de vehículos).
    - `domain.py`: Entidades del dominio (e.g., `DetectedVehicle`).
    - `infrastructure/`: Adaptadores para cámaras/video.
    - `pipelines/`: Flujos de procesamiento.
  - **`prediction/`**: Lógica de Predicción de Congestión (modelos de series temporales).
    - `domain.py`: Entidades del dominio (e.g., `TrafficFlowData`).
    - `models.py`: Arquitecturas de modelos (LSTM, GNN).
  - **`control/`**: Lógica de Gestión Semafórica (Control).
    - `domain.py`: Entidades (e.g., `TrafficLightPhase`).
    - `services/`: Lógica de negocio para decisiones de control.

- **`models/`**: Artefactos de modelos entrenados (binarios), separados por dominio.
- **`conf/`**: Configuración centralizada (Hydra o similar), separada por dominio.

## Reglas de Desarrollo

1.  **Desacoplamiento**: Los módulos (`vision`, `prediction`, `control`) deben comunicarse a través de interfaces definidas en `domain.py` o `common/`. Evitar dependencias circulares.
2.  **Inmutabilidad de Datos**: Los datos en `data/raw` nunca deben modificarse.
3.  **Testing**: Cada módulo debe tener sus propias pruebas unitarias.
