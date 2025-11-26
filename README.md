# CerebroVial

Sistema inteligente de gestión de tráfico urbano basado en Visión por Computadora y Machine Learning.

## Estructura del Proyecto

Este proyecto utiliza una arquitectura de **Monolito Modular** para gestionar tres componentes interconectados:

1.  **Visión por Computadora (`src/vision`)**: Detecta y clasifica vehículos en tiempo real.
2.  **Predicción de Congestión (`src/prediction`)**: Predice niveles de tráfico futuros basándose en datos históricos y en tiempo real.
3.  **Control Adaptativo (`src/control`)**: Optimiza los ciclos semafóricos basándose en las predicciones.

## Organización de Carpetas

```
├── data/               # Datos separados por dominio (vision, prediction, control)
├── src/                # Código fuente modular
│   ├── common/         # Utilidades compartidas
│   ├── vision/         # Módulo de Visión
│   ├── prediction/     # Módulo de Predicción
│   ├── control/        # Módulo de Control
│   └── main.py         # Punto de entrada
├── models/             # Modelos entrenados
├── conf/               # Configuración
└── AGENTS.md           # Contexto para asistentes de IA
```

## Cómo Ejecutar

El punto de entrada principal es `src/main.py`.

```bash
# Ejecutar módulo de visión
python src/main.py vision

# Ejecutar módulo de predicción
python src/main.py prediction
```
