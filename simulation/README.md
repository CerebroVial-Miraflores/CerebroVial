# `simulation/` — Módulo SUMO de CerebroVial (TTH-07)

Topología, escenarios de demanda, generador de dataset, adaptador TraCI
↔ motor adaptativo y baseline Webster fijo para validación cuantitativa.

> Este README es un esbozo de F1. La versión final reproducible por un
> tercero (CT-07.7) se completa en F6.

## Setup rápido (macOS arm64 / Linux manylinux)

```bash
cd simulation
python3.11 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env  # editar si SUMO_HOME del wheel no resuelve
```

## Decisiones y deps

- Pin: `eclipse-sumo==1.26.0`, `traci==1.26.0`. `libsumo` no se usa (gap
  PyPI 1.26 documentado en handoff F0).
- Wheel `eclipse-sumo` es autocontenido (bin/, tools/, data/). `SUMO_HOME`
  apunta al wheel del venv, no al framework system-wide.
- Motor por HTTP externo (`ENGINE_URL=http://localhost:8001/control/recommend`),
  cero acoplamiento de código con `core_management_api/`.
