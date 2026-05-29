# `simulation/` — Módulo SUMO de CerebroVial (TTH-07)

Topología, escenarios de demanda, generador de dataset, adaptador TraCI
↔ motor adaptativo y baseline Webster fijo para la validación
cuantitativa del sistema integrado.

CTs cubiertos: 07.1 (topología), 07.2 (patrones + cobertura jam),
07.3/07.4 (dataset + particiones), 07.5 (TraCI ↔ motor), 07.6 (KPIs
comparativos), 07.7 (este README + scripts), 07.8 (tests integración).

## Quick start (CT-07.7)

### Setup (macOS arm64 / Linux manylinux)

```bash
cd simulation
python3.11 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install -e ".[dev]"   # opcional: pytest + pytest-cov
cp .env.example .env                 # ENGINE_URL + SUMO_HOME
```

El wheel `eclipse-sumo==1.26.0` es autocontenido (binarios + tools/ +
data/) — `SUMO_HOME=$(pwd)/.venv/lib/python3.11/site-packages/sumo` y
el framework system-wide NO se usa.

### Configurar SUMO_HOME y PATH

Cualquier script en `scripts/` cabla estos valores automáticamente.
Manualmente:

```bash
export SUMO_HOME="$(pwd)/.venv/lib/python3.11/site-packages/sumo"
export PATH="$SUMO_HOME/bin:$PATH"
```

### Reproducir la red (CT-07.1)

```bash
.venv/bin/python scripts/build_network.py
# → conf/network/miraflores_4way.net.xml
# → conf/network/miraflores_4way.tllogic.add.xml (6 sub-fases SUMO baseline)
# → conf/network/linkstates.json (derivado por introspección sumolib)
```

Parámetros editables en `conf/network/network_params.yaml` (carriles,
largos, vmax, fases del semáforo).

### Ejecutar un patrón (CT-07.2)

```bash
export SUMO_HOME=... PATH=$SUMO_HOME/bin:$PATH
sumo -c conf/scenarios/am_peak.sumocfg --seed 1
```

Los 4 patrones disponibles: `am_peak`, `pm_peak`, `offpeak`, `weekend`.
Flujos editables en `conf/scenarios/pattern_params.yaml`. Regenerar las
rutas tras editar:

```bash
.venv/bin/python scripts/generate_routes.py
```

### Verificar cobertura jam level

```bash
.venv/bin/python -m cerebrovial_simulation.coverage_check
# → reporta jam ≥3 sostenido por patrón, spillback, cola estabilizada.
```

### Regenerar el dataset (CT-07.3 + CT-07.4)

```bash
bash scripts/regenerate_dataset.sh           # full (100 corridas)
bash scripts/regenerate_dataset.sh --quick   # 8 corridas (smoke)
```

Output: `data/train/{pattern}_seed{N}.parquet` (80 archivos) +
`data/valid/{pattern}_seed{N}.parquet` (20 archivos). Sin overlap por
construcción (seeds 1-20 train, 21-25 valid). Schema declarado en
`src/cerebrovial_simulation/dataset/schema.py`.

### Ejecutar e2e con motor adaptativo (CT-07.5)

Requiere el core `core_management_api` corriendo en otra terminal:

```bash
# Terminal 1 (desde la raíz del repo)
invoke up

# Terminal 2
cd simulation
bash scripts/run_e2e_with_engine.sh --pattern am_peak --seed 1 --end 600
```

El script valida `curl /control/health` antes de arrancar. Si el motor
no responde, falla rápido con instrucciones.

### Comparación adaptive vs fixed-Webster (CT-07.6)

```bash
# Terminal 1
invoke up

# Terminal 2
bash scripts/run_comparison.sh           # 3 seeds × 4 patrones × 2 modes = 24 corridas
bash scripts/run_comparison.sh --quick   # 1 seed × 4 × 2 = 8 corridas (smoke)
bash scripts/run_comparison.sh --skip-adaptive  # solo fixed-Webster (no motor)
```

Output: `data/kpis/comparison.csv` con columnas
`(mode, pattern, seed, sim_duration_s, mean_travel_time_s,
total_delay_s, throughput_veh_per_h, max_queue_m_{N,S,E,W},
mean_queue_m_{N,S,E,W})`.

### Tests (CT-07.8)

```bash
.venv/bin/pytest tests/ -v
# 27 tests: F1 topología (a1-a8), F2 patrones (b1×4, b2), F3 dataset (c1-c5),
# F4 TraCI e2e (d1-d5), F5 Webster + KPIs (e1-e4).
```

Los tests F4 usan engine mockeado (no requieren `invoke up`). El smoke
manual de F4 con motor real está documentado en
[handoffs/tth-07/tth-07-fase4-handoff.md](../documentation/handoffs/tth-07/tth-07-fase4-handoff.md).

## Arquitectura

### Estructura

```
simulation/
├── conf/
│   ├── network/                    Topología generada + linkstates + detectores
│   │   ├── network_params.yaml     parámetros editables
│   │   ├── miraflores_4way.net.xml red SUMO
│   │   ├── miraflores_4way.tllogic.add.xml  programa baseline (6 sub-fases)
│   │   ├── webster_fixed.add.xml   programa fixed-Webster precomputado
│   │   ├── edgedata.add.xml        meandata por edge
│   │   ├── lanearea.add.xml        10 detectores E2 (3+3+2+2)
│   │   └── linkstates.json         derivado via sumolib introspección
│   └── scenarios/
│       ├── pattern_params.yaml     flujos por patrón
│       ├── {pattern}.sumocfg       4 patrones
│       └── routes/                 .rou.xml generados
├── src/cerebrovial_simulation/
│   ├── jam_level.py                D-009 canónico
│   ├── coverage_check.py           Verifica CT-07.2 (jam ≥3 / ≤2 + spillback)
│   ├── dataset/                    Generación + esquema + particiones
│   ├── traci_adapter/              TraCI ↔ motor HTTP
│   │   ├── state_reader.py         flow = tasa de paso (Catch C)
│   │   ├── phase_aggregator.py     4 dirs → 2 PhaseFlow (Option A)
│   │   ├── engine_client.py        POST /control/recommend
│   │   ├── tllogic_applier.py      Catch A: aplica en borde de ciclo
│   │   └── run_e2e.py
│   ├── fixed_control/
│   │   └── webster_fixed.py        Catch B: Webster offline directo
│   └── kpis/                       KPIs comparativos
├── scripts/                        CLIs reproducibles
├── tests/                          27 tests pytest
└── data/                           Outputs gitignored
```

### Decisiones lockeadas

- **Pin exacto**: `eclipse-sumo==1.26.0`, `traci==1.26.0`.
  `sumolib==1.27.0` transitivo (gap PyPI 1.22-1.26). `libsumo`
  NO se usa (gap PyPI también; opción C del handoff F0).
- **Topología**: genérica vía `netconvert` (NO OSM). Larco × Schell
  como referencia documentada de parámetros, no geometría.
- **tlLogic**: Option A — 2 fases NS+EW lefts permissive. Linkstates
  derivados por `sumolib` introspección, NO hardcodeados.
- **Motor**: HTTP externo en `http://localhost:8001/control/recommend`
  (puerto lockeado en el core_management_api). Cero acoplamiento de
  código.
- **Transporte SUMO**: TraCI cross-process.
- **Dataset**: Parquet single-write OK; outputs multi-interval
  (edgeData, laneArea) usan XML como fallback documentado (SUMO 1.26
  arrow writer no soporta append) y se convierten a Parquet final via
  pyarrow downstream.

### Contratos referenciados

- [D-009](../documentation/lean-inception/4-decisiones/DECISIONS.md#L183-L279)
  — variable de estado predicha (jam level 0-5).
- [engine_recommend_contract.md](../documentation/contracts/engine_recommend_contract.md)
  — shape canónico del motor.

### Trabajos futuros / fuera de scope MVP1

- **Turns en la demanda**: routes son straight-only en F2. Refinamiento
  posterior fuera de scope.
- **libsumo in-process** como fallback de throughput si TraCI no
  alcanza (decisión gated por measurement; opción C del F0 lo
  diferió).
- **Calibración fina del lost_time** por número de fases si el motor se
  extiende a 4-fase split-phase u otras tlLogic.
- **Δt_in del dataset** (CT-07.3) provisional en 60 s — cierre formal
  por TTH-11 CT-11.8.
