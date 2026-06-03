# TTH-12 — Replay de congestión por arista (receta reproducible)

> Fase 2 de TTH-12. Documenta el día usado para el feed de congestión y cómo
> regenerar el dataset (el Parquet está gitignored). CT-12.4.

## Día usado: `seed062`

Elegido en el perfilado de los 60 días (`miraflores_laborable_60d`) por **amplitud de
transición de niveles**: es el día con más aristas que alcanzan congestión marcada.

| Métrica (sobre las 375 aristas LCC) | seed062 |
|---|---|
| Aristas que alcanzan jam ≥ 3 en algún paso | **219** (el máximo de los 60 días) |
| % de buckets con jam ≥ 3 | 3.77% |
| Pico de congestión | 35 aristas @ ts=29160 s (≈ 8 h, hora punta AM) |
| Histograma de niveles (lvl0..5) | 492275 / 17839 / 9509 / 8339 / 10702 / 1336 |

El día es **parámetro** del CLI (`--day seed062` es solo el default); la lógica del
adaptador no lo hardcodea.

## Fuente del Parquet

- Ruta (gitignored, regenerable): `simulation/data/datasets/miraflores_laborable_60d/day_seed062.parquet`
- Columnas: `edge_id, timestep, speed, timeLoss, traveltime, flow, density, speedRelative`
- Grano: 375×1440 = 540 000 filas/día = 375 aristas × 1440 pasos de 60 s (24 h).
- Universo de aristas: las **375 del LCC**, alineadas con
  `ia_prediction_service/src/data/artifacts/miraflores_graph_lcc_mapping.json` (`sumo_edge`).

### Regeneración del dataset
El dataset por arista se produce con el pipeline de simulación de TTH-07
(`simulation/`), recorriendo los seeds del perfil laborable. Ver
`simulation/src/cerebrovial_simulation/dataset/` (builder por arista) y
`documentation/handoffs/corredor-larco/` para la receta de demanda B1. El día
`seed062` corresponde al seed 62 de esa corrida.

## Mapeo a `waze_jams` (V1, SUMO como stand-in del feed de Waze)

| Columna waze_jams | Origen |
|---|---|
| `edge_id` | `edge_id` SUMO (FK → `graph_edges`, las 375 de Fase 1) |
| `snapshot_timestamp` | `DAY_EPOCH (2025-01-06 00:00) + timestep s` |
| `congestion_level` | `ratio_to_jam_level(speedRelative)` (0-5, D-009); **NaN → 0** (sin observación = flujo libre) |
| `speed_mps` | `speed`; si sin observación → velocidad de flujo libre de la arista (`.net.xml`) |
| `delay_seconds` | `round(timeLoss)`; sin observación → 0 |
| `event_uuid` | `uuid5(ns, "edge_id|timestamp")` — determinista (pre-siembra idempotente) |
| `geom` | la de `graph_edges` (UPDATE-join por `edge_id`) |
| `jam_length_m` | **centinela −1** — sin fuente en el feed SUMO V1 (ver DHU-028) |
| `road_type` | **centinela 0** — idem |

**Regla del NaN:** `ratio_to_jam_level` con NaN crudo devolvería 4 (incorrecto). El
adaptador intercepta "sin observación" (`speedRelative` NaN; el dataset lo emite con
`sampledSeconds=0` y sin `speed`) ANTES de llamar a la función → nivel 0.

## Uso

```bash
# Pre-siembra (batch, 540k filas, idempotente):
.venv/bin/python scripts/replay_congestion.py --mode presiembra --day seed062

# Replay en vivo (gotea a cadencia; speedup 60 = 1 paso/seg):
.venv/bin/python scripts/replay_congestion.py --mode vivo --day seed062 --speedup 60
```

Código: `core_management_api/src/congestion/` (interfaz de feed CT-12.3 +
repositorio `waze_jams` CT-12.5 + adaptador de replay CT-12.4). La integración con
el broadcaster SSE y los endpoints es Fase 3.

## Deuda registrada (no se arregla en esta TTH)
- `congestion_level` no tiene CHECK en BD; el rango efectivo del sistema es **0-5**
  (D-009). El "1-5" del DATA_MODEL heredado es obsoleto → saneamiento documental futuro.
- `jam_length_m` / `road_type` en centinela (−1 / 0): sin fuente en el feed SUMO; el
  ingestor de Waze real los traerá.

---

> **Forward-note (B3.2.e):** la cadena fue re-sembrada sobre el universo v2 (1660 aristas,
> `--day seed051`) en B3.2.e. Este documento refleja el estado de **TTH-12** (375 aristas,
> seed062) y se preserva como registro histórico de esa época; no se reescribe. Para el estado
> actual de la fuente, el universo y la alineación, ver `documentation/contracts/congestion_contract.md`.
