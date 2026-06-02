# Contrato del módulo de congestión por arista (TTH-12)

> Documento vivo del producto. Define la interfaz de feed, los endpoints HTTP y el
> canal SSE que habilitan HU-22 (mapa de congestión de la red). Habilita: geometría
> de red (CT-12.2), feed de estado por arista (CT-12.6), robustez (CT-12.7),
> alineación de IDs (CT-12.8). Fuente V1: replay de dataset SUMO (seed062).

## Unidad y alcance
- Unidad de análisis: **tramo de vía = arista del grafo** (`edge_id` SUMO, p. ej. `-129822384#0`).
- Universo: las **375 aristas del LCC** (`miraflores_graph_lcc_mapping.json`, campo `sumo_edge`).
- Nivel de congestión: entero **0-5** (escala Waze, D-009; 0 = flujo libre). La columna
  `waze_jams.congestion_level` no enforça rango — el rango efectivo del sistema es 0-5.

## Interfaz de feed de congestión por arista (CT-12.3)
`src/congestion/application/feed.py`. Punto de desacople de la FUENTE.

- `EdgeCongestion(edge_id: str, congestion_level: int 0-5, snapshot_timestamp: datetime)`.
- `CongestionFeed` (Protocol): `timesteps() -> list[int]`, `levels_at(timestep) -> list[EdgeCongestion]`,
  `snapshots() -> Iterable[EdgeCongestion]`.
- Implementación V1: `SumoReplayAdapter` (replay de un día del dataset; deriva con
  `ratio_to_jam_level`, NaN→0). Sustituir la fuente (SUMO en vivo vía TraCI, ingestor
  de Waze real) **no toca a los consumidores**: solo cambia quién escribe `waze_jams`.

## Endpoints HTTP
Ambos protegidos con `require_role(OPERATOR, ADMIN)` (TTH-01). Prefijo `/congestion`.

### `GET /congestion/geometry` — geometría de la red (CT-12.2)
Estática, cacheable. Lee `graph_edges` (las 375). Respuesta GeoJSON:

```json
{
  "type": "FeatureCollection",
  "count": 375,
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "LineString", "coordinates": [[-77.0335, -12.1180], ...]},
      "properties": {"edge_id": "-129822384#0", "source_node": "sumo_138854736",
                     "target_node": "sumo_262576671", "distance_m": 241.2, "lanes": 1}
    }
  ]
}
```
Geometría en **EPSG:4326**, orden GeoJSON **[lon, lat]**.

### `GET /congestion/state` — estado de congestión actual por arista (CT-12.6/12.7)
Dinámico. Lee el **último snapshot por arista** desde `waze_jams` (DISTINCT por `edge_id`,
el más reciente). Respuesta:

```json
{
  "count": 375,
  "edges": [
    {"edge_id": "-129822384#0", "congestion_level": 0, "snapshot_timestamp": "2025-01-06T23:59:00"}
  ]
}
```
**Robustez (CT-12.7):** nunca falla si la fuente (adaptador) está detenida — devuelve el
último estado conocido con su `snapshot_timestamp`. La decisión de "desactualizado"
(HU-22, CA-22.4) es del consumidor; el endpoint solo expone el timestamp.

### `GET /congestion/state/stream` — canal SSE de red (CT-12.6)
Patrón SSE-wake/REST-read (DHU-021 #15). Emite el evento `congestion-updated` **sin
payload** (wake-up de RED, no per-arista). El cliente, al despertar, **re-lee**
`GET /congestion/state` para el estado autoritativo. Implementado con
`NetworkCongestionBroadcaster` (singleton in-memory per-proceso, propio del dominio —
no reusa el de `control/`).

## Escritura del feed (replay)
`SumoReplayAdapter` + `WazeJamsRepo`, vía `scripts/replay_congestion.py`:
- **pre-siembra** (batch, 375×1440 idempotente; event_uuid uuid5 + ON CONFLICT).
- **replay en vivo** (gotea a cadencia; tras cada paso invoca `wake` → el broadcaster
  publica el wake-up SSE). La pre-siembra NO emite wakes.

Mapeo a `waze_jams` y regla del NaN: ver `documentation/docs/tth12_congestion_replay.md`.
Centinelas sin fuente en el feed SUMO V1: `jam_length_m = -1`, `road_type = 0` (DHU-028).

## Alineación de IDs end-to-end (CT-12.8)
`mapping.sumo_edge` == `graph_edges.edge_id` == `waze_jams` DISTINCT `edge_id` == **375**.
Verificable con `scripts/verify_edge_id_alignment.py` (375 = 375 = 375).

## Desacople de fuente (resumen)
| Hoy (V1) | Mañana |
|---|---|
| Replay de dataset SUMO pre-generado (seed062) escribe `waze_jams` | SUMO en vivo (TraCI) o ingestor de Waze real escribe `waze_jams` |
| Detrás de `CongestionFeed` | Misma interfaz; solo cambia el escritor |
Los consumidores (repositorio, endpoints, vista HU-22) no cambian.
