# Modelo de Datos — CerebroVial

> Referencia técnica del schema de base de datos. Para entender
> **por qué** está así y qué datos llena cada tabla, ver
> `DATA_MODEL_AUDIT.md`.

## Diagrama Entidad-Relación

```mermaid
erDiagram
    IntersectionDB ||--o{ CameraDB : "accesorio"
    IntersectionDB ||--o{ IntersectionEdgeDB : "puente"
    GraphEdgeDB ||--o{ IntersectionEdgeDB : "referenciada por"
    GraphNodeDB ||--o{ GraphEdgeDB : "source"
    GraphNodeDB ||--o{ GraphEdgeDB : "target"
    GraphEdgeDB ||--o{ WazeJamDB : "congestión en"
    GraphEdgeDB ||--o{ WazeAlertDB : "evento en"
    GraphEdgeDB ||--o{ VisionFlowDB : "flujo desde"
    GraphEdgeDB ||--o{ VisionFlowDB : "flujo hacia"
    CameraDB ||--o{ VisionTrackDB : "captura"
    CameraDB ||--o{ VisionFlowDB : "captura"

    GraphNodeDB {
        string node_id PK
        float lat
        float lon
        bool has_camera
        geometry geom "POINT 4326"
    }

    GraphEdgeDB {
        string edge_id PK
        string source_node FK
        string target_node FK
        float distance_m
        int lanes
        geometry geom "LINESTRING 4326"
    }

    IntersectionDB {
        string intersection_id PK
        string junction_id
        float lat
        float lon
        string los_pmu "nullable"
        string tls_id "nullable"
        geometry geom "POINT 4326"
    }

    IntersectionEdgeDB {
        string intersection_id PK_FK
        string edge_id PK_FK
        string direction "incoming|outgoing"
    }

    CameraDB {
        string camera_id PK
        string intersection_id FK "nullable"
        string stream_url "nullable"
        float lat
        float lon
        float heading
        float fov
        geometry geom "POINT 4326"
    }

    WazeJamDB {
        string event_uuid PK
        datetime snapshot_timestamp PK
        string edge_id FK "nullable"
        float speed_mps
        int delay_seconds
        int congestion_level "1-5"
        int jam_length_m
        int road_type
        geometry geom "LINESTRING 4326"
    }

    WazeAlertDB {
        string alert_uuid PK
        datetime timestamp PK
        string edge_id FK "nullable"
        string alert_type
        string alert_subtype "nullable"
        int reliability "0-10"
        int confidence "0-10"
        geometry geom "POINT 4326"
    }

    VisionTrackDB {
        string track_uuid PK
        datetime entry_timestamp PK
        string camera_id FK
        datetime exit_timestamp
        int class_id "COCO id"
        float avg_speed_px
        geometry geom "LINESTRING 4326"
    }

    VisionFlowDB {
        string flow_id PK
        datetime timestamp_bin PK
        string camera_id FK
        int period_seconds
        string from_edge_id FK "nullable"
        string to_edge_id FK "nullable"
        string turn_direction "nullable"
        int vehicle_count
        float avg_speed_mps "nullable"
    }
```

## Tablas

### `graph_nodes` — Intersecciones

Una fila por **intersección física** del grafo vial. El sistema modela
la red vial como un grafo dirigido donde los nodos son cruces de calles.

| Columna | Tipo | Notas |
|---|---|---|
| `node_id` | string PK | ID legible (ej. `"larco_diagonal"`) |
| `lat`, `lon` | float | Coordenadas WGS84 |
| `has_camera` | bool | True si esta intersección tiene cámara YOLO |
| `geom` | POINT 4326 | Geometría espacial para queries PostGIS |

**Quién la llena:** seed inicial (`scripts/seed.py`, en E5). Estática
después.

**Quién la lee:** frontend (mapa de intersecciones), motor de control
(busca planes semafóricos), GRU (lookup de geografía).

### `graph_edges` — Calles (aristas)

Una fila por **calle dirigida** entre dos nodos. Una calle de doble
sentido se modela como **dos aristas**.

| Columna | Tipo | Notas |
|---|---|---|
| `edge_id` | string PK | |
| `source_node`, `target_node` | string FK → `graph_nodes` | Define la dirección |
| `distance_m` | float | Largo en metros |
| `lanes` | int | Número de carriles |
| `geom` | LINESTRING 4326 | Geometría de la calle |

**Quién la llena:** seed inicial. Estática después.

**Quién la lee:** Waze (los jams y alertas se asocian a un `edge_id`),
visión (los flujos turning conectan `from_edge_id` con `to_edge_id`),
control adaptativo, GRU.

### `intersections` — Intersecciones semaforizadas del PMU (Fase A)

Una fila por **intersección semaforizada** del Plan de Movilidad Urbana de
Miraflores. Entidad de primera clase (D-016): el control, las cámaras y el
mapeo al grafo cuelgan de acá. Fuente: `documentation/contracts/mapeo_pmu_edges_v2.yaml`.

| Columna | Tipo | Notas |
|---|---|---|
| `intersection_id` | string PK | = `nombre` del mapeo (ej. `"larco_benavides"`) |
| `junction_id` | string | ID del junction SUMO (puede ser un `cluster_...`) |
| `lat`, `lon` | float | `coord_gazetteer` (WGS84) |
| `los_pmu` | string, nullable | Nivel de servicio del PMU (ej. `"C/D"`); NULL si el PMU no lo fija |
| `tls_id` | string, nullable | ID del TLS SUMO. Poblado SOLO si está verificado (hoy: `larco_benavides`). DEUDA-CTRL-TLS |
| `geom` | POINT 4326 | Sin índice GIST (D-016) |

**Quién la llena:** `scripts/seed_intersections.py` (las 11 semaforizadas; excluye
`ovalo_gutierrez`, rotonda sin TLS). **NO** hay FK a `graph_nodes`: `junction_id`
es opaco; el puente al grafo va por `intersection_edges`.

**Quién la lee:** `/api/intersections` (vía `cameras`), trabajos de control futuros.

### `intersection_edges` — Puente intersección → arista (Fase A)

Mapea cada intersección a sus aristas del grafo, con dirección. PK compuesta.

| Columna | Tipo | Notas |
|---|---|---|
| `intersection_id` | string FK → `intersections` | PK |
| `edge_id` | string FK → `graph_edges` | PK. ID SUMO crudo (ej. `129466113#3`) |
| `direction` | string | `'incoming'` \| `'outgoing'` (CHECK) |

**Quién la llena:** `scripts/seed_intersections.py`, desde `edges_incoming`/
`edges_outgoing` del mapeo. **Requiere el net real cargado** (`scripts/build_graph_geometry.py`)
porque `edge_id` es FK→`graph_edges`. Orden: `invoke seed` → `build_graph_geometry.py`
→ `invoke seed-intersections` (pre-check fail-fast si falta el net).

### `cameras` — Cámaras de visión (accesorio de intersección)

Una fila por cámara desplegada. Desde Fase A, la cámara es **accesorio de una
intersección** (ya no ancla a `graph_nodes`).

| Columna | Tipo | Notas |
|---|---|---|
| `camera_id` | string PK | |
| `intersection_id` | string FK → `intersections`, nullable | Reemplaza a `node_id` (Fase A) |
| `stream_url` | string, nullable | HLS de Claro. Asociación cámara↔intersección NOMINAL — DEUDA-CAM-GEO |
| `lat`, `lon` | float | Coordenadas WGS84 (hoy = coord de la intersección, nominal) |
| `heading` | float | Ángulo de orientación 0-360° |
| `fov` | float | Field of view en grados |
| `geom` | POINT 4326 | |

**Quién la llena:** `scripts/seed_intersections.py` (1 cámara por intersección,
`stream_url` de Claro asignado 1:1 arbitrariamente).

**Quién la lee:** módulo de visión (`edge_device`), frontend (mapa de cámaras y
stream), `/api/intersections`.

### `waze_jams` — Snapshots de congestión de Waze

Serie temporal: cada snapshot es un punto de datos para un jam activo.
Mismo `event_uuid` puede tener múltiples snapshots a lo largo del
tiempo. Candidata a hypertable de TimescaleDB (E3).

| Columna | Tipo | Notas |
|---|---|---|
| `event_uuid` | string PK | ID del jam |
| `snapshot_timestamp` | datetime PK | Momento del snapshot |
| `edge_id` | string FK → `graph_edges`, nullable | Nullable porque jams fuera del grafo se descartan al asociar |
| `speed_mps` | float | Velocidad media |
| `delay_seconds` | int | Demora respecto a flujo libre |
| `congestion_level` | int | 1-5, **clasificación de Waze, ground truth para GRU** |
| `jam_length_m` | int | Metros de cola |
| `road_type` | int | Tipo de calle según Waze |
| `geom` | LINESTRING 4326 | Segmento congestionado |

**Quién la llena:** ingestor de Waze API (futuro, fuera de alcance
actual); para entrenamiento, dataset sintético generado en F2.

**Quién la lee:** GRU (input + ground truth), frontend (heatmap de
congestión), motor de control.

### `waze_alerts` — Alertas puntuales de Waze

Distintas a los jams. Eventos puntuales reportados por usuarios
(accidentes, peligros, policía, calle cerrada). Candidata a hypertable.

| Columna | Tipo | Notas |
|---|---|---|
| `alert_uuid` | string PK | |
| `timestamp` | datetime PK | |
| `edge_id` | string FK, nullable | |
| `alert_type` | string | `"ACCIDENT"`, `"HAZARD"`, `"ROAD_CLOSED"`, `"POLICE"`, `"JAM"` |
| `alert_subtype` | string nullable | Subcategoría más específica |
| `reliability` | int | 0-10, calculado por Waze |
| `confidence` | int | 0-10, calculado por Waze |
| `geom` | POINT 4326 | |

**Quién la llena:** ingestor de Waze API (futuro). Opcional para tesis.

**Quién la lee:** GRU como feature contextual (opcional), frontend.

### `vision_tracks` — Trayectorias individuales (modelada, no llenada)

Una fila por vehículo individual detectado y trackeado. Candidata a
hypertable.

| Columna | Tipo | Notas |
|---|---|---|
| `track_uuid` | string PK | |
| `entry_timestamp` | datetime PK | |
| `camera_id` | string FK → `cameras` | |
| `exit_timestamp` | datetime | |
| `class_id` | int | ID de clase COCO (2=car, 3=motorcycle, 5=bus, 7=truck) |
| `avg_speed_px` | float | **En píxeles/s, requiere calibración para m/s** |
| `geom` | LINESTRING 4326 | Trayectoria del vehículo |

**Quién la llena:** **nadie en alcance actual.** Modelada para futuro
trabajo de integración del pipeline de visión a BD.

**Quién la lee:** futuro. No usado por el GRU en alcance actual.

### `vision_flows` — Flujos turning por arista (modelada, no llenada)

Agregados de movimientos turning en intersecciones. Candidata a
hypertable.

| Columna | Tipo | Notas |
|---|---|---|
| `flow_id` | string PK | |
| `timestamp_bin` | datetime PK | Inicio de la ventana de agregación |
| `camera_id` | string FK | |
| `period_seconds` | int | Duración de la ventana |
| `from_edge_id` | string FK, nullable | Arista de origen del giro |
| `to_edge_id` | string FK, nullable | Arista de destino del giro |
| `turn_direction` | string nullable | `"left"`, `"right"`, `"straight"`, `"u-turn"` |
| `vehicle_count` | int | Cantidad de autos en este flujo |
| `avg_speed_mps` | float nullable | Velocidad media (calibrada) |

**Quién la llena:** **nadie en alcance actual.** Modelada para futuro
trabajo. El control adaptativo eventualmente usará esta información
para ajustar fases de semáforos.

**Quién la lee:** futuro.

### `vision_aggregates` — Persistencia BD de los datos del CSV (E18-E21)

> ⚠️ Tabla **a crear en E18**. No existe todavía.

Schema alineado con `csv_repository.py` para que la persistencia a
BD funcione sin refactor del pipeline de visión:

| Columna | Tipo | Notas |
|---|---|---|
| `id` | uuid PK | |
| `timestamp` | datetime, indexed | |
| `camera_id` | string FK → `cameras` | |
| `street_monitored` | string | Free-form, no FK al grafo |
| `car_count`, `bus_count`, `truck_count`, `motorcycle_count` | int | Conteo por tipo |
| `total_vehicles` | int | |
| `occupancy_rate` | float | |
| `flow_rate_per_min` | float | |
| `avg_speed` | float nullable | Sin unidad explícita en CSV; documentar |
| `avg_density` | float | |
| `zone_id` | string | |
| `duration_seconds` | float | Duración de la ventana de agregación |

**Quién la llena:** `PostgresAggregateRepository` (a implementar en
E19) recibe los mismos `TrafficData` que hoy van a CSV.

**Quién la lee:** dashboard del frontend (KPIs de visión por cámara),
demos de defensa.

### `predictions` — Registro de predicciones GRU (CT-09.5)

> Tabla creada por la migración
> `core_management_api/alembic/versions/f2c9d7a4b6e1_add_predictions_table.py`
> (TTH-09 Fase 5). **Append-only**, grano fila-por-paso: una fila por
> (inferencia × dirección × paso); una llamada a `/predictions/predict`
> produce 4 direcciones × 30 pasos = **120 filas**. Tabla relacional simple
> (NO hypertable).

El modelo es un **clasificador** (D-009, jam level 0-5): se persiste el `level`
discreto (`argmax`) y el vector `probs`, **no** un ratio continuo (obsoleto; ver
`documentation/contracts/prediction_contract.md` §6/§8, Nota D-005).

| Columna | Tipo | Notas |
|---|---|---|
| `prediction_id` | string(36) PK, default uuid4 | indexado |
| `intersection_id` | string | Opaco del request; **sin FK** al grafo (contrato §8) |
| `direction` | string | `"N"` / `"S"` / `"E"` / `"W"` |
| `step` | int | Paso de horizonte 1..30 (t+1 … t+30) |
| `level` | int | Nivel discreto 0..5 (`argmax` de `probs`) |
| `probs` | JSONB/JSON | `list[float]` de 6 probabilidades softmax |
| `model_version` | string | Versión del modelo; discrimina principal vs respaldo (HU-20) |
| `generated_at` | datetime tz-aware | ≡ `generated_at` de la response (contrato §8) |

Índice compuesto `ix_predictions_intersection_id_generated_at` sobre
`(intersection_id, generated_at)` para la consulta por intersección dentro de
una ventana temporal (HU-14).

**Quién la llena:** el handler `POST /predictions/predict`
(`PredictionsRepo.insert_batch`), de forma **best-effort** — si la DB falla, se
loguea y la predicción se devuelve igual (la persistencia no bloquea la respuesta).

**Quién la lee:** HU-14 (métricas del modelo). El join predicho-vs-real, la
observación real y la ventana de 24h son scope de HU-14, no de TTH-09.

> ⚠️ **Gap preexistente (no resuelto en esta pasada):** las tablas
> `motor_decisions`, `engine_active_state` (migración `b1f7c4d2a890`) y `users`
> (migración `99319147948b`) tampoco están documentadas en este archivo; el doc
> quedó desactualizado tras esas migraciones. Fuera del scope de TTH-09 Fase 5;
> se deja anotado para una pasada futura de sincronización del schema canónico.

## Hypertables (TimescaleDB)

En E3 se convierten las siguientes tablas en hypertables (chunk
time-based):

- `waze_jams` — partition por `snapshot_timestamp`
- `waze_alerts` — partition por `timestamp`
- `vision_tracks` — partition por `entry_timestamp` (aunque vacía hoy)
- `vision_flows` — partition por `timestamp_bin` (aunque vacía hoy)
- `vision_aggregates` — partition por `timestamp` (cuando se cree, E18+)

`chunk_time_interval`: a definir en E3. Default de TimescaleDB
(7 días) probablemente sirve. Para datasets sintéticos chicos se
puede ajustar a 1 día para tener más granularidad de chunks.

## Índices espaciales (PostGIS)

Todos los campos `geom` tienen índice GIST automáticamente (vía
`Geometry()` de GeoAlchemy2). Esto permite queries espaciales
eficientes:

- "intersecciones dentro de 500m" — `ST_DWithin`
- "jams que cruzan esta zona" — `ST_Intersects`
- "asignar cámara a la intersección más cercana" — `ST_Distance`

> **Nota (Fase A / D-016):** `intersections.geom` se declara con
> `spatial_index=False` — **sin** índice GIST. Las queries `ST_DWithin` corren
> igual (validado en `tests/intersections/test_spatial_e2e.py`), solo sin el
> índice; crearlo queda fuera del scope de Fase A. (En la práctica, los índices
> GIST del schema están comentados desde la migración inicial `775d2d1db8b4`.)

## Tablas internas de PostGIS

PostgreSQL con PostGIS instalado tiene tablas internas
(`spatial_ref_sys`, `layer`, `topology`) que NO son del modelo de
CerebroVial. La configuración de Alembic en `env.py` excluye estas
tablas del autogenerate vía `include_object` callback (configurado
en E2).