# TomTom — Fase B-2: cierre (matching geométrico edge→OpenLR + persistencia)

**Rama:** `feature/tomtom`. **Alcance:** la lógica de backend que, dado un corredor (cadena
ordenada de aristas) y las respuestas de `flowSegmentData` que el **frontend** ya obtuvo de
TomTom, determina para cada arista a qué segmento OpenLR de TomTom pertenece, y persiste el
mapping en `corridor_edges` (tablas creadas en Fase B-1).

**Fuera de alcance (fases siguientes):** el front (dibujo, arrastre del corredor), la consulta
a TomTom (la hace el front), KPIs, vista y hover.

## Frontera con TomTom (ToS 11.4 / 11.6.1)

- **El backend NO consulta TomTom.** El front consulta `flowSegmentData` y le pasa al backend
  las respuestas (geometría + openlr de cada segmento). El backend sólo matchea geometría
  contra `graph_edges`.
- **La geometría de TomTom es INPUT EFÍMERO.** Entra al endpoint, se usa para el matching como
  bind param (`ST_GeomFromText(:wkt, 4326)`) en el query de overlap, y se descarta. Lo único
  que se persiste es `tomtom_openlr` (ID OpenLR, string) + `edge_id` + `sequence`.
- Verificación a nivel de código: un grep de `wkt`/`coordinates`/`ST_GeomFromText` sobre
  `corridors/` confirma que la geometría TomTom aparece sólo como bind param de un `SELECT`
  read-only y en construcción en memoria; nunca dentro de un `INSERT`/`UPDATE` ni en columna
  persistida. Los `add()` de persistencia escriben sólo IDs.

## Qué se entregó

Dominio nuevo `core_management_api/src/corridors/` (application / infrastructure / presentation),
espejo de `congestion/` y `control/`. `shared/.../models.py` **sin cambios** (las tablas ya
existían de Fase B-1).

1. **`application/geometry.py` (PURO, sin SQL/DB):** `bearing`, `angular_diff`, `same_direction`
   y `local_bearing`. El rumbo del segmento TomTom se calcula **LOCAL** al tramo de la polilínea
   más cercano al centro de la arista, no extremo→extremo: el segmento es largo (~4 km) y curvo,
   y un rumbo global mentiría donde la avenida dobla (empieza al N, termina al E → "NE" global),
   dando falso descarte/match en avenidas curvas.

2. **`application/matching.py`:** constantes nombradas y ajustables —`BUFFER_METERS = 15`,
   `MIN_OVERLAP_RATIO = 0.70`, `MAX_BEARING_DIFF_DEG = 90`—, `match_corridor` (el SENTIDO manda
   sobre el overlap: un segmento de mayor overlap pero sentido opuesto se descarta) y las
   validaciones puras de la cadena (`validate_sequences`, `validate_continuity`).

3. **`infrastructure/repositories.py`:** overlap geométrico en proyección métrica vía
   `::geography` (convención del repo) con guard de dialecto; lectura de la cadena por sus
   columnas `source_node`/`target_node` y los extremos vía `graph_nodes` (sin tocar `geom` → corre
   en SQLite); persistencia transaccional de sólo IDs. Fórmula del overlap:
   `ST_Length(ST_Intersection(edge.geom, ST_Buffer(line::geography, 15)::geometry)::geography) /
   ST_Length(edge.geom::geography)`.

4. **`presentation/api/` (`POST /corridors`):** `require_role(OPERATOR, ADMIN)`. Valida que las
   aristas existan en `graph_edges`, que la `sequence` sea contigua sin huecos/duplicados y que
   la cadena sea continua (`target_node` == `source_node` consecutivos) → 4xx claro **sin
   persistir cadena rota**. Corre el matching y devuelve el mapping `edge → openlr | null` para
   que el front sepa qué aristas quedaron sin cobertura. Registrado en `main.py`.

## Proyección métrica (decisión)

Se usa `::geography` (no reproyección a UTM) para que "15 m sea 15 m", siguiendo la convención
ya establecida en el repo (el único query métrico previo, `ST_DWithin` en
`test_spatial_e2e.py`, usa `geom::geography`). El buffer se construye en geography y se castea a
geometry para intersectar en 4326; las longitudes se miden en geography (metros reales).

## Verificación

- `pytest tests/` desde `core_management_api/`: **232 verde** (192 previos + 40 nuevos).
- **Unit (SQLite, corren siempre):** `bearing`/`local_bearing` parametrizados —incluido el caso
  curvo que prueba que el rumbo es local y no global—, `match_corridor` contra un repo falso (el
  sentido manda sobre el overlap; empate por mayor overlap; sin candidatos → `None`),
  validaciones de cadena, y el endpoint (éxito operator/admin, 403 rol no autorizado, 401 sin
  token, 422 cadena rota / sequence con huecos / arista inexistente, con asserción de que la
  cadena rota no persiste nada).
- **e2e PostGIS (marker `e2e`, skip-graceful sin Docker):** dos calzadas de sentidos opuestos
  separadas ~11 m, con buffers de 15 m que cubren AMBAS calzadas (el overlap por sí solo es
  ambiguo): el sentido desambigua. Cada calzada matchea el OpenLR de SU sentido y NO el del
  opuesto; una arista fuera de cobertura → `NULL`. Persistencia transaccional verificada contra
  el dialecto real.
- `ruff check` limpio sobre el código nuevo.
- Diff acotado: `main.py` +2 líneas (import + `include_router`); `corridors/` + `tests/corridors/`
  nuevos. `models.py`, `congestion/`, `control/` y el frontend intactos.

## Deudas y follow-ups

- **DEUDA-MATCHING-CALIBRACION (ampliada).** Además de calibrar los 3 umbrales
  (`BUFFER_METERS`/`MIN_OVERLAP_RATIO`/`MAX_BEARING_DIFF_DEG`) con corredores reales, vigilar el
  caso de **calzadas paralelas cercanas en óvalos/cruces** (las 11 intersecciones): puede no
  resolverse con buffer y requerir lógica de **desempate por mejor alineación de bearing**, no
  sólo el umbral binario. Probar primero sobre un **óvalo** (caso difícil) en Fase B-front. Hoy,
  ante empate de overlap entre candidatos del mismo sentido, gana el de mayor overlap; el
  desempate fino por bearing no está implementado. Anotada en `matching.py` junto a las
  constantes.
- **e2e con geometría sintética.** El e2e usa dos calzadas opuestas con coordenadas explícitas
  (autocontenido, sin depender de cargar el net SUMO) en lugar de las calzadas reales de
  `406008845#1`. Cubre la misma propiedad lógica —distinguir ida de vuelta por sentido—. Un e2e
  contra la geometría real queda como follow-up de calibración.
