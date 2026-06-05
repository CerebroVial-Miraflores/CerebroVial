# Contrato del modelo de intersecciones — Fase A

> Documento vivo del producto. Define el modelo de datos de las intersecciones
> semaforizadas del PMU de Miraflores: schema de tablas (`intersections`,
> `intersection_edges`, `cameras`), semántica de campos, el puente al grafo vial y
> el alcance honesto de la siembra. Habilita: cámaras como accesorio de intersección
> (Fase B/C) y la futura fase de control sobre el resto de intersecciones.

**Estado al cierre de Fase A:** Parcial honesto — 11 intersecciones sembradas; control
poblado solo para `larco_benavides`; asociación cámara↔intersección nominal.
**Última actualización:** 2026-06-05 (Fase A).
**Autoridad:** `documentation/lean-inception/4-decisiones/DECISIONS.md` § D-016.
**Fuente de datos:** `documentation/contracts/mapeo_pmu_edges_v2.yaml`.

---

## 1. Modelo

### `intersections` — entidad de primera clase

Una fila por intersección **semaforizada** del PMU (11; excluye `ovalo_gutierrez`,
rotonda sin TLS). El control, las cámaras y el mapeo al grafo cuelgan de acá.

| Columna | Tipo | Semántica |
|---|---|---|
| `intersection_id` | string PK | = `nombre` del mapeo (ej. `larco_benavides`). ID estable y legible. |
| `junction_id` | string NOT NULL | ID del junction SUMO (puede ser `cluster_...`). **Opaco**: no hay FK a `graph_nodes`. |
| `lat`, `lon` | float NOT NULL | `coord_gazetteer` del mapeo (WGS84). |
| `los_pmu` | string nullable | Nivel de servicio del PMU (ej. `C/D`). NULL cuando el PMU no lo fija (ej. `ovalo_miraflores`, `benavides_panama`). |
| `tls_id` | string nullable | ID del TLS SUMO. Poblado **solo si verificado** (hoy: `larco_benavides`). DEUDA-CTRL-TLS. |
| `geom` | POINT 4326 | `spatial_index=False` — sin GIST (D-016). |

### `intersection_edges` — puente intersección → grafo

Mapea cada intersección a sus aristas, con dirección. PK compuesta
`(intersection_id, edge_id)`.

| Columna | Tipo | Semántica |
|---|---|---|
| `intersection_id` | string FK → `intersections` | Parte de la PK. |
| `edge_id` | string FK → `graph_edges` | Parte de la PK. **ID SUMO crudo** (ej. `129466113#3`). |
| `direction` | string NOT NULL | `'incoming'` \| `'outgoing'` (CHECK constraint). |

El puente al grafo es **por aristas**, no por nodo: `junction_id` queda opaco y la
resolución a la topología se hace `intersección → intersection_edges → graph_edges`.

### `cameras` — accesorio de intersección

Desde Fase A la cámara **ya no ancla a `graph_nodes`**. Pierde `node_id`; gana
`intersection_id` (FK→intersections) y `stream_url`.

| Columna | Tipo | Semántica |
|---|---|---|
| `camera_id` | string PK | `cam_<intersection_id>`. |
| `intersection_id` | string FK → `intersections`, nullable | Reemplaza a `node_id`. |
| `stream_url` | string nullable | HLS de Claro. Asociación **nominal** — DEUDA-CAM-GEO. |
| `lat`, `lon` | float | Hoy = coord de la intersección (nominal). |
| `heading`, `fov` | float | Defaults `0.0` / `90.0`. |
| `geom` | POINT 4326 | |

## 2. Siembra y orden de carga

El seed (`scripts/seed_intersections.py`) **requiere el net real cargado** porque
`intersection_edges.edge_id` es FK→`graph_edges` (ids SUMO que puebla
`scripts/build_graph_geometry.py`, no `invoke seed`). Orden:

```
invoke seed                         # 5 nodos de control + 6 edges de juguete + admin
python scripts/build_graph_geometry.py   # 1660 edges SUMO reales + 904 nodos sumo_
invoke seed-intersections           # 11 intersecciones + 67 intersection_edges + 11 cámaras
```

**Pre-check fail-fast:** si algún `edge_id` del mapeo no está en `graph_edges`, el seed
aborta con mensaje claro (corré `build_graph_geometry.py` primero) en vez de un error
críptico de FK. El builder, a su vez, aborta si `intersection_edges` ya está poblada
(su `DELETE FROM graph_edges` violaría la FK) — truncar el puente antes de re-cargar el net.

Idempotente (`session.merge`). Resultado validado: **11 / 67 / 11**, 1 con `tls_id`.

## 3. Consumidores

`GET /api/intersections` deriva el `name` de `cameras.intersection_id` (ya no de
`node_id`). El control (`motor_decisions`/`engine_active_state`) ancla a
`graph_nodes.node_id` y es **independiente** de este modelo — no se toca.

## 4. Alcance NO cubierto (deudas y fases siguientes)

- **DEUDA-CAM-GEO** — la asignación de los 11 `stream_url` de Claro a las intersecciones
  es **arbitraria 1:1** (por orden del mapeo), sin concordancia geográfica real. Pendiente:
  verificar y reasignar. (`documentation/docs/TODO.md`).
- **DEUDA-CTRL-TLS** — 10 de 11 intersecciones sin `tls_id`/nodo de control. El modelo lo
  soporta; falta poblarlo en una fase de control futura. `arequipa_angamos` es el caso
  "casi listo" (ya es nodo de control sembrado; falta verificar su `tls_id` SUMO).
- **Frontend (Fase B)** — `frontend_ui/src/components/views/CameraDetailView.tsx` tiene ids
  de cámara hardcodeados (`CAM_001`…) que no concuerdan con `cam_<intersection_id>`. Se
  ajusta en Fase B. No rompe la migración (usa `camera_id`, no `node_id`).
- **Edge / YOLO (Fase C)** — `edge_device/run_server.py` usa ids de cámara hardcodeados. Se
  ajusta en Fase C.
- **Índices GIST** — fuera de scope de Fase A.
