# TTH-08 — Cierre de Fase 9 + handoff de cierre del sprint

**Rama**: `feature/tth-08-fase9-docs` (desde `master@15bc6ca4` = merge PR #33, Fase 7).
**Fecha de cierre**: 2026-05-29.
**Estado al cierre**: **TTH-08 Parcial — F8 diferida, C7.6 reabierta como F9.z.**
**Restricciones honradas**: cero código productivo, cero tests nuevos. Único cambio fuera de `documentation/` es el `git mv` de `edge_device/conf/vision/javier_prado.yaml` (config muerta confirmada por F7) a `documentation/legacy/vision_configs/` con header explicando el contexto histórico.

---

## 1. Lo que Fase 9 entregó

Alcance de F9 según DHU-024 (`documentation/lean-inception/4-decisiones/DECISIONS_HU.md` línea 2539): *"Documentación contractual y cross-refs (cierre C7.6, F41, retiro de C1.x) — 0.5 SP"*. Entregables reales:

| # | Entregable | Archivo | Commit |
|---|------------|---------|--------|
| 9a | **Contrato canónico del módulo de visión** (DHU-024 §5) | `documentation/contracts/vision_contract.md` (nuevo) | `docs(tth-08): 9a — contrato canónico de visión` |
| 9b | **`javier_prado.yaml` movida a `documentation/legacy/`** (alt. iii) con header | `documentation/legacy/vision_configs/javier_prado.yaml` (movido + header) | `chore(tth-08): 9b — mueve javier_prado.yaml a documentation/legacy` |
| 9c | **TODO.md** — retira C1.5/C1.6/C1.7/C1.8, reabre C7.6 como F9.z, nomina C9.7/C9.8/F9.y/F9.z | `documentation/docs/TODO.md` | `docs(tth-08): 9c — retira C1.x, reabre C7.6 como F9.z y nomina C9.7/C9.8` |
| 9d | **Addendum F9 a DHU-024** con estado real al cierre (append-only) | `documentation/lean-inception/4-decisiones/DECISIONS_HU.md` | `docs(tth-08): 9d — addendum F9 a DHU-024` |
| 9e | **ESTIMACION_SP** — TTH-08 fila Parcial + F8 diferida + subtotal recalculado | `documentation/lean-inception/planificacion/ESTIMACION_SP.md` | `docs(tth-08): 9e — TTH-08 marcado como Parcial` |
| 9f | **Notas F41 reafirmadas** en EVOLUCION_TESIS §8 y LEAN_INCEPTION "Trabajos Futuros" | `documentation/lean-inception/1-contexto/EVOLUCION_TESIS.md` + `LEAN_INCEPTION_CEREBROVIAL.md` | `docs(tth-08): 9f — reafirma F41 como integración futura` |
| 9g | **Cross-refs forward** desde handoff F7 hacia F9 | `documentation/handoffs/tth-08/tth-08-fase7-handoff.md` | `docs(tth-08): 9g — cross-refs forward desde handoff F7 hacia F9` |
| 9h | **Este handoff** | `documentation/handoffs/tth-08/tth-08-fase9-handoff.md` (nuevo) | `docs(tth-08): 9h — handoff Fase 9 + cierre documental del sprint TTH-08` |

8 commits, 2 docs nuevos, 6 modificaciones, 1 `git mv` con header. Cero `.py`, cero `requirements*.txt`, cero Dockerfile, cero YAML de configuración activa.

---

## 2. Estado de los 11 CTs al cierre del sprint TTH-08

| CT | Descripción | Estado | Dónde se valida |
|----|-------------|--------|-----------------|
| **CT-08.1** | YOLO produce `list[DetectedVehicle]` | ✅ Validado | F4b — `test_yolo_detector.py` |
| **CT-08.2** | Conteo por zona / ROI con `mean_occupancy` (DHU-025) | ✅ Validado | F4a — `test_zone_counter_basic.py`, `test_zones.py` |
| **CT-08.3** | Métricas direccionales (`flow`, `density`, `mean_speed_kmh`) | ✅ Validado | F5b — `test_compute_traffic_data.py` (11 tests) |
| **CT-08.4** | Input modes (file/youtube/ip_cam/auto) | ✅ Validado | F4b — tests de fuentes + dispatcher |
| **CT-08.5** | Persistencia a `vision_aggregates` (mapping, NULLs, idempotencia) | ✅ Validado | F4c — `test_postgres_repository.py` |
| **CT-08.6** | Endpoint `GET /vision/state/{intersection_id}` + branch 5xx | ✅ Validado | F6e — `test_state_endpoint.py` |
| **CT-08.7** | Stream procesado (`/vision/streaming/...`) | ✅ Validado | F6 — suite de streaming |
| **CT-08.8** | Componente demostrable end-to-end | ✅ Validado | Smoke manual 4c |
| **CT-08.9** | Dataset etiquetado ≥200 frames + mAP/precisión/recall | ⏸️ **Diferido** | **F8 — diferida por decisión del usuario** |
| **CT-08.10** | Health check OK / Degradado / Fuera de servicio | ✅ Validado | F6f — `test_health.py` |
| **CT-08.11(a–d, f)** | Detección + asignación + métricas + endpoint + health | ✅ Validado | F4–F6 |
| **CT-08.11(e)** | Integración persistencia repo↔Postgres vivo (testcontainers) | ✅ Validado **acotado** | F7 — `test_persistence_e2e.py`. **Alcance: repo↔modelo↔Postgres vivo. NO valida migración↔modelo, NO pipeline-de-video end-to-end.** Ver handoff F7 §4.1–§4.2. |

**Resumen**: 10 de 11 CTs validados. CT-08.9 diferido. TTH-08 = **Parcial — F8 diferida**.

---

## 3. Diferimientos y deuda heredada (cierre honesto)

### 3.1 F8 diferida — CT-08.9 (validación de detección)

**Decisión del usuario, registrada en handoff F7 §6.1 línea 157**: no se etiqueta dataset, no se mide precisión/recall/mAP dentro del sprint TTH-08. F8 queda fuera del sprint como sub-fase de **datos** (etiquetado manual con Roboflow/CVAT/labelImg), ~1.5 SP estimados.

**Implicancia sobre el 88.2%** (D-005, `DECISIONS.md` líneas 70–79):
- El valor 88.2% es el número aspiracional declarado en la spec original.
- **No tiene sustento reproducible al cierre de TTH-08**: no hay dataset etiquetado ni medición cuantitativa propia.
- Será **sustituido por el mAP real** medido cuando corra F8. Si la realidad medida es peor que 88.2%, se reporta la realidad (per D-005).
- Mientras tanto, **no debe afirmarse como validado** en docs derivados — debe declararse como "pendiente de medición, ver F8".

Estado del diferimiento registrado en:
- `documentation/handoffs/tth-08/tth-08-fase7-handoff.md` §6.1 (origen)
- `documentation/lean-inception/4-decisiones/DECISIONS_HU.md` addendum F9 al pie de DHU-024
- `documentation/lean-inception/planificacion/ESTIMACION_SP.md` fila TTH-08 (Parcial — F8 diferida)
- `documentation/contracts/vision_contract.md` §5.2

### 3.2 C7.6 reabierta como F9.z

**DHU-024 §7 declaró**: *"al reescribir `requirements.txt` desde cero, se definen las dependencias con `--index-url https://download.pytorch.org/whl/cpu`"*. Auditoría F9 confirmó que **NO se aplicó**: `edge_device/requirements.txt:5` sigue siendo `torch` sin `--index-url`.

F9 reabre la deuda honestamente:
- Addendum F9 al pie de DHU-024 §7 reconoce que la promesa no se materializó.
- `TODO.md` C7.6 reescrita con estado "REABIERTA como F9.z al cierre de TTH-08 F9".
- Nuevo item `F9.z` agregado a `TODO.md` como sub-fase de infra separable post-F9.
- **No se aplica el fix en F9** porque rompe la restricción dura "cero código productivo".

### 3.3 Retiro de C1.5/C1.6/C1.7/C1.8

| Item | Estado nuevo | Razón |
|------|--------------|-------|
| **C1.5** | `[x]` resuelta | F5c rediseñó la concurrencia del pipeline; `AsyncVisionPipeline` reemplazado por `FrameProducer.read()` que evita la race. Test verde en `test_async_pipeline.py:101`. xfail original retirado. |
| **C1.6** | `[~]` obsoleta | F4b reescribió la capa `application/`; `MultiCameraManager` reemplazado por `CameraManager`. Tests legacy en `edge_device/tests/vision/unit/test_multi_camera_manager.py` quedan huérfanos — nominados a **F9.y** (barrido). |
| **C1.7** | `[~]` obsoleta | DHU-024 §3 declaró `SmartDetectionProcessor.get_analysis_for_frame()` lógica muerta; tests del processor ya eliminados. Source `edge_device/src/vision/application/processors/smart_detection.py` queda huérfano — nominado a **F9.y**. |
| **C1.8** | `[x]` resuelta | F4a + DHU-025 (2026-05-28): `ZoneCounter` extendido con `mean_occupancy`. Test verde en `test_zone_counter_basic.py`. CT-08.2 cubierto. |

### 3.4 javier_prado.yaml — limpieza

Auditoría F7 §6.2 confirmó la config muerta (Hydra no la compone, cero matches en `.py`, `persistence.type: csv` bit-rot post-F5b). F9 decidió **alt. (iii)**: `git mv` a `documentation/legacy/vision_configs/javier_prado.yaml` + header de comentarios YAML explicando estado histórico, bit-rot detectado y condición de reactivación. Alt. (ii) (reescribir a postgres) se descartó porque ataría con la calibración direccional que F8 difiere.

---

## 4. Backlog post-TTH-08 (infra/cleanup separable)

Cinco ítems descubribles como conjunto. **Ninguno se ejecuta en F9** — todos quedan trackeados y nominados, ninguno escondido.

| Ítem | Tipo | Qué hace falta | Origen | Dónde trackeado |
|------|------|----------------|--------|-----------------|
| **F8** | Datos + validación | Etiquetar dataset ≥200 frames + medir mAP/precisión/recall del detector + rastrear 88.2% honestamente | CT-08.9 + D-005 + DHU-024 §6 | Handoff F7 §6.1; addendum F9 a DHU-024; `ESTIMACION_SP.md` TTH-08; `vision_contract.md` §5.2 |
| **C9.7** | Tests (deuda chica) | Test con `alembic.autogenerate.api.compare_metadata` para detectar divergencia migración↔modelo | Handoff F7 §6.2 (paridad migración) | `TODO.md` C9.7 |
| **C9.8** | Infra CI | Job CI nuevo con Docker + caché TimescaleDB + decisión sobre deps pesadas (YOLO/torch) para correr `edge_device/tests` | Handoff F7 §5 + §6.3 (wirear CI) | `TODO.md` C9.8 |
| **F9.y** | Cleanup código | Borrar `edge_device/src/vision/application/processors/smart_detection.py` y `edge_device/tests/vision/unit/test_multi_camera_manager.py` (sin consumidor runtime tras refactor) | C1.6, C1.7 (TODO) + DHU-024 §3 | `TODO.md` F9.y |
| **F9.z** | Infra reqs | Aplicar `--index-url https://download.pytorch.org/whl/cpu` (o equivalente) a `edge_device/requirements.txt:5` + smoke build/import | C7.6 reabierta + addendum F9 a DHU-024 §7 | `TODO.md` C7.6 (reabierta) + F9.z (nuevo) |

Ninguno bloquea el cierre de TTH-08 como Parcial. F8 + F9.z requieren decisión del usuario para arrancar (datos / infra). C9.7, C9.8, F9.y son cleanup recogedibles en cualquier momento.

### 4.1 Notas chicas (sub-deudas separables)

- **Docstring stale en `edge_device/tests/vision/integration/test_persistence_e2e.py:4`**. La cabecera del test referencia `documentation/docs/tth-08-fase6-handoff.md` §1 y §4.3, ruta vieja del handoff F6 que la reorganización F9i movió a `documentation/handoffs/tth-08/tth-08-fase6-handoff.md`. **F9i NO actualizó esa línea del `.py`** porque tocar código productivo rompe el guard "cero código productivo" de la rama de cierre. Es un docstring informativo, sin impacto en comportamiento. **Se corrige cuando F9.y o un TTH-03 retomado toque ese test** — ese día se aprovecha para reapuntar la ruta del docstring a la ubicación actual.

---

## 5. Cross-refs bidireccionales clave (F9 ↔ resto de la documentación)

| Doc | Anclaje desde F9 | Anclaje hacia F9 |
|-----|------------------|------------------|
| `documentation/contracts/vision_contract.md` | §1 contract entregado | §7 deudas heredadas (F9.z, C9.7, C9.8, F9.y) y §8 cross-refs apuntan al handoff F9 |
| `DECISIONS_HU.md` DHU-024 | §3.1 (F8) y §3.2 (C7.6) citan el addendum F9 | Addendum F9 al pie de DHU-024 cita este handoff |
| `tth-08-fase7-handoff.md` | §3 retiro C1.x e historial de cierre | §6.1, §6.2, §6.3 con punteros forward al handoff F9 (commit `docs(tth-08): 9g`) |
| `TODO.md` | §3.3 retiro C1.x; §4 backlog post-TTH-08 | C1.5/C1.8 `[x]`, C1.6/C1.7 `[~]`, C7.6 reabierta, C9.7/C9.8/F9.y/F9.z nuevos — todos con cross-ref a este handoff |
| `ESTIMACION_SP.md` | §2 estado de los 11 CTs | Fila TTH-08: "Parcial — F8 diferida. Ver `tth-08-fase9-handoff.md`" |
| `EVOLUCION_TESIS.md` §8 + `LEAN_INCEPTION.md` "Trabajos Futuros" | §3 (no editado, F41 reafirmada sin cambio estructural) | Nota al pie F41: "reafirmada en TTH-08 F9, ver `vision_contract.md` §6" |
| `documentation/legacy/vision_configs/javier_prado.yaml` | §3.4 limpieza | Header del YAML: "movido por F9 desde `edge_device/conf/vision/`; ver tth-08-fase9-handoff.md" |

---

## 6. Estado de la rama al cierre

- **Branch**: `feature/tth-08-fase9-docs` (desde `master@15bc6ca4`).
- **Working tree clean** al cierre del último commit (este handoff).
- **Commits desde `master@15bc6ca4`** (8 totales):
  1. `920bcc75` — 9a `docs(tth-08): contrato canónico de visión (vision_contract.md, DHU-024 §5)`.
  2. `31e1b3b5` — 9b `chore(tth-08): mueve javier_prado.yaml a documentation/legacy`.
  3. `2dc07cb3` — 9c `docs(tth-08): retira C1.5/C1.6/C1.7/C1.8, reabre C7.6 como F9.z y nomina C9.7/C9.8`.
  4. `e92d8be7` — 9d `docs(tth-08): addendum F9 a DHU-024 con estado real al cierre del sprint`.
  5. `d2a11611` — 9e `docs(tth-08): TTH-08 marcado como Parcial con F8 diferida en ESTIMACION_SP`.
  6. `7eab9d06` — 9f `docs(tth-08): reafirma F41 como integración futura con cross-ref al contract`.
  7. `f3aa3d24` — 9g `docs(tth-08): cross-refs forward desde handoff F7 hacia F9`.
  8. (este) — 9h `docs(tth-08): handoff Fase 9 + cierre documental del sprint TTH-08`.

**Pre-requisitos antes de mergear** (decisión humana, no del agente):
- Verificar suite `pytest tests/vision/` siga en 124 passed con Docker / 120 passed + 4 skipped sin Docker (no cambia nada de código, debería seguir igual).
- Verificar `git diff --stat master..feature/tth-08-fase9-docs -- '*.py' 'requirements*.txt' 'pyproject.toml' '*.yml' '*.yaml' 'Dockerfile*'` muestra **solo el rename `javier_prado.yaml`** y nada más — guard de "cero código productivo".
- Abrir PR con este handoff como descripción.
- El agente **no mergea ni hace push** — esa decisión queda fuera de su scope.

Con Fase 9 mergeada, **TTH-08 cierra como Parcial — F8 diferida, C7.6 reabierta como F9.z**, con todos los entregables documentales prometidos por DHU-024 §5 entregados, cross-refs bidireccionales consistentes y el backlog post-TTH-08 explícito como deuda separable.
