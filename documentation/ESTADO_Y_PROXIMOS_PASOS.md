# Estado de la tesis CerebroVial — actualizado 2026-06-01

## Dónde estoy
Ciclo SDD (Spec Kit v0.8.11, brownfield) cerrado y sellado. 6/6 artefactos poblados y verificados:
constitution, spec, plan, tasks, data-model, quickstart. /speckit-analyze: 0 errores CRITICAL/HIGH.
Rama de trabajo del SDD: feature/SDD. Snapshot de adopción en documentation/sdd/SPECKIT_MAPPING.md §5.

## Siguiente paso (Tier 4 — construcción del Sprint 4)
Orden comprometido (19 SP, de tasks.md): TTH-01 (Auth JWT+bcrypt) → HU-01 (RBAC) →
TTH-10 (cierre Motor) → HU-05 (ControlView pasiva) → TTH-03 (cierre CI).
Comando de arranque: /speckit-implement sobre TTH-01.
Autoridad del alcance del sprint: tasks.md (NO los 32 elementos del inventario; solo estos 5).

## Corredor Larco / IE05 (validación SUMO) — Etapa 2 (2026-05-30)
**IE05 (RD% ≥ 15%): ALCANZADO EN LA MEDIA con adaptación local (cumplimiento marginal)** (track
paralelo al Sprint 4).
- **Número final: RD% RED = +15.7% ± 8.1 (10 semillas, 9/10 positivas)** con métrica de demora
  **puerta-a-puerta robusta a censura** (cuenta espera para entrar + autos abandonados en la cola,
  no solo los que completan dentro de la red). Sistema = **MP per-node de ciclo variable**.
  Cumplimiento **ajustado** (media−SD = +7.6%); mejora robusta y significativa, dispersión
  reportada. Beneficio físico: **−67% de espera para entrar**, adentro casi igual.
- **El "empate" previo (+1.0% ± 7.5) era artefacto de medición** (la métrica vieja no contaba a los
  autos que el fijo deja sin insertar: 68 vs 23). Mecanismo Benavides→Schell 10/10 intacto. Detalle
  y framing honesto en `documentation/handoffs/corredor-larco/etapa-2-cierre-ie05.md`.
- **Onda verde (offsets) y ciclo común fijo: explorados y DESCARTADOS** (offset=0 óptimo; ciclo
  fijo no mejora al variable, no generaliza).
- **Mirar-al-vecino (MP de red, downstream del link interno): EJECUTADO y DESCARTADO (2026-05-30).**
  Se extendió el motor con término downstream opcional (Etapa 1, retrocompat bit-a-bit) y se corrió
  pareado 42–51 (Etapa 2). **Refutado:** MP-red **empeora** la demora RED vs per-node — Δ pareado
  **+35.07 s**, IC **[+21.59, +48.55] excluye 0**, Wilcoxon **p=0.002**, **0/10** favorables; MP-red
  **−9.2% vs fijo**. Mecanismo (capacidad-limitado): alivia el link interno (benSch mean −22%) pero
  relocaliza la cola a la entrada (larcoS 144.6→172.2 m) y cuadruplica la espera para entrar (w_wait
  12.2→48.2 s). **Barrido τ (0/0.5/0.75/1.0) ejecutado:** eje monótono +15.6% → +2.5% → −2.6% → −9.2%
  vs fijo; ningún τ supera al per-node (IC pareado excluye 0 en los tres τ>0). **Sistema adoptado: MP
  per-node (τ=0), óptimo del eje.** Detalle: `etapa-2-cierre-mp-red.md`; síntesis legible (benchmark,
  glosario, supuestos, configs): `sintesis-corredor-larco.md` (ambos en `documentation/handoffs/corredor-larco/`).
- **Runtime:** `:8001` corre ahora `feature/corredor-larco-mp-red` (Etapa-1 deployada vía rebuild del
  contenedor `core_management_api`); ruta sin-downstream byte-idéntica al per-node, **sin migración**,
  **sin mergear a master**.

## STGNN Fase 1 — cierre (grafo LCC 375) (2026-06-01)
**Cierre de Fase 1 del track STGNN Miraflores (D-011/D-012, investigación paralela, fuera de
producción).** Decisión a posteriori del subgrafo de modelado cerrada con evidencia empírica.

Entregado en esta sesión (rama `feature/stgnn-corredor-larco`, **sin push/PR/merge**):
- **`ia_prediction_service/src/data/miraflores_graph_builder.py`** — modo componente-grande:
  parámetros `largest_component_only` + `expected_component_sizes`. Siempre construye y valida
  primero el grafo completo de 381 (descubrimiento, regla `<connection>`, gate de 12 fantasmas —que
  NO se saltea—); luego computa componentes conexas, recorta a la mayor (375 de 381), re-ordena
  lexicográfico y reasigna `node_index` 0..N-1. Gate de componentes opt-in; mutua exclusión con
  `edge_ids`. Default (381) intacto.
- **Dos mappings JSON versionados** en `ia_prediction_service/src/data/artifacts/`:
  `miraflores_graph_mapping.json` (381, evidencia del análisis de componentes) y
  `miraflores_graph_lcc_mapping.json` (375, canónico de modelado, con procedencia `derived_from` /
  `full_graph_nodes` / `dropped_component_nodes` / `dropped_edges`).
- **`tests/test_miraflores_graph_builder.py`** — 13 tests verdes (6 previos + 7 nuevos: LCC 375,
  gate de componentes, mutua exclusión, conexidad, islita excluida, determinismo, gate de fantasmas
  bajo LCC).

**Topología resultante:** 375 nodos / 504 aristas dirigidas, 1 sola componente conexa. Islita de
6 edges desconectada excluida. **Señal espacial confirmada** (mediana corr vecinos-1 `speedRelative`
0.46 vs no-vecinos 0.01; decaimiento monótono 0.46→0.17→0.01; replicado en `density` +0.38),
distribuida (269/381 edges), recorte por **topología**, no por densidad. Esparsidad ~82% diferida
a modelado Fase 2/3. Decisión fundamentada anexada a **D-012**; respaldo crudo en
`documentation/handoffs/stgnn-fase1/REPORTE_CRUDO_BLOQUES_0_1.md`.

**Target del track — cerrado (D-013, 2026-06-01):** el reemplazo `jam_level` (ratio de velocidad)
→ **demora continua (`meanTimeLoss`)** quedó decidido, cerrando la deuda que D-012 y su anexo dejaban
diferida a Fase 2. El modelo entrena sobre demora continua; la escala 0–5 estilo Waze pasa a ser capa
de presentación, no target. **D-009 sigue vigente y sin enmienda para producción** — el GRU de TTH-09
y sus consumidores siguen usando jam_level; D-013 es excepción acotada del track de investigación.

**Nota de convención:** los handoffs del track Miraflores se siembran bajo
`documentation/handoffs/stgnn-fase1/` (las fases siguientes heredarán `stgnn-fase2/`, etc.),
separados del histórico `corredor-larco/` (escenario Larco descartado por D-012).

## Configuración intencional preservada
`CerebroVial/.gemini/settings.json` (5 líneas) configura Gemini CLI para que cargue
`CLAUDE.md` como contexto del proyecto. Es flujo multi-agente intencional del equipo
(consumidor humano: un compañero del proyecto usa `gemini` CLI sobre este repo). NO es
deuda ni candidato a remover; queda versionado tal cual. Misma lógica que la guardia
de ThesisModal en `CLAUDE.md`.

La pasada original de "limpieza ligera del repo" (basura .DS_Store, reubicación de docs
sueltos, archivado de guía obsoleta, actualización quirúrgica de CLAUDE.md) ya fue
ejecutada en `chore/orden-repo` (merge a master en commit `d3994e22`).

## TTH-10 — cierre parcial (2026-05-26)
Entregado en esta sesión:
- `motor_decisions` (append-only) + `engine_active_state` (mutable) modelados y migrados
  (`b1f7c4d2a890_motor_decisions_engine_state`). FK a `graph_nodes.node_id`.
- Write-path en `POST /control/recommend`: resolver `intersection_id → node_id` antes
  del cálculo (DHU-021 V1, fail-fast con HTTP 422 `unknown_intersection`), persistencia
  dentro de la transacción del request con `inputs_snapshot`, `flow_total` y
  `y_load_factor` reales (no recalculados). `ControlRecommendation` Pydantic intacto:
  contrato HTTP sin cambios.
- `EngineActiveStateRepo.activate(...)` construido + testeado (insert + update);
  NO cableado a ningún endpoint (HU-05/HU-07).
- `ControlSettings` (pydantic-settings) en `src/control/config.py`: extracción
  de constantes de CT-10.4 / CT-10.6 sin recalibrar (env vars `CONTROL_*`).
- `GET /control/health` sin auth (probes de orquestador). `/api/health` ADMIN no se tocó.
- 16 tests CT-10.X.Y verdes (CT-10.4.1, CT-10.6.1, CT-10.9.1..10.9.10, CT-10.13.1/.2).

Diferido a R2 (registrado en `specs/001-cerebrovial-mvp/data-model.md` § Trabajo futuro):
- CT-10.10 (integración GRU/TTH-09).
- CT-10.11 (integración SUMO/TTH-07 vía TraCI).
- CT-10.12 (parámetros configurables vía HU-15).
- CT-10.13 cascada (consumo del health check por TTH-04 Nivel 3).
- Activación de `engine_active_state` (responsabilidad HU-05/HU-07).

## Tareas de saneamiento diferidas (NO bloquean Sprint 4)
- SAN-01 ✓ resuelta (2026-05-26, rama `san-06`): se eligió el camino "purgar torch del módulo"
  (no se relajó la regla CLAUDE.md). Se eliminaron 6 archivos STGCN muertos de
  `core_management_api/src/prediction/` y la línea `torch` de `core_management_api/requirements.txt`.
  El runtime vivo (`predictor.py → engine.py`) usa RandomForest + joblib, sin torch. La regla
  CLAUDE.md "No instalar torch en core_management_api" permanece como guardia anti-regresión
  **general**, CON la excepción registrada en **D-010** (2026-05-31): se admite `torch` CPU-only
  en el core **exclusivamente para servir el predictor GRU de TTH-09** (inferencia in-process,
  clase `GRUMultiOutput` vendorizada). D-010 no des-cierra esta SAN-01 (aquí se purgó torch
  *muerto*); revisa la regla para reintroducir torch *vivo y justificado*. Ver
  `lean-inception/4-decisiones/DECISIONS.md` § D-010.
  Cierra simultáneamente C7.5 (TODO.md).
- SAN-02: decidir destino de componentes Gemini huérfanos (Art. 21 los declara fuera de arquitectura).
- SAN-03: crear tabla vision_aggregates + cableado (Delta-05). Es Trabajo Futuro, no Sprint 4. (Absorbido por TTH-08 / DHU-024.)
- SAN-04 ✓ resuelto (2026-05-25, rama `fix/consolidar-decisiones`): el canónico
  `documentation/lean-inception/4-decisiones/DECISIONS.md` (D-001…D-009) quedó como única fuente
  viva. La copia vieja se archivó como `documentation/legacy/DECISIONS_2026-05-13_OBSOLETO.md` con
  cabecera de obsolescencia (mismo patrón que `AGENTS_v2_2025-11_OBSOLETO.md`). `constitution.md`
  (preámbulo + Governance) y `documentation/sdd/SPECKIT_MAPPING.md` apuntan ahora a la canónica;
  el pie de versión de la constitución pasó a 1.0.1 (Last Amended 2026-05-25) por actualización
  de cita de fuente del Art. 8 / D-009. Punteros restantes a la ruta vieja en
  `documentation/docs/20260503_PHASE1_CLOSURE.md`, `documentation/docs/DISCOVERY_2026-05-10.md` y
  `documentation/docs/PLAN.md` se preservan intencionalmente como testimonios fechados; el
  linaje queda documentado en la cabecera del archivo legacy.

**Deuda del track STGNN / generación de demanda Miraflores (registrada 2026-06-01):**
- **Perfiles de día sin calibrar.** Solo el perfil **laborable** está calibrado para Miraflores
  (`scale=0.20`, validado sobre la 24h **continua** contra el colapso por carryover). Los perfiles
  finde/feriado/especial **NO** están calibrados — su scale podría diferir, y el "especial" (meseta
  alta) tiene alto riesgo de gridlock. Pendiente: calibrar cada uno con su propio barrido sobre la
  24h **continua** (NO ventanas aisladas — son ciegas al carryover, lección del fallo a `scale=0.35`).
  A cerrar cuando el pipeline esté completo, antes de armar un dataset multi-perfil realista en
  proporción de calendario.
- **Artefactos `corridor_*` obsoletos.** Los 4 archivos `corridor_*` + el handoff de Fase 1 quedaron
  atados al escenario Larco descartado (ya anotado en **D-012**). Migrar/reemplazar al reconstruir
  Fase 1 sobre Miraflores.
- **Dataset actual = solo laborable.** El dataset multi-día en generación es homogéneo (60 días
  laborables, variación por seed). Suficiente para construir/validar el pipeline end-to-end; **NO**
  representa finde/feriado. Enriquecer con otros perfiles tras cerrar el pipeline (depende del punto 1).
- **Clúster tutorial Lightning muerto (registrado Fase 4, 2026-06-01).**
  `ia_prediction_service/src/training/predictor.py` (`tsl.engines.Predictor` + `MaskedMAE`) y los
  scripts `scripts/train.py` / `predict.py` / `evaluate.py` (PyTorch-Lightning) son el pipeline STGNN
  original del tutorial: **desconectado del flujo end-to-end y ya roto en ejecución** (`create_model`
  llamaba a un `print_architecture` sin bindear → `AttributeError`). En Fase 4 se reescribió
  `time_then_space.py` **preservando la firma de `create_model`** (`config, n_nodes=, input_size=`)
  para no cambiar el modo de falla de esos scripts; siguen sin uso. **NO se borraron** (Opción A): borrar
  solo `predictor.py` rompería los 3 imports, y el clúster está cableado en
  `ia_prediction_service/Dockerfile:19` (`CMD ["python", "scripts/train.py"]`). El baseline (Fase 3) y
  el STGNN (Fase 4) usan loop manual en `scripts/train_miraflores_*.py`, NO este clúster.
  **Disparador de limpieza:** "limpieza del Dockerfile" — al tocar el Dockerfile, borrar el clúster
  completo (predictor.py + los 3 scripts) y reapuntar/quitar el `CMD`.

**Deuda de entorno — choque numpy tsl ↔ opencv (registrada 2026-06-01; RESUELTA 2026-06-01 vía separación de venvs):**
- **Framework de modelado decidido (gate de viabilidad tsl).** El gate confirmó que tsl
  (`torch-spatiotemporal`) + PyG instalan limpio en Apple Silicon contra **torch 2.9.1** —wheels
  `universal2` del índice pyg, **sin compilación desde fuente** (incluido `torch-scatter`/
  `torch-sparse`), sin downgrade de torch. Decisión del track: la **Fase 4 (STGNN) usará tsl**; el
  **baseline de Fase 3 se construye en torch puro** (no requiere tsl).
- **Salvedad — pin de numpy (atribución corregida).** El pin `numpy<2` es **directo de tsl**
  (`Requires-Dist: numpy<2,>1.20.3` en el `pyproject` del ref `08473ed2`), **no** transitivo vía
  `tables`/`blosc2` —esas admiten numpy≥2 (`tables` pide `numpy>=1.20`, `blosc2` `numpy>=1.26`). tsl
  ancla numpy a **1.26.4**, lo que choca con el módulo de visión (`edge_device`: opencv + ultralytics),
  que va a **numpy≥2**.
- **RESUELTO (2026-06-01) — separación de venvs.** Se separó el entorno en dos venvs aislados:
  - `ia_prediction_service/.venv` — entrenamiento, **numpy 1.26.4** + tsl (`invoke setup-train`).
  - `.venv` (raíz) — core + visión, **numpy≥2** (`invoke setup-dev`); `torch==2.9.1` se preserva
    (excepción D-010) instalando core+edge+dev en **una sola resolución de pip** (si se instalara edge en
    una invocación aparte, ultralytics/torchvision pisarían torch con una versión más nueva).
  Smokes verdes: el venv de training importa tsl/PyG y corre `train --quick`; el venv core+visión importa
  cv2/ultralytics sobre numpy≥2 (tsl ausente) y `invoke test` pasa (core 124 + edge 124). Pins en
  `ia_prediction_service/requirements.txt` (tsl@08473ed2, numpy==1.26.4, `--find-links` PyG) y
  `edge_device/requirements.txt` (opencv → headless).
- **Asterisco honesto.** `opencv-python` (variante GUI) entra **transitivamente** por
  `supervision`/`ultralytics` (ambos lo declaran como dep dura: `opencv-python>=…`), así que coexiste con
  `opencv-python-headless` en el `.venv` raíz. No es bloqueante (cv2 importa, misma versión 4.13.0.92),
  pero excluir del todo la variante GUI exigiría un constraints/override sobre esos paquetes — follow-up menor.
- **Deuda nueva de seguimiento (reunificación de venvs).** Si un tsl posterior **relaja** `numpy<2`, se
  puede revertir a un venv único. Antes de hacerlo, verificar: (a) que el `Requires-Dist` de tsl upstream
  ya no pinee `numpy<2`, **y** (b) que el código de entrenamiento corra de verdad contra numpy 2 (no solo
  que la metadata lo permita). **Disparador:** querer reunificar los venvs.
- **Referencia.** Snapshot del `.venv` raíz pre-separación en `/tmp/venv_root_pre_separacion.txt`
  (efímero, no versionado).

## Dónde vive cada cosa (índice)
- Guía para agentes IA (canon): CLAUDE.md (raíz).
- Estado del SDD: documentation/sdd/SPECKIT_MAPPING.md.
- Artefactos Spec Kit: specs/001-cerebrovial-mvp/.
- Constitución del proyecto: .specify/memory/constitution.md.
- Backlog / HUs / DHU: documentation/lean-inception/.
- Decisiones técnicas: documentation/lean-inception/4-decisiones/ (DECISIONS.md, DECISIONS_HU.md).
- Modelo de datos: documentation/docs/DATA_MODEL.md.
- Plan operativo histórico: documentation/docs/PLAN.md.
