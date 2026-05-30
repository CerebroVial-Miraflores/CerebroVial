# Estado de la tesis CerebroVial — actualizado 2026-05-30

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
- **Próximo paso (opcional, plan-first — toca el motor): mirar-al-vecino (network-aware MP).** Como
  el +15.7% es ajustado, podría correr el cumplimiento de marginal a holgado. No imprescindible.

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
  CLAUDE.md "No instalar torch en core_management_api" permanece como guardia anti-regresión.
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

## Dónde vive cada cosa (índice)
- Guía para agentes IA (canon): CLAUDE.md (raíz).
- Estado del SDD: documentation/sdd/SPECKIT_MAPPING.md.
- Artefactos Spec Kit: specs/001-cerebrovial-mvp/.
- Constitución del proyecto: .specify/memory/constitution.md.
- Backlog / HUs / DHU: documentation/lean-inception/.
- Decisiones técnicas: documentation/lean-inception/4-decisiones/ (DECISIONS.md, DECISIONS_HU.md).
- Modelo de datos: documentation/docs/DATA_MODEL.md.
- Plan operativo histórico: documentation/docs/PLAN.md.
