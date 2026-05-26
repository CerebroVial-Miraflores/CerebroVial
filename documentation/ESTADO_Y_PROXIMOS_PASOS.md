# Estado de la tesis CerebroVial — actualizado 2026-05-25

## Dónde estoy
Ciclo SDD (Spec Kit v0.8.11, brownfield) cerrado y sellado. 6/6 artefactos poblados y verificados:
constitution, spec, plan, tasks, data-model, quickstart. /speckit-analyze: 0 errores CRITICAL/HIGH.
Rama de trabajo del SDD: feature/SDD. Snapshot de adopción en documentation/sdd/SPECKIT_MAPPING.md §5.

## Siguiente paso (Tier 4 — construcción del Sprint 4)
Orden comprometido (19 SP, de tasks.md): TTH-01 (Auth JWT+bcrypt) → HU-01 (RBAC) →
TTH-10 (cierre Motor) → HU-05 (ControlView pasiva) → TTH-03 (cierre CI).
Comando de arranque: /speckit-implement sobre TTH-01.
Autoridad del alcance del sprint: tasks.md (NO los 32 elementos del inventario; solo estos 5).

## Configuración intencional preservada
`CerebroVial/.gemini/settings.json` (5 líneas) configura Gemini CLI para que cargue
`CLAUDE.md` como contexto del proyecto. Es flujo multi-agente intencional del equipo
(consumidor humano: un compañero del proyecto usa `gemini` CLI sobre este repo). NO es
deuda ni candidato a remover; queda versionado tal cual. Misma lógica que la guardia
de ThesisModal en `CLAUDE.md`.

La pasada original de "limpieza ligera del repo" (basura .DS_Store, reubicación de docs
sueltos, archivado de guía obsoleta, actualización quirúrgica de CLAUDE.md) ya fue
ejecutada en `chore/orden-repo` (merge a master en commit `d3994e22`).

## Tareas de saneamiento diferidas (NO bloquean Sprint 4)
- SAN-01 ✓ resuelta (2026-05-26, rama `san-06`): se eligió el camino "purgar torch del módulo"
  (no se relajó la regla CLAUDE.md). Se eliminaron 6 archivos STGCN muertos de
  `core_management_api/src/prediction/` y la línea `torch` de `core_management_api/requirements.txt`.
  El runtime vivo (`predictor.py → engine.py`) usa RandomForest + joblib, sin torch. La regla
  CLAUDE.md "No instalar torch en core_management_api" permanece como guardia anti-regresión.
  Cierra simultáneamente C7.5 (TODO.md).
- SAN-02: decidir destino de componentes Gemini huérfanos (Art. 21 los declara fuera de arquitectura).
- SAN-03: crear tabla vision_aggregates + cableado (Delta-05). Es Trabajo Futuro, no Sprint 4.
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
