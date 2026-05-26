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

## Pendiente de orden (esta rama chore/orden-repo)
Limpieza ligera del repo: basura .DS_Store, reubicación de 2 docs sueltos, archivado de guía
obsoleta, actualización quirúrgica de CLAUDE.md, mecanismo multi-agente (.gemini/settings.json).

## Tareas de saneamiento diferidas (NO bloquean Sprint 4)
- SAN-01: contradicción regla-vs-código. CLAUDE.md (línea 84) prohíbe `torch` en
  `core_management_api`, pero `requirements.txt` y `prediction/*.py` lo usan (justificación D-006/GRU).
  **NO ejecutar como "quitar torch" sin resolver antes** si se corrige el código (purgar torch del
  módulo) o la regla (relajar el CLAUDE.md). Decisión de arquitectura para cuando se aborde TTH-09.
- SAN-02: decidir destino de componentes Gemini huérfanos (Art. 21 los declara fuera de arquitectura).
- SAN-03: crear tabla vision_aggregates + cableado (Delta-05). Es Trabajo Futuro, no Sprint 4.

## Dónde vive cada cosa (índice)
- Guía para agentes IA (canon): CLAUDE.md (raíz).
- Estado del SDD: documentation/sdd/SPECKIT_MAPPING.md.
- Artefactos Spec Kit: specs/001-cerebrovial-mvp/.
- Constitución del proyecto: .specify/memory/constitution.md.
- Backlog / HUs / DHU: documentation/lean-inception/.
- Decisiones técnicas: documentation/lean-inception/4-decisiones/ (DECISIONS.md, DECISIONS_HU.md).
- Modelo de datos: documentation/docs/DATA_MODEL.md.
- Plan operativo histórico: documentation/docs/PLAN.md.
