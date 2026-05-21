# SPECKIT_MAPPING — Mapeo del corpus CerebroVial → GitHub Spec Kit

> **Modo de adopción: Brownfield / Iterative Enhancement.**
> Spec Kit v0.8.11 se instaló únicamente como **andamiaje** (CLI, plantillas vacías,
> skills `/speckit-*`). El contenido de specs de CerebroVial **ya existe y está curado**;
> **NO se regenera** con `/speckit-specify`, `/speckit-plan`, `/speckit-tasks`,
> `/speckit-constitution` ni `/speckit-implement`. Este documento define **de dónde** se
> tomaría el contenido para poblar cada artefacto de Spec Kit, cuando se decida hacerlo
> en una sesión posterior con supervisión humana.

Fecha de instalación: 2026-05-20 · Branch: `feature/SDD` · Spec Kit: v0.8.11

---

## 1. Tabla de correspondencia (paths reales verificados)

| Artefacto Spec Kit | Documento(s) fuente reales en el repo | Notas |
|---|---|---|
| `.specify/memory/constitution.md` | `documentation/lean-inception/1-contexto/LEAN_INCEPTION_CEREBROVIAL.md`; `documentation/lean-inception/4-decisiones/DECISIONS.md` (D-001…D-009); `documentation/lean-inception/4-decisiones/DECISIONS_HU.md` (DHU-001…DHU-020); `documentation/docs/DECISIONS.md` | Principios rectores + decisiones técnicas y de HU. |
| `specs/NNN-*/spec.md` | `documentation/lean-inception/2-backlog/HU_BLOQUE_A.md` … `HU_BLOQUE_F.md`, `HU_LITE.md`, `HU_MVP2.md`; `documentation/lean-inception/3-requisitos/REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md`, `RF_RNF_LITE.md`; `documentation/lean-inception/BACKLOG_OVERVIEW.md` | El "qué" y el "por qué": 21 HU + 22 RF + 53 RNF. |
| `specs/NNN-*/plan.md` | `documentation/docs/ARCHITECTURE_TARGET.md` (+ SDD en redacción en branch `feature/SDD`) | **Pendiente:** el `SDD_CEREBROVIAL.md` del prompt original **aún no existe** como archivo; ver §3. |
| `specs/NNN-*/data-model.md` | `documentation/docs/DATA_MODEL.md`; `documentation/docs/DATA_MODEL_AUDIT.md` | Incluye persistencia de estado vigente (DHU-020). |
| `specs/NNN-*/tasks.md` | `documentation/lean-inception/planificacion/REPORTE_PLANIFICACION_SPRINT_4.md` (+ `DISTRIBUCION_SPRINTS.md`, `HU_PRIORIZADAS_SPRINTS.md`, `PROTOCOLO_DISTRIBUCION_SPRINTS.md`) | Desglose de tareas del sprint vigente. |
| Estado real / auditoría (insumo de `/speckit-analyze` y matriz de trazabilidad) | `documentation/lean-inception/planificacion/AUDITORIA_HU_CODIGO.md` | Auditoría HU↔código (deltas implementación vs spec). |

> **Corrección respecto del prompt original:** los nombres "ideales" del prompt
> (`LEAN_INCEPTION_CEREBROVIAL.md` en raíz, `SDD_CEREBROVIAL.md`, `AUDITORIA_HU_CODIGO.md`
> en raíz, etc.) no viven en la raíz del repo. El corpus real está organizado bajo
> `documentation/lean-inception/{1-contexto, 2-backlog, 3-requisitos, 4-decisiones,
> planificacion}/` y `documentation/docs/`. La tabla de arriba ya usa los **paths reales**.

---

## 2. Qué instaló Spec Kit (referencia)

- **`.specify/`** (36 archivos): `memory/constitution.md` (plantilla vacía), `templates/`
  (`spec`, `plan`, `tasks`, `checklist`, `constitution`), `scripts/bash/`,
  `workflows/speckit/`, y `extensions/git/` (extensión git, instalada por detectar repo
  existente).
- **`.claude/skills/`** (14 skills): `speckit-constitution`, `speckit-specify`,
  `speckit-plan`, `speckit-tasks`, `speckit-implement`, `speckit-clarify`,
  `speckit-analyze`, `speckit-checklist`, los 5 `speckit-git-*` y `speckit-taskstoissues`.
- En esta versión los comandos son **skills con guion** (`/speckit-constitution`, …),
  no con punto.
- `CLAUDE.md` recibió un append inocuo entre marcadores `<!-- SPECKIT START/END -->`
  (puntero de contexto). Las reglas del proyecto, incluida **"NO refactorizar
  `edge_device/src/vision/`"**, quedaron intactas.

---

## 3. Pendientes de decisión humana (antes de poblar plantillas)

1. **Estructura de `specs/NNN-*/`:** ¿una sola feature que englobe todo el MVP, o varias
   features (p. ej. una por bloque de HU A–F, o una por dominio: edge/vision, core API,
   IA-predicción, frontend)? Spec Kit asume "una feature = una branch = una carpeta
   `specs/NNN-nombre/`"; hay que decidir la granularidad antes de generar carpetas.
2. **Convención de nombres** de las carpetas `specs/NNN-slug/` (numeración y slug) y cómo
   se relaciona con la numeración de HU/RF/RNF ya existente.
3. **Trazabilidad de las 20 DHU** dentro del formato Spec Kit: definir dónde viven
   (¿en `constitution.md`? ¿como sección de cada `spec.md`? ¿una matriz aparte?) para no
   perder el hilo DHU ↔ HU ↔ RF/RNF ↔ código que ya documenta `AUDITORIA_HU_CODIGO.md`.
4. **Faltante `SDD_CEREBROVIAL.md`:** el `plan.md` de Spec Kit necesita una fuente de
   diseño técnico. Hoy no existe ese archivo; el candidato más cercano es
   `documentation/docs/ARCHITECTURE_TARGET.md`, y hay un SDD en redacción en la branch
   `feature/SDD`. Decidir: ¿se completa primero el SDD y luego se mapea, o `plan.md` se
   alimenta de `ARCHITECTURE_TARGET.md` por ahora?
5. **`.gitignore` y `.claude/`:** decidido **no** ignorar `.claude/` (los skills se
   versionan con el repo). Revisar que no se cuelen credenciales del agente.

---

## 4. Regla operativa

Mientras este documento sea el único mapeo acordado, **no** ejecutar comandos generativos
de Spec Kit. Poblar plantillas es una fase posterior, explícita y supervisada.
