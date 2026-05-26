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
| `.specify/memory/constitution.md` | `documentation/lean-inception/1-contexto/LEAN_INCEPTION_CEREBROVIAL.md`; `documentation/lean-inception/4-decisiones/DECISIONS.md` (D-001…D-009); `documentation/lean-inception/4-decisiones/DECISIONS_HU.md` (DHU-001…DHU-022); `documentation/docs/DECISIONS.md` | Principios rectores + decisiones técnicas y de HU. |
| `specs/NNN-*/spec.md` | `documentation/lean-inception/2-backlog/HU_BLOQUE_A.md` … `HU_BLOQUE_F.md`, `HU_LITE.md`, `HU_MVP2.md`; `documentation/lean-inception/3-requisitos/REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md`, `RF_RNF_LITE.md`; `documentation/lean-inception/BACKLOG_OVERVIEW.md` | El "qué" y el "por qué": 21 HU + 22 RF + 53 RNF. |
| `specs/NNN-*/plan.md` | **`documentation/sdd/SDD_CEREBROVIAL.md`** (SDD verificado contra el repo, 2026-05-20) | Fuente única del plan técnico (estructura híbrida 4+1). `ARCHITECTURE_TARGET.md` quedó archivado en `legacy/` (DHU-021 §5) y NO se usa. |
| `specs/NNN-*/data-model.md` | §4 de `documentation/sdd/SDD_CEREBROVIAL.md` + `documentation/docs/DATA_MODEL.md`; `documentation/docs/DATA_MODEL_AUDIT.md` | Modelo heredado + las dos entidades nuevas (`motor_decisions`, `engine_active_state`) verificadas; estado vigente (DHU-020). |
| `specs/NNN-*/tasks.md` | `documentation/lean-inception/planificacion/REPORTE_PLANIFICACION_SPRINT_4.md` + `ESTIMACION_SP.md` + `AUDITORIA_HU_CODIGO.md` + `MOSCOW_RATIFICADA.md` (+ `DISTRIBUCION_SPRINTS.md`, `HU_PRIORIZADAS_SPRINTS.md`, `PROTOCOLO_DISTRIBUCION_SPRINTS.md`) | Inventario de 32 elementos (estado + MoSCoW + SP) + Sprint 4 vigente. |
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

## 3. Decisiones de poblado (resueltas el 2026-05-20)

Los pendientes que este documento registró en la instalación quedaron resueltos en la
sesión de verificación + poblado:

1. **Estructura de `specs/`:** **una sola feature**, `specs/001-cerebrovial-mvp/`. El SDD
   es un único documento de arquitectura del MVP; fragmentarlo por bloque de HU o por
   dominio dispersaría la fuente. Se descarta multi-feature por ahora.
2. **Convención de nombres:** `specs/001-cerebrovial-mvp/` (numeración Spec Kit `NNN-slug`).
   La numeración de HU/RF/RNF/TTH se conserva intacta dentro de los artefactos; no se
   renumera nada del corpus.
3. **Trazabilidad de las DHU (ya 22):** su **hogar canónico** es
   `DECISIONS_HU.md`; los artefactos de Spec Kit la **citan**, no la duplican. La matriz
   HU/TTH ↔ componente ↔ estado ↔ delta vive en §10 del SDD (y de ahí en `plan.md`).
4. **`SDD_CEREBROVIAL.md`:** ya existe y está **verificado contra el repo** (2026-05-20).
   Es la fuente de `plan.md` y, con `DATA_MODEL.md`, de `data-model.md`.
   `ARCHITECTURE_TARGET.md` quedó archivado en `legacy/` (DHU-021 §5) y no se cita.
5. **`.gitignore` y `.claude/`:** decidido **no** ignorar `.claude/`. Antes de cualquier
   commit que lo incluya, verificar que no contenga credenciales/tokens (check de seguridad).

---

## 4. Estado del poblado

**Los cinco artefactos están poblados (2026-05-20), todos por mapeo brownfield — sin comandos
generativos.** La regla sigue vigente: mapear, no regenerar.

| Artefacto | Estado | Fuente del mapeo |
|---|---|---|
| `.specify/memory/constitution.md` | ✓ poblado (2026-05-20) | `DECISIONS.md` (D-001…009) + `DECISIONS_HU.md` (DHU-001…022). 22 artículos: Tít. I (8, D-002+D-006 fusionados) + Tít. II (Arts. 9-22, 14 artículos cubriendo las 22 DHU por fusión y agrupación). |
| `specs/001-cerebrovial-mvp/spec.md` | ✓ poblado (2026-05-20) | `BACKLOG_OVERVIEW.md` + `HU_BLOQUE_*`/`HU_MVP2` + `RF_RNF_LITE.md`. 21 HU por bloque/Persona; IDs nativos; CAs Gherkin enlazados, no copiados. |
| `specs/001-cerebrovial-mvp/plan.md` | ✓ poblado (2026-05-20) | `SDD_CEREBROVIAL.md`. |
| `specs/001-cerebrovial-mvp/data-model.md` | ✓ poblado (2026-05-20) | `SDD_CEREBROVIAL.md` §4 + `DATA_MODEL.md`. |
| `specs/001-cerebrovial-mvp/tasks.md` | ✓ poblado (2026-05-20) | `REPORTE_PLANIFICACION_SPRINT_4.md` + `ESTIMACION_SP.md` + `AUDITORIA_HU_CODIGO.md` + `MOSCOW_RATIFICADA.md`. Inventario de 32 elementos con nota de protección; estado reverificado contra HEAD (sin avance de construcción tras la auditoría del 2026-05-18). |

Regla de fuente común a los cinco: cada artefacto referencia su fuente por ID y archivo; no
duplica texto Gherkin, justificaciones ni catálogos. El SDD sigue siendo canónico de arquitectura
y `DECISIONS.md`/`DECISIONS_HU.md` de las decisiones.

---

## 5. Estado de adopción al 2026-05-25 (snapshot)

Cierre del ciclo Spec Kit tras una sesión de `/speckit-analyze` + remediación
documental ejecutada el 2026-05-25. **Working tree limpio** en rama `feature/SDD`.

### 5.1 Artefactos del ciclo Spec Kit (6 de 6 poblados)

| Artefacto | Commit | Fecha | Notas |
|---|---|---|---|
| `.specify/memory/constitution.md` | `57e9c016` | 2026-05-20 | 22 artículos vinculantes |
| `specs/001-cerebrovial-mvp/spec.md` | `9ab94d4c` (+ remediación) | 2026-05-20 / 2026-05-25 | 21 HU + 22 RF + 53 RNF; HU-05 agnóstica (Art. 14); nota de alcance SC-001/SC-003 |
| `specs/001-cerebrovial-mvp/plan.md` | `90453dc3` (+ refresco) | 2026-05-20 / 2026-05-25 | Constitution Check y bloque Documentation actualizados |
| `specs/001-cerebrovial-mvp/data-model.md` | `90453dc3` | 2026-05-20 | SDD §4 + DATA_MODEL.md |
| `specs/001-cerebrovial-mvp/tasks.md` | `93a49ed0` | 2026-05-20 | Inventario 32 elementos + nota de protección + SAN-01/02/03 |
| `specs/001-cerebrovial-mvp/quickstart.md` | `8bf95677` | 2026-05-25 | Mapeo brownfield del flujo `invoke up` |

`research.md` no se requiere — las decisiones técnicas viven en `DECISIONS.md` y `DECISIONS_HU.md` (regla §3).

### 5.2 Tooling Spec Kit (destrabado para brownfield)

`commit d2dab5b6` (2026-05-25) cierra la fricción entre la rama brownfield
`feature/SDD` y el patrón Spec Kit `NNN-feature-name`:

- **`.specify/feature.json`** declara explícitamente `feature_directory:
  specs/001-cerebrovial-mvp`. Permite a los scripts resolver FEATURE_DIR sin
  depender del nombre de la rama (mecanismo previsto en `common.sh:188-207`).
- **`.specify/scripts/bash/check-prerequisites.sh`** alineado con
  `setup-plan.sh`/`setup-tasks.sh`: si `feature.json` resuelve al FEATURE_DIR
  activo, se omite el chequeo estricto de nombre de rama. Cambio mínimo y
  consistente con el patrón ya usado por los otros scripts.

Con esto, `/speckit-analyze`, `/speckit-implement` y `/speckit-checklist`
operan sobre la rama actual sin renombrarla.

### 5.3 Tareas de saneamiento (SAN-01/02/03)

Creadas en `tasks.md` (commit `93a49ed0`) como encapsulación de la regla de
CLAUDE.md ("no `torch` en core") + Delta-13 (Gemini huérfano) + Delta-05
(`vision_aggregates` no migrada). **Ninguna bloquea la construcción del
Sprint 4** (TTH-01 → HU-01 → TTH-10 → HU-05 → TTH-03).

### 5.4 Decisiones cerradas que cierran deltas

- **DHU-020** (Cerrada 2026-05-20): resuelve Delta-08 / R1 (semántica de
  ControlView). Vista pasiva de HU-05 prevalece; playground se preserva como
  herramienta de Administrador; Delta-07/08/09 se abordan en bloque; persistencia
  de "estado vigente del motor" autorizada como cambio estructural. Codificada
  como Artículo 20 de constitution. `AUDITORIA_HU_CODIGO.md`,
  `REPORTE_PLANIFICACION_SPRINT_4.md` y `DELTA_08_ANALISIS.md` ya apuntan al
  cierre (commit `d0ac9238`, 2026-05-25).
- **DHU-022** (Cerrada 2026-05-20): cierra Delta-02 fijando
  `operator/manager/admin` como claims técnicos canónicos con labels en
  español en frontend.

### 5.5 Resultado de `/speckit-analyze` (run #2, 2026-05-25)

- **0 errores CRITICAL** · **0 errores HIGH** · **0 errores LOW abiertos**
- 1 MEDIUM (auto-cubierto por este snapshot) y 2 LOW intencionales
  (B1 confirmaciones con asesor, A1 duplicación controlada de HU-09 por DHU-017).

### 5.6 Próximo paso recomendado

El ciclo SDD está **operacionalmente sellado**. Cuando se decida avanzar:

- **Tier 2 — Operacionalización:** `/speckit-taskstoissues` para los 5
  elementos del Sprint 4; `/speckit-checklist` por HU/TTH; convención de
  branching por elemento.
- **Tier 3 — Saneamiento:** cerrar SAN-01 (torch en core), decidir SAN-02
  (Gemini: HU formal o remover), cerrar SAN-03 (migration `vision_aggregates`
  + cableado del aggregator).
- **Tier 4 — Ejecución:** `/speckit-implement` sobre TTH-01 (Auth JWT +
  bcrypt) y seguir el orden semanal del Sprint 4.

Estos Tiers **no son parte de este sellado**; quedan como hoja de ruta.
