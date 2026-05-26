---
description: "Inventario de tareas de la feature 001-cerebrovial-mvp (mapeo brownfield)"
---

# Tasks: CerebroVial — Control adaptativo de semáforos (MVP1 + MVP2)

**Input**: Design documents from `/specs/001-cerebrovial-mvp/` (plan.md, spec.md, data-model.md).

**Fuentes**: `REPORTE_PLANIFICACION_SPRINT_4.md`, `ESTIMACION_SP.md`, `AUDITORIA_HU_CODIGO.md`,
`MOSCOW_RATIFICADA.md` (en `documentation/lean-inception/planificacion/`).

> **Adopción brownfield (DHU-021).** Este archivo **mapea** la planificación ya existente al formato
> de Spec Kit; no se genera con `/speckit-tasks`. El estado de construcción se reverificó contra HEAD
> el 2026-05-20: sin commits de código posteriores a la auditoría (2026-05-18), por lo que los estados
> de abajo están vigentes.

> ## ⚠ Nota de protección — leer antes de interpretar esta lista
>
> *Este `tasks.md` lista el backlog completo (32 elementos) como inventario de tareas de la feature,
> con su estado real de construcción y su prioridad MoSCoW. NO debe leerse como "32 elementos en
> ejecución simultánea en el Sprint 4". El alcance comprometido del Sprint 4 vigente son 5 elementos
> (19 SP), definido en `REPORTE_PLANIFICACION_SPRINT_4.md`; el resto (~205 SP) está formalmente
> declarado Trabajo Futuro operacional. Esta presentación de inventario completo es una elección
> deliberada; el corte ejecutable autoritativo es el del reporte de Sprint 4.*

## Leyenda

- **Estado** (de `AUDITORIA_HU_CODIGO.md` §1): Completo / Parcial / No iniciado / Fuera de scope.
- **MoSCoW** (de `MOSCOW_RATIFICADA.md`): Must / Should / Could / Won't.
- **SP tot/ejec/rest** (de `ESTIMACION_SP.md`): puntos de historia totales / ejecutados / restantes.
- **S4**: marca de pertenencia al alcance comprometido del Sprint 4 (orden #).
- **Deltas**: divergencias spec↔código consolidadas en `AUDITORIA_HU_CODIGO.md` §5.

## Tabla maestra del inventario (32 elementos: 11 TTH + 21 HU)

### Tareas Técnicas Habilitadoras (TTH)

| ID | Título corto | Estado | MoSCoW | SP tot | ejec | rest | S4 | Deltas |
|---|---|---|---|---|---|---|---|---|
| TTH-01 | Auth JWT + bcrypt | No iniciado | Must | 5 | 1 | 4 | **#1** | Delta-02 |
| TTH-02 | Docker Compose | Completo | Must | 5 | 5 | 0 | — | — |
| TTH-03 | Repo + CI cobertura | Parcial | Must | 5 | 3 | 2 | **#5** | Delta-03 |
| TTH-04 | Fallback en cascada | No iniciado | Must | 13 | 0 | 13 | — | — |
| TTH-05 | Tiempos degradado nivel 3 | No iniciado | Must | 5 | 0 | 5 | — | — |
| TTH-06 | Capa DTOs | Fuera de scope | Won't | 8 | 0 | — | — | — |
| TTH-07 | Integración SUMO | No iniciado | Must | 13 | 0 | 13 | — | — |
| TTH-08 | Visión computacional | Parcial | Must | 13 | 5 | 8 | — | Delta-04, Delta-05 |
| TTH-09 | GRU servido vía API | No iniciado | Must | 13 | 1 | 12 | — | Delta-01 |
| TTH-10 | Motor adaptativo | Parcial | Must | 13 | 8 | 5 | **#3** | Delta-10 |
| TTH-11 | Spike hiperparámetros | No iniciado | Should | 5 | 0 | 5 | — | — |

### Historias de Usuario (HU)

| ID | Título corto | Estado | MoSCoW | SP tot | ejec | rest | S4 | Deltas |
|---|---|---|---|---|---|---|---|---|
| HU-01 | Acceso por rol (RBAC) | No iniciado | Must | 5 | 0 | 5 | **#2** | Delta-02 |
| HU-02 | Monitoreo estado actual | No iniciado | Must | 8 | 1 | 7 | — | Delta-06, Delta-07 |
| HU-03 | Predicción de congestión | No iniciado | Must | 5 | 1 | 4 | — | Delta-01, Delta-07 |
| HU-04 | Vista combinada | No iniciado | Must | 5 | 0 | 5 | — | Delta-07 |
| HU-05 | Estrategia de control activa | Parcial | Must | 5 | 2 | 3 | **#4** | Delta-08 (DHU-020) |
| HU-06 | Explicación de selección | Parcial | Must | 5 | 1 | 4 | — | Delta-09 |
| HU-07 | Notificación de cambios | No iniciado | Must | 5 | 0 | 5 | — | Delta-07 |
| HU-08 | Historial decisiones motor | No iniciado | Must | 8 | 0 | 8 | — | Delta-10 |
| HU-09 | Notas de turno *(MVP2)* | No iniciado | Should | 3 | 0 | 3 | — | — |
| HU-10 | Alerta transversal | No iniciado | Must | 13 | 0 | 13 | — | Delta-11, Delta-12 |
| HU-11 | Estado componentes (Op) | No iniciado | Must | 5 | 0 | 5 | — | — |
| HU-12 | Explicación degradado | No iniciado | Must | 5 | 0 | 5 | — | — |
| HU-13 | Salud técnica (Admin) | No iniciado | Must | 5 | 1 | 4 | — | Delta-12 |
| HU-14 | Métricas modelo | No iniciado | Must | 13 | 0 | 13 | — | Delta-12 |
| HU-15 | Config parámetros | No iniciado | Must | 13 | 0 | 13 | — | — |
| HU-16 | KPIs Gerente | No iniciado | Must | 13 | 1 | 12 | — | — |
| HU-17 | Comparativa periodos | No iniciado | Must | 8 | 0 | 8 | — | — |
| HU-18 | Drill-down (Gerente) *(MVP2)* | No iniciado | Could | 13 | 0 | 13 | — | — |
| HU-19 | Exportación PDF/Excel *(MVP2)* | No iniciado | Could | 13 | 0 | 13 | — | Delta-13 |
| HU-20 | Comparativa modelos *(MVP2)* | No iniciado | Could | 13 | 0 | 13 | — | Delta-13 |
| HU-21 | Escalamiento Op→Admin *(MVP2)* | No iniciado | Could | 13 | 0 | 13 | — | — |

### Totales de control (de `ESTIMACION_SP.md` §5)

- **Suma de este inventario de 32 elementos:** 90 SP (TTH, excl. Won't TTH-06) + 176 SP (21 HU) =
  **266 SP**; ejecutado 23 (TTH) + 7 (HU) = **30**.
- **SP total del backlog (excl. Won't):** **277** = 266 (este inventario) + **11 SP** de tres tareas
  derivadas de RNF (RNF-INT-02, RNF-INT-04, RNF-MNT-01) que `ESTIMACION_SP.md` estima por separado y
  que no forman parte de este inventario de 32 elementos. **Ejecutado:** 30 · **restante:** 247.
- **Estado de construcción:** 1 Completo (TTH-02), 5 Parcial (TTH-03, TTH-08, TTH-10, HU-05, HU-06),
  25 No iniciado, 1 Fuera de scope (TTH-06).
- **Sprint 4 comprometido:** 19 SP (5 elementos). **Postergado como Trabajo Futuro operacional:** ~205 SP.

## Sprint 4 vigente — orden de ejecución

Alcance comprometido (5 elementos / 19 SP), por semana (de `REPORTE_PLANIFICACION_SPRINT_4.md`):

1. **Semana 1 — TTH-01** (Auth JWT + bcrypt): 4 SP restantes. Habilita HU-01.
2. **Semana 2 — HU-01** (Acceso por rol / RBAC): 5 SP. Cierra Delta-02 (nomenclatura, DHU-022).
3. **Semana 3 — TTH-10** (Motor adaptativo): 5 SP restantes. Integraciones + persistencia `motor_decisions`.
4. **Semanas 3-4 — HU-05** (Estrategia de control activa): 3 SP restantes. Vista pasiva (DHU-020, Delta-08).
5. **Semana 4 — TTH-03** (Repo + CI cobertura): 2 SP restantes. Jobs CI faltantes + mypy.

**Estrategia ante imprevistos (§5.2 del reporte):** si el sprint se aprieta, descartar primero
**TTH-03**; luego reducir CAs no críticos de **HU-05** (robustez CA-05.4, redirección de login
CA-05.5); por último reducir **TTH-10** a RandomForest + persistencia sin integración auditada.

## Tareas de saneamiento derivadas de deltas

Tres deudas heredadas, fuera del alcance comprometido del Sprint 4, registradas para no perderlas:

- **SAN-01 ✓ resuelta** (2026-05-26, rama `san-06`): `torch` removido de
  `core_management_api/requirements.txt` + 6 archivos STGCN muertos eliminados de
  `core_management_api/src/prediction/`. La regla CLAUDE.md "no `torch`/`ultralytics` en el
  núcleo" permanece como guardia anti-regresión. Cierra simultáneamente C7.5 (TODO.md).
- **SAN-02 — Gemini huérfano (Delta-13):** `ReportModal.tsx` (API Gemini), `AIChatWidget.tsx` y
  `ThesisModal.tsx` existen sin HU/TTH que los respalde. **Decisión metodológica pendiente** (elevar
  a HU formal / deshabilitar / remover). El Artículo 21 (DHU-021) declara Gemini fuera de la
  arquitectura objetivo: la remoción es saneamiento diferido.
- **SAN-03 — `vision_aggregates` (Delta-05):** tabla planificada pero inexistente; el pipeline de
  visión persiste a CSV, no a BD. Tarea técnica: migración Alembic + cableado del aggregator a BD.
  **Preservar** la guardia de `CLAUDE.md`: NO migrar a `vision_tracks`/`vision_flows`.

## Trabajos Futuros operacionales (~205 SP)

El alcance postergado tras el loop MoSCoW⇄Planning Poker (subsistemas técnicos Should↓, bloques HU
de MVP1 reducidos, MVP2 completo, RNF transversal y TTH-06) está detallado en **§7 de
`REPORTE_PLANIFICACION_SPRINT_4.md`**. No se reproduce aquí.
