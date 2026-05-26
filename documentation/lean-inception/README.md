# Lean Inception — Documentación del Producto CerebroVial

Esta carpeta contiene el cuerpo documental del producto CerebroVial: contexto metodológico, backlog formal, requisitos, decisiones de respaldo y planificación SCRUM.

## Punto de entrada

**[`BACKLOG_OVERVIEW.md`](BACKLOG_OVERVIEW.md)** — visión de conjunto del producto en 10 minutos (4 Personas, 4 Objetivos, mapa de 21 HUs + 11 TTH, navegación al detalle).

## Estructura de carpetas

```
.
├── README.md                ← este archivo
├── BACKLOG_OVERVIEW.md      ← entry point del producto
│
├── 1-contexto/              ← marco metodológico (3 docs)
├── 2-backlog/               ← historias de usuario + tareas técnicas + features (10 docs)
├── 3-requisitos/            ← RF + RNF según ISO 25010 (2 docs)
├── 4-decisiones/            ← trazabilidad de decisiones (2 docs)
└── planificacion/           ← artefactos del proceso SCRUM (3 protocolos + 5 entregables + 2 manuales)
```

## Orden de lectura sugerido

| # | Carpeta / Doc | Cuándo abrir |
|---|---|---|
| 0 | `BACKLOG_OVERVIEW.md` | Primer contacto con el proyecto. |
| 1 | `1-contexto/` | Entender el origen, la fundamentación metodológica y la narrativa del proyecto en sus 4 fases. |
| 2 | `2-backlog/` | Detalle implementable: 21 HUs operativas organizadas en bloques A-F + 5 MVP2 + 11 TTH + 41 features. |
| 3 | `3-requisitos/` | Catálogo formal de 22 RF + 53 RNF según ISO 25010, derivado del backlog. |
| 4 | `4-decisiones/` | Trazabilidad de decisiones técnicas (D-001 a D-009) y metodológicas (DHU-001 a DHU-022). |
| 5 | `planificacion/` | Ceremonias SCRUM ejecutadas (MoSCoW, Planning Poker, Distribución de sprints) y planificación del sprint 4. |

## Navegación por necesidad

| Si necesitas... | Abre |
|---|---|
| Entender el producto sin detalle implementable | [`BACKLOG_OVERVIEW.md`](BACKLOG_OVERVIEW.md) |
| Lectura humana de las HUs en formato corto | [`2-backlog/HU_LITE.md`](2-backlog/HU_LITE.md) |
| Implementar o auditar HUs con CAs numerados | [`2-backlog/HU_BLOQUE_A.md`](2-backlog/HU_BLOQUE_A.md) a [`HU_BLOQUE_F.md`](2-backlog/HU_BLOQUE_F.md), [`HU_MVP2.md`](2-backlog/HU_MVP2.md) |
| Tareas técnicas habilitadoras | [`2-backlog/TAREAS_TECNICAS_HABILITADORAS.md`](2-backlog/TAREAS_TECNICAS_HABILITADORAS.md) |
| Detalle de features (origen, clasificación MVP) | [`2-backlog/FEATURE_BACKLOG_DETALLADO.md`](2-backlog/FEATURE_BACKLOG_DETALLADO.md) |
| Requisitos formales con trazabilidad completa | [`3-requisitos/REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md`](3-requisitos/REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md) |
| Requisitos en formato lite | [`3-requisitos/RF_RNF_LITE.md`](3-requisitos/RF_RNF_LITE.md) |
| Decisiones técnicas del producto | [`4-decisiones/DECISIONS.md`](4-decisiones/DECISIONS.md) |
| Decisiones metodológicas sobre HUs | [`4-decisiones/DECISIONS_HU.md`](4-decisiones/DECISIONS_HU.md) |
| Workshop Lean Inception + investigación de marco | [`1-contexto/LEAN_INCEPTION_CEREBROVIAL.md`](1-contexto/LEAN_INCEPTION_CEREBROVIAL.md), [`LEAN_INCEPTION_INVESTIGACION.md`](1-contexto/LEAN_INCEPTION_INVESTIGACION.md) |
| Narrativa de las 4 fases del proyecto | [`1-contexto/EVOLUCION_TESIS.md`](1-contexto/EVOLUCION_TESIS.md) |
| Planificación SCRUM (MoSCoW + Planning + Distribución sprints) | [`planificacion/`](planificacion/) — empezar por [`PROMPT_ARRANQUE_CLAUDE_CODE_v2.md`](planificacion/PROMPT_ARRANQUE_CLAUDE_CODE_v2.md) |
| Resumen ejecutivo del sprint 4 a ejecutar | [`planificacion/REPORTE_PLANIFICACION_SPRINT_4.md`](planificacion/REPORTE_PLANIFICACION_SPRINT_4.md) |

## Nota sobre referencias en backticks

Dentro de los documentos de esta carpeta, las referencias cruzadas se escriben con backticks textuales (e.g., `` ver `DECISIONS_HU.md` ``) en lugar de markdown links navegables. Esto es deliberado y se preserva tras la reorganización: el nombre del archivo es único en toda la carpeta `lean-inception/`, así que la referencia sigue siendo resoluble por búsqueda. Los markdown links navegables solo se usan en este README y en los documentos de `planificacion/`.
