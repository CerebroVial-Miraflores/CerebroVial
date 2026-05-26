# Reporte de Planificación — Sprint 4 (CerebroVial)

> Generado: 2026-05-18 por Claude Code
> Versión: v1
> Fase de origen: 4.5 (Reporte de cierre)
> Documentos consolidados:
> - [`AUDITORIA_HU_CODIGO.md`](AUDITORIA_HU_CODIGO.md) (Fase 4.1)
> - [`MOSCOW_RATIFICADA.md`](MOSCOW_RATIFICADA.md) (Fase 4.2)
> - [`ESTIMACION_SP.md`](ESTIMACION_SP.md) (Fase 4.3)
> - [`DISTRIBUCION_SPRINTS.md`](DISTRIBUCION_SPRINTS.md) (Fase 4.4)
>
> **Propósito de este documento:** entregable académico de cierre de la planificación y punto de entrada operativo al sprint 4. Una sola lectura debe responder: "¿qué se hace en sprint 4, en qué orden, con qué SP, con qué riesgos identificados?"

---

## 1. Contexto

Proyecto académico individual **CerebroVial** — sistema inteligente de control adaptativo de semáforos para intersección de Miraflores, Lima. Estructurado en 4 sprints SCRUM; sprints 1-3 ya ejecutados sin estimación formal; sprint 4 por empezar.

La presente planificación reconstruyó forensicamente los sprints 1-3 (mapeo commits → HU/TTH → SP ejecutado), ratificó prioridades MoSCoW sobre 107 elementos del backlog, estimó SP con Planning Poker (incluyendo SP ejecutado retroactivo), y distribuyó la carga en los 4 sprints — disparando el loop MoSCoW⇄Planning Poker para resolver la brecha estructural entre alcance ambicioso del MVP1 documentado y velocity real disponible.

---

## 2. Resumen ejecutivo del Sprint 4

**Capacidad:** 15-20 SP (basada en velocity histórica de ~10 SP/sprint y un sprint potencialmente extendido).

**Alcance comprometido:** **19 SP** distribuidos en 5 elementos:

| # | Elemento | SP rest | Semana sugerida |
|---|---|---|---|
| 1 | **TTH-01** — Autenticación JWT + bcrypt (runtime completo) | 4 | 1ª |
| 2 | **HU-01** — Acceso diferenciado por rol (RBAC frontend + backend) | 5 | 2ª |
| 3 | **TTH-10** — Cierre integraciones del motor adaptativo | 5 | 3ª |
| 4 | **HU-05** — Visualización pasiva de la estrategia activa (refactor ControlView) | 3 | 3ª-4ª |
| 5 | **TTH-03** — Cierre CI (tests edge_device/shared/ia_prediction + mypy) | 2 | 4ª |

**Objetivo demostrable al cierre del sprint:** sistema con autenticación funcional, control de acceso por rol, motor adaptativo activo con persistencia mínima de decisiones, y vista pasiva del estado vigente del motor. Es el aporte de ingeniería central + el sustrato mínimo de operatividad. Cubre los 4 Objetivos del Producto en mínima viable defendible.

---

## 3. Estado del backlog al inicio del sprint 4

Tras la auditoría HU↔código (Fase 4.1, 32 elementos auditados):

| Estado | Cuenta | % |
|---|---|---|
| Completo | 1 (TTH-02 Docker) | 3% |
| Parcial | 5 (TTH-03, TTH-08, TTH-10, HU-05, HU-06) | 16% |
| No iniciado | 25 | 78% |
| Fuera de scope | 1 (TTH-06 Trabajos Futuros) | 3% |

**Cobertura efectiva ≈ 25%.** El sprint 4 cierra solo el 11% del SP restante del MVP1 original (19 de 179 SP Must); el resto se reclasifica como Trabajos Futuros operacionales.

**13 deltas críticos** documentados en `AUDITORIA_HU_CODIGO.md` §5. Los más relevantes para sprint 4:

- **Delta-01:** endpoint `POST /predictions/predict` existe pero con contrato divergente del spec de TTH-09. Sprint 4 lo deja como está (TTH-09 postergado).
- **Delta-04:** TTH-08 visión — refactor diferido por decisión del usuario. Código actual preservado.
- **Delta-08:** ControlView es playground request-response, no vista pasiva. **Decisión metodológica abierta para HU-05 del sprint 4.**
- **Delta-10:** sin tabla `motor_decisions`. Sprint 4 debe crear migración mínima para persistir decisiones del motor (parte del cierre TTH-10).
- **Delta-13:** features huérfanas en frontend (ReportModal con Gemini API, AIChatWidget, ThesisModal) sin HU. Sprint 4 las deja como están — decisión de elevar a HU formal o remover es sesión metodológica posterior.

---

## 4. Decisiones tomadas durante la planificación

### 4.1 De la Ceremonia MoSCoW (Fase 4.2)

- **Ratificación 99.1%** de las prioridades sugeridas del backlog (106 de 107 elementos sin ajuste).
- Único ajuste: TTH-11 (spike hiperparámetros) → Should, sin sugerida explícita previa.
- **Alcance MVP1 v1 declarado:** 16 HUs Must + 9 TTH Must = 209 SP total.

### 4.2 Del Planning Poker (Fase 4.3)

- **Anclas adoptadas:** HU-09 (3 SP baja), TTH-10 (13 SP alta).
- **Velocity histórica:** 10 SP/sprint promedio.
- **SP restante para sprint 4 con MoSCoW v1:** 179 SP Must — inviable (excede 12× la capacidad).
- **Alerta crítica:** loop MoSCoW⇄Planning Poker obligatorio.

### 4.3 De la Distribución de Sprints (Fase 4.4)

- **Fronteras de sprints declaradas tentativas** vía heurística de valles temporales en git log (sin cronograma formal).
- **Sprint 3 reportado como sprint metodológico** (0 SP de implementación; produjo todo el backlog formal: 21 HUs + 11 TTH + 22 RF + 53 RNF + 19 DHU).
- **Loop MoSCoW disparado y convergido en 2 iteraciones:**
  - Iteración 1: 13 elementos Must descendieron a Should/Could (152 SP postergados).
  - Iteración 2: 4 elementos adicionales descendieron (22 SP postergados adicionales).
  - Resultado: 5 elementos Must residuales con SP restante = 19 → cabe en capacidad sprint 4.
- **Bajadas de Must a Should/Could son operacionales del sprint 4**, NO ratificaciones del backlog formal. El backlog mantiene sus prioridades formales; este reporte declara qué subset es ejecutable en este sprint.

### 4.4 De la Auditoría (Fase 4.1)

- **Sin pendientes de consulta humana al cierre** de la auditoría: 32 elementos clasificados con confianza.
- **Decisión humana sobre Delta-04 (TTH-08 visión):** refactor diferido — el código actual preserva funcionalidad operativa básica; la restricción de CLAUDE.md de no refactorizar `edge_device/src/vision/` se levantará en iteración futura.

---

## 5. Plan de implementación del sprint 4

### 5.1 Orden de ejecución (4 semanas)

**Semana 1 — Auth runtime.** TTH-01 cierre: endpoint `POST /auth/login` con bcrypt + `get_current_user` dependency + tests unitarios. La tabla `users` y el modelo Alembic ya existen. Patrón estándar FastAPI + passlib. **Bajo riesgo.**

**Semana 2 — RBAC.** HU-01: routing condicional en frontend según rol del JWT + decorators FastAPI por endpoint validando claim `role` (HTTP 403 sin filtrar). Ocultación de rutas no accesibles. Reaprovecha tabla `users` con campo `role`. Tests unitarios para los 6 CAs. **Bajo riesgo,** depende de TTH-01.

**Semana 3 — Cierre motor + refactor ControlView.** Ejecución paralela:
- TTH-10 cierre: persistencia mínima de decisiones del motor (migración `motor_decisions` con timestamp/estrategia/razón/parámetros/anterior). Consumo de parámetros de HU-15 vía `.env` o config file (HU-15 postergada). Health check expuesto (consumo futuro por TTH-04 también postergado).
- HU-05 cierre: refactor ControlView de playground a vista pasiva del estado vigente del motor. Mostrar timestamp activación de estrategia. Conectar a endpoint `/control/recommend` con polling o invocación pre-rendered. **Medio riesgo** — requiere decisión metodológica previa (ver §6).

**Semana 4 — CI cierre + buffer.** TTH-03: agregar jobs CI para `edge_device/tests`, `shared/tests`, `ia_prediction_service/tests` + mypy verification. Configuración pura, sin código. Reserva: si las semanas previas se desbordan, sacrificar TTH-03 (deuda técnica conocida postergable) y dedicarla a estabilización de TTH-01/HU-01/TTH-10/HU-05.

### 5.2 Estrategia ante imprevistos

**Si el sprint 4 se comprime** (alguna tarea toma más SP que estimado):
1. **Descartar primero TTH-03 cierre** (2 SP, deuda técnica postergable).
2. **Reducir HU-05** declarando CAs no críticos (CA-05.4 robustez, CA-05.5 login redirect) como Could.
3. **Reducir TTH-10 cierre** a solo consumo del RandomForest + persistencia decisiones sin auditoría ampliada.

**Si el sprint 4 se expande** (sobra capacidad):
1. Subir parte de HU-06 (catálogo plantillas de razonamiento — 4 SP).
2. Empezar wiring TTH-08 con `vision_aggregates` BD (subir parcial — 4-5 SP).
3. Subir HU-02 vista intra-intersección con polling 5s (no SSE) — 5-7 SP.

---

## 6. Riesgos identificados

| Riesgo | Severidad | Mitigación propuesta |
|---|---|---|
| **R1 — Decisión semántica de HU-05 (Delta-08)** ✓ **Resuelto por DHU-020 (2026-05-20)** | Alta | **Resuelto** por DHU-020 (`DECISIONS_HU.md`): vista pasiva de HU-05 prevalece, playground se preserva como herramienta de Administrador, Delta-07/08/09 se refactorizan en bloque, persistencia de "estado vigente del motor" autorizada como cambio estructural deliberado. Sin trabajo no planificado adicional para el Sprint 4 (item #4 ya contempla 3 SP para HU-05). |
| **R2 — Integraciones de TTH-10:** la auto-clasificación del spec (5 pendientes) referenciaba TTH-09 GRU + TTH-07 SUMO + TTH-04 fallback + HU-15 parámetros + persistencia. Cuatro de cinco están postergados; el sprint 4 solo cierra "persistencia decisiones + consumo RandomForest + health endpoint mínimo". El cierre será **parcial respecto a la spec original**. | Alta | Documentar explícitamente en commit message y en el TODO del proyecto qué fracción de TTH-10 spec queda como Trabajo Futuro. |
| **R3 — Trabajo metodológico paralelo no contabilizado:** el sprint 4 también puede producir el SDD (Software Design Document) y la defensa de tesis, que consumen tiempo del estudiante no contabilizado en SP de implementación. | Media | Reservar al menos 30% del tiempo del sprint para trabajo metodológico paralelo. Si el SDD compite por tiempo, aceptar reducir TTH-03 cierre. |
| **R4 — Brecha academia vs implementación:** el backlog formal documenta 25 HUs Must del MVP1, pero el sprint 4 solo cierra 2. La defensa de tesis debe argumentar por qué esta brecha es razonable (proyecto académico individual + sprints sin estimación previa + descubrimiento progresivo del scope). | Alta | Usar este reporte + DISTRIBUCION_SPRINTS §9 como evidencia argumentativa de la brecha. La auditoría rigurosa de los 32 elementos + 13 deltas documentados refuerza la honestidad del análisis. |
| **R5 — Features huérfanas (Delta-13):** ReportModal usa Gemini API externa que tiene implicaciones de privacidad/datos no auditadas. AIChatWidget también es externo. | Baja-media | Sprint 4 los deja como están. Decisión metodológica posterior: elevar a HU formal, deshabilitar API key, o remover. |
| **R6 — Velocity podría caer:** los sprints 1-3 fueron 19+11+0 = 30 SP en 22 días calendario. El sprint 4 de 19 SP en ~28 días extendidos es factible solo si el estudiante mantiene foco. Imprevistos académicos paralelos (entregables, exámenes) reducen velocity. | Media | Monitoreo semanal de progreso real vs plan. Replanificación intra-sprint admitida por protocolo SCRUM. |

---

## 7. Trabajos futuros declarados (alcance postergado)

Esta planificación postergó 152 SP de elementos clasificados como Must en el backlog formal pero descendidos a Should/Could/Won't operacionales del sprint 4. Estos son los Trabajos Futuros para iteraciones académicas posteriores o post-académicas:

### 7.1 Subsistemas técnicos centrales (Should↓ tras loop)

- **TTH-04 Fallback en cascada** (13 SP) — robustez ante caídas de componentes.
- **TTH-07 Integración SUMO** (13 SP) — validación cuantitativa del sistema.
- **TTH-09 GRU servido** (12 SP restante) — modelo principal aspiracional declarado en D-006.
- **TTH-08 Refactor visión** (8 SP restante) — `vision_aggregates` BD + endpoint `/vision/state`.

### 7.2 HUs del MVP1 postergadas (Should↓/Could↓)

- **Bloque B (operador tiempo real):** HU-02 monitoreo, HU-03 predicción, HU-04 vista combinada, HU-06 explicación cierre, HU-07 notificación, HU-08 historial. ~33 SP.
- **Bloque C (degradado):** HU-10 alerta transversal, HU-11 componentes operador, HU-12 explicación degradado. ~23 SP.
- **Bloque D (admin):** HU-13 salud técnica, HU-14 métricas modelo, HU-15 configuración. ~30 SP.
- **Bloque F (gerente):** HU-16 KPIs, HU-17 comparativa. ~20 SP.

### 7.3 MVP2 completo (Could original)

- HU-09 (notas turno, 3 SP), HU-18 (drill-down, 13 SP), HU-19 (export PDF/Excel, 13 SP), HU-20 (comparativa modelos, 13 SP), HU-21 (escalamiento incidentes, 13 SP). 55 SP.

### 7.4 RNF transversales

- RNF-INT-02 WCAG 2.1 AA (5 SP), RNF-INT-04 coherencia visual (3 SP), RNF-MNT-01 catálogos como datos (3 SP). 11 SP.

### 7.5 TTH ya declarado Trabajos Futuros

- **TTH-06 capa DTOs** (8 SP gross) por DHU-014.
- **RNF-FLX-03 escalabilidad red urbana** (no estimable individual) — STGNN, F37.

**Total postergado:** ~205 SP de los 277 SP del backlog completo (74% del alcance documentado pasa a futuros).

---

## 8. Entregables académicos producidos por esta planificación

Los 5 documentos generados en `documentation/lean-inception/planificacion/` constituyen material académico citable directamente en el capítulo de metodología y ejecución de la tesis:

1. [`AUDITORIA_HU_CODIGO.md`](AUDITORIA_HU_CODIGO.md) — 414 líneas, 32 elementos auditados, 13 deltas críticos.
2. [`MOSCOW_RATIFICADA.md`](MOSCOW_RATIFICADA.md) — 330 líneas, 107 elementos clasificados.
3. [`ESTIMACION_SP.md`](ESTIMACION_SP.md) — 222 líneas, 32 elementos estimados + 3 RNF transversales, alerta de loop documentada.
4. [`DISTRIBUCION_SPRINTS.md`](DISTRIBUCION_SPRINTS.md) — 333 líneas, forense de sprints 1-3 + loop MoSCoW + plan sprint 4 + verificaciones.
5. **Este reporte** — síntesis ejecutiva y punto de entrada operativo.

Más los 3 protocolos normativos preexistentes (`CEREMONIA_MOSCOW.md`, `CEREMONIA_PLANNING_POKER.md`, `PROTOCOLO_DISTRIBUCION_SPRINTS.md`) que cierran el cuerpo metodológico completo.

---

## 9. Próximos pasos inmediatos

1. ~~**Decisión metodológica sobre R1** (semántica de HU-05 / Delta-08) antes de iniciar la semana 3 del sprint. Sesión corta con el estudiante.~~ **Resuelto por DHU-020 (2026-05-20):** ver §6 (R1) y `DECISIONS_HU.md`. Pendiente solo la ejecución del refactor de HU-05 en la semana 3 (item #4 del sprint, 3 SP).
2. **Arrancar sprint 4 con TTH-01** (semana 1): patrón JWT+bcrypt+get_current_user estándar. Ver TTH-01 spec en `TAREAS_TECNICAS_HABILITADORAS.md`.
3. **Monitorear progreso semanal** contra el plan §5.1. Si la velocity real diverge en >30%, replanificar intra-sprint.
4. **Mantener este reporte vivo:** cuando la realidad del sprint 4 diverja del plan, agregar sub-apartado "Ejecución real del sprint 4" documentando la divergencia (per protocolo §11 de `PROTOCOLO_DISTRIBUCION_SPRINTS.md`).
5. **Eventual sesión metodológica posterior** para:
   - Ratificar formalmente en el backlog las bajadas de Must→Should/Could del loop (si se decide), actualizando `DECISIONS_HU.md` con un nuevo DHU.
   - Resolver Delta-02 (nomenclatura roles inconsistente) antes de TTH-01.
   - Resolver Delta-13 (features huérfanas Gemini/AIChat).
   - Eventualmente formalizar la cobertura efectiva del MVP1 y declarar oficialmente los Trabajos Futuros operacionales como tales en `EVOLUCION_TESIS.md`.

---

## 10. Conclusión

El proyecto CerebroVial entra al sprint 4 con:

- **Backlog formal completo y auditado** (21 HUs + 11 TTH + 22 RF + 53 RNF + 19 DHU).
- **Sustento técnico parcial pero defendible:** motor adaptativo central operativo, infra Docker + CI estable, sustrato de auth modelado.
- **Brecha estructural reconocida y documentada** entre alcance MVP1 ambicioso (209 SP Must) y velocity real del estudiante individual (10 SP/sprint promedio).
- **Plan de sprint 4 honesto y viable:** 19 SP en 5 elementos cierran lo mínimo demostrable defendible (auth + RBAC + vista pasiva motor + cierre integraciones + CI completo).
- **205 SP postergados como Trabajos Futuros** con trazabilidad completa de la decisión.

La defensa de tesis puede sustentarse en: (a) el aporte de ingeniería central (motor adaptativo Webster + MaxPressure + MTC) está construido y demostrable; (b) el sustrato académico (backlog formal + decisiones metodológicas + auditoría rigurosa) está completo; (c) la brecha entre alcance documentado y alcance implementado está honestamente reconocida y explicada por la naturaleza del proyecto académico individual y la ejecución de sprints sin estimación previa formal.
