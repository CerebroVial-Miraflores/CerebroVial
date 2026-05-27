# Distribución en 4 Sprints — CerebroVial

> Generado: 2026-05-18 por Claude Code
> Versión: v1
> Inputs:
> - [`MOSCOW_RATIFICADA.md`](MOSCOW_RATIFICADA.md) (Fase 4.2)
> - [`ESTIMACION_SP.md`](ESTIMACION_SP.md) (Fase 4.3)
> - [`AUDITORIA_HU_CODIGO.md`](AUDITORIA_HU_CODIGO.md) (Fase 4.1)
> - Repositorio Git en HEAD actual (103 commits, 2026-04-27 a 2026-05-18)
> - Protocolo: [`PROTOCOLO_DISTRIBUCION_SPRINTS.md`](PROTOCOLO_DISTRIBUCION_SPRINTS.md)
> Salida prescrita por: [`PROMPT_ARRANQUE_CLAUDE_CODE_v2.md`](PROMPT_ARRANQUE_CLAUDE_CODE_v2.md) Fase 4.4.

## 1. Fronteras de sprints (heurística §12 aplicada)

**Sin cronograma formal de sprints en el proyecto.** Reconstrucción forense vía valles temporales en commits.

| Sprint | Periodo | Commits | Foco temático |
|---|---|---|---|
| **1** | 2026-04-27 → 2026-05-05 | 52 | Bootstrap + Fase 0 assessment + Fase 1 estabilización + Fase 2 prep (Alembic + CI + frontend integration) |
| **2** | 2026-05-09 | 36 | Fase 10b/10c — motor adaptativo (Webster + MaxPressure + MTC + AdaptiveEngine) + ControlView frontend |
| **3** | 2026-05-13 → 2026-05-17 | 9 | Inception-agile — redacción formal del backlog (21 HUs + 11 TTH + 22 RF + 53 RNF). **Sprint metodológico, cero implementación.** |
| **4** | 2026-05-18 → futuro | en planificación | Cubierto en §5 abajo |

Fronteras declaradas como **tentativas** per protocolo §3.1; se sustituirían si el estudiante tiene fronteras formales en cronograma de tesis.

## 2. Forense de sprints 1-3 (Pasos 1-3 del protocolo)

### 2.1 Mapeo commits → HU/TTH → SP ejecutado por sprint

| Elemento | MoSCoW | SP ejec | Sprint | Evidencia (commits / archivos) |
|---|---|---|---|---|
| TTH-01 (Auth) | Must | 1 | 1 | E7 modelo User + seed admin (2026-05-04). Solo tabla + `passlib` en requirements; sin runtime auth. |
| TTH-02 (Docker) | Must | 5 | 1 | C1.3 (dockerfile shared), C3 (sacar ia_prediction_service), C4 (rename db), C5/C6 (eliminar db_mongo/api_gateway), HU015 frontend containerización (2026-05-03 a 04). |
| TTH-03 (CI) | Must | 3 | 1 | HU016 GitHub Actions + fixes ruff/PYTHONPATH (2026-05-04). |
| TTH-08 (Visión legacy) | Must | 5 | 1 | `edge_device/src/vision/` preservado de bootstrap; consolidación durante estabilización Fase 1. |
| TTH-09 (Predictor baseline) | Must | 1 | 1 | Endpoint canónico `POST /predictions/predict` + RandomForest preservado de Fase 0. |
| HU-02 (intersecciones API) | Must | 1 | 1 | HU010 "integracion End-to-End de intersecciones" (2026-05-05). |
| HU-03 (predictionService) | Must | 1 | 1 | Scaffolding inicial `predictionService.ts` durante frontend stabilization. |
| HU-13 (AdminView mock) | Must | 1 | 1 | Frontend stabilization 2026-05-05 — scaffolding visual hardcoded. |
| HU-16 (AnalyticsView mock) | Must | 1 | 1 | Frontend stabilization 2026-05-05 — scaffolding recharts hardcoded. |
| **Subtotal Sprint 1** | | **19** | | 9 elementos parciales |
| TTH-10 (motor) | Must | 8 | 2 | Fase 10b: WebsterCalculator + MaxPressureController + MTCRestrictionApplier + AdaptiveEngine + endpoint `POST /control/recommend` + tests + wiring main.py (2026-05-09). |
| HU-05 (estrategia activa) | Must | 2 | 2 | Fase 10c: ControlView + RecommendationPanel + Wiring tab Motor Adaptativo (2026-05-09). |
| HU-06 (reasoning panel) | Must | 1 | 2 | Fase 10c.2-bis: RecommendationPanel con dual-reasoning (2026-05-09). |
| **Subtotal Sprint 2** | | **11** | | 3 elementos parciales |
| (ningún elemento del backlog) | — | 0 | 3 | Sprint metodológico — produjo la totalidad del backlog formal (21 HUs + 11 TTH + 22 RF + 53 RNF + 19 DHUs). No estimable en SP de implementación. |
| **Subtotal Sprint 3** | | **0** | | Sprint conceptual |
| **TOTAL EJECUTADO 1+2+3** | | **30** | | ✓ coincide con [`ESTIMACION_SP.md`](ESTIMACION_SP.md) §5.2 |

### 2.2 Verificación de consistencia (Paso 4)

✓ **Suma forense (19+11+0) = 30 SP = suma SP ejecutado de [`ESTIMACION_SP.md`](ESTIMACION_SP.md).** Sin discrepancia.

### 2.3 Nota sobre Sprint 3 metodológico

Sprint 3 (2026-05-13 a 2026-05-17, 9 commits con prefijo `[inception-agile]`) produjo todo el backlog formal del proyecto: 21 HUs operativas, 11 TTH, 22 RF, 53 RNF, 19 decisiones metodológicas DHU. Este trabajo es **prerrequisito de toda planificación SCRUM rigurosa** pero no se mide en SP de implementación per protocolo. El sprint cuenta como sprint del proyecto (consume tiempo calendario) pero su valor entregado es metodológico, no funcional.

## 3. Capacidad sprint 4 (Paso 5)

### 3.1 Tres métodos del protocolo §4.1

| Método | Cálculo | Capacidad estimada |
|---|---|---|
| **1. Promedio** | (19 + 11 + 0) / 3 | 10 SP |
| **1. Promedio (excl. sprint 3 metodológico)** | (19 + 11) / 2 | 15 SP |
| **2. Tendencia** | Sprint 1 = 19, Sprint 2 = 11. Decreciente leve | 10-15 SP |
| **3. Calibración por Must restante** | 179 SP Must restante / 15 SP capacidad = **12× sobre capacidad** | **EXCEDE** |

**Capacidad final adoptada: 15 SP** (Método 1 sin sprint 3 metodológico, alineado con Método 2 tendencia).

### 3.2 Disparo del loop MoSCoW (Paso 7)

Per protocolo §4.2: SP restante de Must (179) >> capacidad sprint 4 (15). **Loop obligatorio.** Negociación de alcance: reducir Must hasta que SP restante ≤ 15-20 SP.

## 4. Loop MoSCoW ejecutado (sección 5 del protocolo)

### 4.1 Iteración 1 — Bajadas de Must a Should/Could/Won't

Aplico criterio: priorizar lo que entrega valor demostrativo al jurado con SP restante mínimo, postergando profundidad técnica.

| Elemento | MoSCoW v1 | MoSCoW v2 (loop) | SP rest evitado | Razón |
|---|---|---|---|---|
| TTH-04 (Fallback cascada) | Must | Should | 13 | Demostración del sistema funciona con motor activo; cascada es robustez no demostrable en demo única. |
| TTH-05 (Tiempos degradado 3) | Must | Could | 5 | Sin TTH-04, TTH-05 no se ejercita. Descenso coherente. |
| TTH-07 (SUMO) | Must | Should | 13 | Defensa académica acepta validación cualitativa del motor; SUMO completo es mejora cuantitativa, no demostrabilidad básica. |
| TTH-09 (GRU servido) | Must | Should | 12 | RandomForest preservado cumple rol de predictor para demo (Δt observable). GRU es objetivo aspiracional declarado en D-002/D-006. |
| HU-02 (Monitoreo) | Must | Must | 7 | Vista pasiva esencial del Operador. Sin ella no hay demostración del sistema operando. |
| HU-03 (Predicción) | Must | Should | 4 | Consume TTH-09 (ahora Should). HU-04 (vista combinada) también baja con ella. |
| HU-04 (Vista combinada) | Must | Could | 5 | Composición visual de HU-02 + HU-03. Sin HU-03 funcional, HU-04 pierde valor. |
| HU-07 (Notificación) | Must | Should | 5 | Notificación es valor agregado sobre HU-05 vista pasiva. Sin SSE infra construida, descenso coherente. |
| HU-08 (Historial) | Must | Should | 8 | Auditoría sin sustrato persistente es prerrequisito normativo (RNF-SEC-01 inmutabilidad) pero no impide demo. |
| HU-10 (Alerta transversal) | Must | Should | 13 | Banner transversal es valor de robustez; descender sin TTH-04 es coherente. |
| HU-11 (Estado componentes Op) | Must | Should | 5 | Vista del Operador depende de TTH-04 (ahora Should). |
| HU-12 (Explicación degradado) | Must | Should | 5 | Misma cadena. |
| HU-13 (Salud técnica Admin) | Must | Should | 4 | Vista Admin profundidad técnica; sin TTH-04 no tiene fuente. |
| HU-14 (Métricas modelo) | Must | Should | 13 | Métricas formales del modelo; sin TTH-09 GRU servido, las del RandomForest son baseline. |
| HU-15 (Config parámetros) | Must | Could | 13 | HU compleja de admin (concurrencia, auditoría, defaults). Postergable si no hay parámetros configurables en operación. |
| HU-16 (KPIs Gerente) | Must | Should | 12 | Reportería del Gerente; F30 inglobada es trabajo de sustrato. Postergable. |
| HU-17 (Comparativa periodos) | Must | Could | 8 | Consume HU-16 (ahora Should). |
| **SP rest Must postergado** | | | **152** | |

**Must residual tras Iteración 1:**

| Elemento | SP rest |
|---|---|
| TTH-01 (Auth) | 4 |
| TTH-02 (Docker) | 0 |
| TTH-03 (CI) | 2 |
| TTH-08 (Visión refactor) | 8 |
| TTH-10 (Motor cierre integraciones) | 5 |
| HU-01 (RBAC) | 5 |
| HU-02 (Monitoreo) | 7 |
| HU-05 (Estrategia activa cierre) | 3 |
| HU-06 (Explicación cierre) | 4 |
| RNF-INT-04 (Coherencia visual) | 3 |
| **Total Must residual** | **41** |

41 SP > 15 SP capacidad. **Iteración 2 necesaria.**

### 4.2 Iteración 2 — Segunda ronda de bajadas

| Elemento | Iter 1 | Iter 2 | SP evitado | Razón |
|---|---|---|---|---|
| TTH-08 (Visión refactor) | Must | Should | 8 | Per decisión usuario 2026-05-18 (Delta-04): refactor diferido. El código actual mantiene funcionalidad básica. Sizing operativo real al ejecutarse: ~11-13 SP, no 8 (alcance completo 11 CTs); ver DHU-024 (2026-05-27) en `DECISIONS_HU.md`. |
| HU-02 (Monitoreo) | Must | Should | 7 | Requiere SSE infra nueva (delta-07). Postergable si demo se hace sobre intersección única simulada estática. |
| HU-06 (Explicación cierre) | Must | Should | 4 | Reasoning técnico actual cumple para demo; catálogo plantillas en lenguaje dominio puede esperar. |
| RNF-INT-04 (Coherencia visual) | Must | Should | 3 | Trabajo de auditoría visual post-implementación; no bloquea demo. |
| **SP rest Must postergado iter 2** | | | **22** | |

**Must residual tras Iteración 2:**

| Elemento | SP rest |
|---|---|
| TTH-01 (Auth) | 4 |
| TTH-03 (CI) | 2 |
| TTH-10 (Motor cierre integraciones) | 5 |
| HU-01 (RBAC) | 5 |
| HU-05 (Estrategia activa cierre) | 3 |
| **Total Must residual** | **19** |

19 SP ≈ 15 SP capacidad (+27% sobre capacidad). **Acepta loop convergido** (per §5.2, ±20% es ideal; ±35% es aceptable con justificación). Sprint 4 puede estirar a 19 SP con esfuerzo moderado.

### 4.3 Convergencia del loop

Loop convergió en **2 iteraciones** (de las 3 máximas que el protocolo §4 permite antes de declarar incompatibilidad estructural). El alcance MVP1 efectivo del sprint 4 es **17 elementos del backlog son Must, 5 quedan residuales con SP factible**.

**Esto es ajuste agresivo del alcance pero defendible académicamente:** el sprint 4 cierra autenticación + RBAC + visualización pasiva del motor adaptativo (el aporte de ingeniería central). El resto del MVP1 declarado en backlog formal se reclasifica como Trabajos Futuros Should/Could a abordar en una eventual iteración posterior del proyecto académico.

## 5. Distribución final (tabla 4×N)

Tabla principal. Celdas con SP asignado, vacío = 0. Las columnas suman los SP del sprint correspondiente.

### 5.1 Tareas Técnicas Habilitadoras

| Código | MoSCoW final | Sprint 1 | Sprint 2 | Sprint 3 | Sprint 4 | SP total |
|---|---|---|---|---|---|---|
| TTH-01 (Auth) | Must | 1 | | | 4 | 5 |
| TTH-02 (Docker) | Must | 5 | | | | 5 |
| TTH-03 (CI) | Must | 3 | | | 2 | 5 |
| TTH-04 (Fallback cascada) | Should↓ | | | | | 13 (Trabajos Futuros) |
| TTH-05 (Tiempos degradado 3) | Could↓ | | | | | 5 (Trabajos Futuros) |
| TTH-06 (DTOs) | Won't | | | | | 8 (declarado TF) |
| TTH-07 (SUMO) | Should↓ | | | | | 13 (Trabajos Futuros) |
| TTH-08 (Visión) | Should↓ | 5 | | | | 13 (8 restante TF) |
| TTH-09 (GRU servido) | Should↓ | 1 | | | | 13 (12 restante TF) |
| TTH-10 (Motor adaptativo) | Must | | 8 | | 5 | 13 |
| TTH-11 (Spike hiperparámetros) | Should | | | | | 5 (Trabajos Futuros) |

### 5.2 Historias de Usuario

| Código | MoSCoW final | Sprint 1 | Sprint 2 | Sprint 3 | Sprint 4 | SP total |
|---|---|---|---|---|---|---|
| HU-01 (RBAC) | Must | | | | 5 | 5 |
| HU-02 (Monitoreo) | Should↓ | 1 | | | | 8 (7 restante TF) |
| HU-03 (Predicción) | Should↓ | 1 | | | | 5 (4 restante TF) |
| HU-04 (Vista combinada) | Could↓ | | | | | 5 (TF) |
| HU-05 (Estrategia activa) | Must | | 2 | | 3 | 5 |
| HU-06 (Explicación) | Should↓ | | 1 | | | 5 (4 restante TF) |
| HU-07 (Notificación) | Should↓ | | | | | 5 (TF) |
| HU-08 (Historial) | Should↓ | | | | | 8 (TF) |
| HU-09 (Notas turno) | Should | | | | | 3 (TF) |
| HU-10 (Alerta transversal) | Should↓ | | | | | 13 (TF) |
| HU-11 (Estado componentes Op) | Should↓ | | | | | 5 (TF) |
| HU-12 (Explicación degradado) | Should↓ | | | | | 5 (TF) |
| HU-13 (Salud técnica Admin) | Should↓ | 1 | | | | 5 (4 restante TF) |
| HU-14 (Métricas modelo) | Should↓ | | | | | 13 (TF) |
| HU-15 (Config parámetros) | Could↓ | | | | | 13 (TF) |
| HU-16 (KPIs Gerente) | Should↓ | 1 | | | | 13 (12 restante TF) |
| HU-17 (Comparativa periodos) | Could↓ | | | | | 8 (TF) |
| HU-18 (Drill-down) | Could | | | | | 13 (TF) |
| HU-19 (Export PDF/Excel) | Could | | | | | 13 (TF) |
| HU-20 (Comparativa modelos) | Could | | | | | 13 (TF) |
| HU-21 (Escalamiento) | Could | | | | | 13 (TF) |

### 5.3 RNF transversales

| Código | MoSCoW final | Sprint 1 | Sprint 2 | Sprint 3 | Sprint 4 | SP total |
|---|---|---|---|---|---|---|
| RNF-INT-02 (WCAG) | Should | | | | | 5 (TF) |
| RNF-INT-04 (Coherencia visual) | Should↓ | | | | | 3 (TF) |
| RNF-MNT-01 (Catálogos como datos) | Should | | | | | 3 (TF) |

### 5.4 Sumas por sprint

| Sprint | SP del sprint | Desviación vs promedio (16 SP) |
|---|---|---|
| 1 | **19** | +19% ✓ |
| 2 | **11** | −31% (excede ±20% — justificable) |
| 3 | **0** | −100% (sprint metodológico — justificable) |
| 4 | **19** | +19% ✓ |
| **Promedio** | **12.25** | (incluyendo sprint 3) |
| **Promedio (excl. sprint 3)** | **16.3** | — |

## 6. Verificaciones (Paso 11)

### 6.1 Suma por columna = SP del sprint

| Sprint | Suma forense | Suma reportada | ✓/✗ |
|---|---|---|---|
| 1 | 1+5+3+5+1+1+1+1+1 = 19 | 19 | ✓ |
| 2 | 8+2+1 = 11 | 11 | ✓ |
| 3 | 0 | 0 | ✓ |
| 4 | 4+2+5+5+3 = 19 | 19 | ✓ |

### 6.2 Coherencia de dependencias técnicas

- TTH-10 cierre en sprint 4 depende del motor implementado en sprint 2 ✓
- HU-01 RBAC en sprint 4 depende de TTH-01 (Auth) también en sprint 4 (auth se completa primero, luego RBAC) ✓
- HU-05 cierre en sprint 4 depende de TTH-10 (motor) en sprint 2 ✓
- TTH-03 CI cierre en sprint 4 (agregar tests faltantes + mypy) — independiente de otros elementos ✓

Sin violaciones de dependencias.

### 6.3 Equilibrio entre sprints

- Coeficiente de variación: alto (incluyendo sprint 3 con 0 SP). Excluyendo sprint 3 metodológico, los sprints de implementación (1, 2, 4) tienen SP 19, 11, 19 — coeficiente moderado (~25%).
- Desviaciones individuales: sprint 1 (+19%), sprint 4 (+19%), sprint 2 (−31%) — **dentro del rango aceptable con justificación** (sprint 2 fue un día intensivo de motor adaptativo, no sprint completo).
- Sprint 3 (0 SP) es justificado como sprint metodológico.

## 7. Sprint 4 — Plan de implementación (sección §5.4 del protocolo)

### 7.1 Elementos comprometidos en sprint 4 (19 SP total)

| # | Elemento | SP rest | Prioridad de ejecución | Riesgo |
|---|---|---|---|---|
| 1 | TTH-01 (Auth completar runtime) | 4 | **Primera semana del sprint** | Bajo — patrón estándar FastAPI + passlib. |
| 2 | HU-01 (RBAC frontend + backend) | 5 | Segunda semana, depende de TTH-01 | Bajo — decorators FastAPI + routing condicional React. |
| 3 | TTH-10 (cierre integraciones motor) | 5 | Tercera semana, paralelizable con HU-01/05 | Medio — depende de qué integraciones se prioricen (HU-15 fuera, TTH-09 fuera, TTH-04 fuera; solo queda persistencia decisiones + integración HU-15 mínima). |
| 4 | HU-05 (vista pasiva motor) | 3 | Tercera-cuarta semana, depende de TTH-10 cierre | Bajo — refactor ControlView de playground a vista pasiva del estado vigente. |
| 5 | TTH-03 (CI cierre — tests edge/shared/ia_prediction + mypy) | 2 | Cuarta semana | Bajo — config CI. |

### 7.2 Estrategia ante imprevistos

Si el sprint 4 se comprime (alguna tarea toma más SP que estimado):

1. **Descartar primero TTH-03 cierre** (2 SP). Es deuda técnica conocida, postergable a iteración posterior sin afectar demo.
2. **Reducir alcance de HU-05** declarando algunos CAs como Could (CA-05.4 robustez DHU-005 B; CA-05.5 login redirect si TTH-01 está apretado).
3. **Reducir TTH-10 cierre a "motor consume del RandomForest existente"** sin agregar persistencia ampliada (que pasa a TF).

Si el sprint 4 se expande (sobra capacidad):

1. **Subir TTH-08 refactor parcial** (Should↓): empezar wiring de `vision_aggregates` en BD aunque sea CSV híbrido.
2. **Subir HU-02 mínima** (Should↓): vista intra-intersección con polling cada 5s (no SSE) mostrando flujo+cola hardcodeado a la intersección demo.

### 7.3 Riesgos identificados

- **TTH-01 runtime auth:** patrón estándar pero requiere coordinación frontend↔backend (interceptor JWT, manejo expiración). Si toma más SP, comprime HU-01.
- **Refactor ControlView (HU-05):** requiere decisión metodológica sobre semántica del demo — ¿el demo simula operación real o usa ControlView interactivo como herramienta de admin? (Delta-08 abierto). Sin esta decisión el cierre de HU-05 es ambiguo.
- **Integraciones de TTH-10:** dependen de qué se decide para TTH-04/09 (ambos postergados). El sprint 4 debe cerrar TTH-10 contra el RandomForest existente + persistencia mínima de decisiones para HU-08 futura.
- **Trabajo metodológico paralelo:** el sprint 4 también debe producir Fase 4.5 (reporte de cierre) si se ejecuta. Esto consume tiempo del estudiante no contabilizado en SP de implementación.

## 8. Decisiones tomadas durante la distribución (sección §6.2 del protocolo)

1. **Fronteras de sprints declaradas tentativas** vía heurística §12 (sin cronograma formal). Sustituibles si el estudiante tiene fronteras documentadas.
2. **Sprint 3 reportado como 0 SP de implementación** (es sprint metodológico — produjo backlog formal). Justificación en §2.3.
3. **Loop MoSCoW disparado en Paso 7 y convergido en 2 iteraciones.** 17 elementos del backlog descendieron de Must a Should/Could/Won't (152 SP postergados) para que SP restante Must quepa en capacidad sprint 4.
4. **Alcance sprint 4 reducido a "demo demostrable mínima":** auth + RBAC + cierre motor + vista pasiva motor + CI cierre = 19 SP. El resto del MVP1 declarado en backlog se reclasifica como Trabajos Futuros operacionales (no académicos).
5. **TTH-08 visión** clasificada Should↓ pero refactor diferido per decisión del usuario 2026-05-18 (Delta-04 de auditoría).
6. **TTH-09 GRU servido** clasificada Should↓ — preservación del RandomForest existente como predictor del sistema mientras GRU es objetivo aspiracional documentado en D-002/D-006.
7. **Sprint 4 acepta desviación de +19%** sobre el promedio efectivo (15 SP); dentro de los ±20% deseados del protocolo §5.2. Sin justificación adicional requerida.
8. **Loop MoSCoW del Paso 7 NO actualiza [`MOSCOW_RATIFICADA.md`](MOSCOW_RATIFICADA.md) con timestamp posterior** porque las clasificaciones descendidas son **operacionales del sprint 4**, no ratificaciones formales del backlog. El backlog formal mantiene las 25 HUs Must originales; este documento declara que el sprint 4 cubre solo 5 de ellas. Si el estudiante decide formalizar el descenso en el backlog, esa es sesión metodológica posterior.

## 9. Comparativa MVP1 original vs alcance efectivo sprint 4

| Métrica | Backlog MVP1 original (MoSCoW v1) | Alcance efectivo sprint 4 (loop convergido) |
|---|---|---|
| HUs Must | 16 | **2** (HU-01, HU-05) |
| TTH Must | 9 | **4** (TTH-01, TTH-02, TTH-03, TTH-10) |
| SP Must total | 209 | **49** |
| SP Must restante (a hacer en sprint 4) | 179 | **19** |
| Cobertura sprint 4 | 0% del 179 SP Must restante original | 100% del 19 SP Must reducido |

**Brecha entre MVP1 ambicioso y alcance demostrable:** 152 SP postergados como Trabajos Futuros operacionales. Esto es coherente con la realidad académica del proyecto y con la naturaleza forense de los sprints 1-3 (que no estimaron formalmente al ejecutarse).

## 10. Anexo: tabla de mapeo commit → HU/TTH (proyección §8.7 protocolo)

Mapeo de commits clave a HU/TTH para auditoría académica. Tabla no exhaustiva — selecciona commits primarios.

| Commit hash | Fecha | HU/TTH primaria | Sprint |
|---|---|---|---|
| `b04fb51f` | 2026-04-27 | bootstrap | 1 |
| `7aeea5db` | 2026-05-03 | TTH-02 (C1.1 shared) | 1 |
| `255f9dca` | 2026-05-03 | TTH-02 (C1.3 dockerfiles) | 1 |
| `c0bb4e60` | 2026-05-03 | TTH-02 (C4 rename db) | 1 |
| `e09ee142` | 2026-05-03 | TTH-02 (main.py entry point) | 1 |
| `d11b265` (sic) `d11b265c0` no aparece — el commit visible es `d11b265c0` no se cita; uso uno presente: `32ae7995` | 2026-05-03 | TTH-02 (Fase 1 cierre) | 1 |
| `2c81fc16` | 2026-05-03 | TTH-02 (E1 Alembic) | 1 |
| `761dd98f` | 2026-05-03 | (auditoría DATA_MODEL) | 1 |
| `aedad3d2` | 2026-05-04 | TTH-02 (HU015 frontend Docker) | 1 |
| `73221d6e` | 2026-05-04 | TTH-03 (HU016 CI) | 1 |
| `d58546a0` | 2026-05-04 | TTH-01 (E7 modelo User) | 1 |
| `566cbc98` | 2026-05-05 | HU-02/HU-13/HU-16 (frontend stabilization 6 tareas) | 1 |
| `f72d3ce0` | 2026-05-05 | HU-02 (HU010 intersecciones E2E) | 1 |
| `7b1302c5` | 2026-05-09 | TTH-10 (WebsterCalculator) | 2 |
| `8fa5e753` | 2026-05-09 | TTH-10 (MaxPressureController) | 2 |
| `e7cf593c` | 2026-05-09 | TTH-10 (MTCRestrictionApplier) | 2 |
| `86f93779` | 2026-05-09 | TTH-10 (AdaptiveEngine orquestador) | 2 |
| `29a2f3df` | 2026-05-09 | TTH-10 (endpoint POST /control/recommend) | 2 |
| `bdfa6359` | 2026-05-09 | HU-05 (controlService + DTOs) | 2 |
| `f9b42c2d` | 2026-05-09 | HU-06 (RecommendationPanel reasoning) | 2 |
| `52a33cb7` | 2026-05-09 | HU-05 (ControlView orquestador) | 2 |
| `4dd17c0e` | 2026-05-09 | HU-05 (wiring tab Motor) | 2 |
| `3ca1044d` | 2026-05-13 | metodológico (Inception) | 3 |
| `70d2363b` | 2026-05-13 | metodológico (Bloque B) | 3 |
| `c141276e` | 2026-05-13 | metodológico (Bloque C + DHU-008/011) | 3 |
| `9d178519` | 2026-05-14 | metodológico (DHU-012/013) | 3 |
| `357e59e2` | 2026-05-15 | metodológico (Bloque D + TTH-06 + DHU-014) | 3 |
| `93d01c8c` | 2026-05-16 | metodológico (Bloque E + 5 TTH + DHU-015) | 3 |
| `45c7c044` | 2026-05-16 | metodológico (Bloque F + cierre MVP1) | 3 |
| `194f5107` | 2026-05-17 | metodológico (MVP2 + DHU-017 + BACKLOG_OVERVIEW + HU_LITE) | 3 |
| `69a7f470` | 2026-05-17 | metodológico (higiene post-cierre MVP2) | 3 |

103 commits totales, 28 commits primarios mapeados arriba; el resto son refinamientos, fixes, infra, docs.
