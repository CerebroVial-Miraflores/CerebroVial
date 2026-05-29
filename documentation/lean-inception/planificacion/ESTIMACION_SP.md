# Estimación de Story Points — CerebroVial

> Generado: 2026-05-18 por Claude Code
> Versión: v1
> Inputs:
> - [`MOSCOW_RATIFICADA.md`](MOSCOW_RATIFICADA.md) (Fase 4.2)
> - [`AUDITORIA_HU_CODIGO.md`](AUDITORIA_HU_CODIGO.md) (Fase 4.1)
> - Backlog completo en [`documentation/lean-inception/`](../)
> - Protocolo: [`CEREMONIA_PLANNING_POKER.md`](CEREMONIA_PLANNING_POKER.md)
> Salida prescrita por: [`PROMPT_ARRANQUE_CLAUDE_CODE_v2.md`](PROMPT_ARRANQUE_CLAUDE_CODE_v2.md) Fase 4.3.

## 1. Anclas de la escala (Pasada 0)

Per protocolo §2, identificación de los dos puntos de calibración de la escala Fibonacci.

| Ancla | Elemento elegido | SP | Justificación |
|---|---|---|---|
| **Baja** | HU-09 (Notas e incidencias del turno) | **3** | CRUD limpio con paginación, 6 CAs, sin realtime, sin integraciones con subsistemas complejos. Más limpio como anchor que HU-07 (que sugería el protocolo) porque HU-07 requiere SSE nueva y catálogo de notificación con agrupamiento — más cerca de 5 SP. |
| **Alta** | TTH-10 (Motor adaptativo) | **13** | Subsistema central con 3 estrategias (Webster + MaxPressure + MTC) + AdaptiveEngine selector + persistencia ampliada + integraciones (TTH-09/TTH-07/TTH-04/HU-15) + 14 CTs. Aporte de ingeniería principal de la tesis (per `EVOLUCION_TESIS.md` Fase 3). |

## 2. Tareas Técnicas Habilitadoras (Pasada 1)

| Código | Título corto | MoSCoW | SP total | SP ejec | SP rest | Justificación |
|---|---|---|---|---|---|---|
| TTH-01 | Autenticación JWT + bcrypt | Must | 5 | 1 | 4 | 5 CTs: endpoint `/auth/login`, bcrypt hashing, JWT con claims, dependency `get_current_user`, tests. Tipo infra. SP ejec=1: solo `passlib[bcrypt]` en requirements + tabla `users` por migración. Runtime cero. |
| TTH-02 | Docker Compose multi-servicio | Must | 5 | 5 | 0 | 6 CTs: servicios + quickstart máquina limpia + comunicación interna + volúmenes + README + .env.example. Tipo infra alta complejidad inicial. Auditoría: Completo. |
| TTH-03 | Repositorio + CI con cobertura completa | Must | 5 | 3 | 2 | 7 CTs: ramas + .gitignore + trigger CI + tests 4 módulos + lint + mypy + protección. Auditoría Parcial: ruff + pytest core + npm + docker build OK. Falta mypy + tests edge/shared/ia_prediction. Restante ≈40% complejidad. |
| TTH-04 | Lógica fallback en cascada | Must | 13 | 0 | 13 | 10 CTs: health checks + lógica determinista 5 estados + persistencia transiciones + 2 endpoints + 6 campos componente + resiliencia auditoría + atomicidad + conservador + tests unitarios + integración. Subsistema crítico de robustez. Auditoría: cero matches. |
| TTH-05 | Tiempos preconfigurados degradado 3 | Must | 5 | 0 | 5 | 8 CTs: tabla persistente + endpoints GET/PUT RBAC + UI Admin + defaults seguros + auditoría + tests + integración con TTH-04. CRUD estándar con RBAC. Auditoría: No iniciado. |
| TTH-06 | Capa de DTOs transversal | Won't | 8 | 0 | — | Estimación gruesa documental (per §3.3): refactor transversal a todos los endpoints del backend con DTOs Pydantic explícitos. Trabajos Futuros declarado por DHU-014. No entra a sprint 4. |
| TTH-07 | Integración SUMO | Must | 13 | 0 | 13 | 8 CTs: topología + 4 patrones demanda con seeds + script generación dataset + particiones reproducibles + integración TraCI + KPIs comparativos + documentación + tests. Cuello de botella cronológico declarado en spec. Sin código actual. |
| TTH-08 | Visión computacional | Must | 13.5 | 12 | 1.5 | **Estado al cierre de Fase 9 (2026-05-29): Parcial — F8 diferida.** Sprint ejecutado F0→F7 + F9 (~12 SP) con 11 CTs entregados operativamente: YOLO + tracking + ROI + métricas direccionales + persistencia `vision_aggregates` + endpoint `/vision/state` + input modes + stream procesado + health check + tests (124 passed). **CT-08.11(e) acotado** a repo↔modelo↔Postgres vivo (testcontainers F7), NO migración↔modelo ni pipeline-de-video. **CT-08.9 (dataset etiquetado ≥200 frames + mAP/precisión/recall) diferido** por decisión del usuario — trabajo de datos, ~1.5 SP restantes en F8 fuera del sprint TTH-08. C7.6 (pin CPU torch) reabierta como F9.z; deudas C9.7 (paridad migración↔modelo) y C9.8 (wirear edge_device/tests a CI) nominadas. Ver `tth-08-fase9-handoff.md` (cierre) y addendum F9 en DECISIONS_HU.md DHU-024. |
| TTH-09 | Modelo GRU servido vía API | Must | 13 | 1 | 12 | 9 CTs: 4 GRU univariados + multi-output ventana/horizonte + script entrenamiento reproducible + endpoint `POST /predict` con contrato 4 dirs × horizonte + persistencia predicciones + 4 métricas + 80% accuracy aspiracional + fallback HTTP + tests. SP ejec=1: endpoint canónico existe pero con contrato totalmente distinto (Delta-01); andamiaje preservable mínimo. |
| TTH-10 | Motor adaptativo | Must | 13 | 8 | 5 | **Ancla alta.** 14 CTs: Webster + MaxPressure + MTC + AdaptiveEngine selector + endpoint + persistencia decisiones + integraciones TTH-09/TTH-07/TTH-04/HU-15 + tests. Auditoría Parcial: las 4 estrategias + selector + endpoint + tests cubiertos (≈70% complejidad). Falta integraciones + persistencia ampliada. La propia spec se autoclasifica "Construido, integración pendiente". |
| TTH-11 | Spike hiperparámetros temporales | Should | 5 | 0 | 5 | 8 CTs documentales: documento + 4 secciones por hiperparámetro + ≥5 fuentes bibliográficas + ≥3 combinaciones empíricas + tabla resumen + limitaciones + rigor académico + nota Δt para TTH-07. Spike tipo TTH (acepta error mayor). No existe documento. |

**Subtotal TTH (Must+Should):** 90 SP total, 23 SP ejecutado, 67 SP restante.

> **Actualización 2026-05-29 (TTH-08 F9 cierre).** TTH-08 pasa de `13/5/8` a `13.5/12/1.5`: total +0.5 SP (DHU-024 reescala a 13.5 SP estimados), ejec +7 SP (Fases F0–F7 + F9 entregadas), restante −6.5 SP (queda F8 = 1.5 SP diferida fuera del sprint). Subtotal TTH recalculado: **90.5 SP total, 30 SP ejecutado, 60.5 SP restante.** F8 diferida explícitamente — ver `tth-08-fase9-handoff.md` y addendum F9 al pie de DHU-024.

### Revisión cíclica TTH (Paso §4.2 cada 5 elementos)

Tras estimar TTH-01 a TTH-05: TTH-04 con SP=13 es coherente con TTH-10 (ambos subsistemas críticos densos). TTH-01/02/03/05 con SP=5 son comparables entre sí (infra/CRUD estándar). OK.

Tras estimar TTH-06 a TTH-11: TTH-07/08/09 todos SP=13 — los tres son subsistemas con múltiples CTs heterogéneos (SUMO+TraCI, visión+persistencia+demo, GRU+entrenamiento+servicio+persistencia). Comparables. TTH-11 SP=5 es coherente con que es spike documental, no implementación. OK.

## 3. Historias de Usuario (Pasada 2)

### Bloque A

| Código | Título corto | MoSCoW | SP total | SP ejec | SP rest | Justificación |
|---|---|---|---|---|---|---|
| HU-01 | Acceso diferenciado por rol | Must | 5 | 0 | 5 | 6 CAs: 3 vistas segregadas por rol + API 403 + login redirect + token expirado. Vista pasiva pero requiere consumo JWT + enforcement RBAC en frontend (routing condicional) y backend (decorators FastAPI). Habilita por TTH-01. Auditoría: cero enforcement. |

### Bloque B — Operador tiempo real

| Código | Título corto | MoSCoW | SP total | SP ejec | SP rest | Justificación |
|---|---|---|---|---|---|---|
| HU-02 | Monitoreo estado actual | Must | 8 | 1 | 7 | 5 CAs: flujo+cola por acceso + auto-update ≤5s (SSE) + umbrales color + marca "desactualizado" (DHU-005 A) + login. Requiere vista intra-intersección con N accesos + canal realtime. SP ejec=1 por `/api/intersections` existente. La DashboardView actual es multi-intersección (Delta-06), no aprovechable. |
| HU-03 | Predicción de congestión | Must | 5 | 1 | 4 | 5 CAs: nivel 0-5 per acceso horizonte + auto-update ≤5s + resaltado >umbral + marca no confirmada + login. Vista pasiva consumiendo TTH-09 + realtime. SP ejec=1 por endpoint canónico existe (con contrato distinto, requiere refactor). |
| HU-04 | Vista combinada estado + predicción | Must | 5 | 0 | 5 | 5 CAs: integración HU-02+HU-03 + resaltado discrepancia + auto-update independiente + robustez ambas fuentes + login. Composición visual. No existe vista combinada. |
| HU-05 | Estrategia de control activa | Must | 5 | 2 | 3 | 5 CAs: nombre + tiempos verde por acceso + timestamp activación + auto-update + robustez + login. Vista pasiva. SP ejec=2: ControlView muestra strategy output + manejo errores webster_infeasible/invalid_state, pero como playground request-response (Delta-08), no como vista pasiva del estado vigente. ≈40% complejidad. |
| HU-06 | Explicación razón de selección | Must | 5 | 1 | 4 | 5 CAs: texto plantillado con valores + actualización al cambio + fallback genérico + robustez + login. Requiere catálogo plantillas en lenguaje dominio (5-10 textos). SP ejec=1: `reasoning` del motor se renderiza como "Log técnico" (Delta-09), no lenguaje dominio. |
| HU-07 | Notificación cambios estrategia | Must | 5 | 0 | 5 | 6 CAs: notificación 4 campos + auto-descarte 10s + agrupamiento + posición no-central + indicador canal degradado + login. Requiere SSE + componente toast + lógica agrupamiento. Cero infraestructura toast en frontend. |
| HU-08 | Historial decisiones motor | Must | 8 | 0 | 8 | 6 CAs: persistencia F31 inglobada (tabla motor_decisions con timestamp/estrategia/razón/parámetros/anterior) + paginación + filtros + default 24h + resiliencia + login. Tabla nueva + lógica persistencia en endpoint + vista paginada. Cero existe. |

### Bloque C — Operador degradado

| Código | Título corto | MoSCoW | SP total | SP ejec | SP rest | Justificación |
|---|---|---|---|---|---|---|
| HU-10 | Alerta transversal estado operativo | Must | 13 | 0 | 13 | 10 CAs: 5 estados visualmente distintos + banner transversal en App.tsx + persistencia transiciones inmutable inglobada (timestamp/anterior/nuevo/causa) + reconocimiento con identidad + re-escalada automática + retorno normal + resiliencia log + DHU-005 B + login. HU densa de alta criticidad. AlertsView actual es vista propia (Delta-11), no banner. |
| HU-11 | Vista estado componentes (Operador) | Must | 5 | 0 | 5 | 9 CAs: lista cualitativa OK/Degradado/Fuera de servicio + actualización ≤5s + texto impacto operativo + accesibilidad permanente + integración con banner HU-10 + navegación + robustez + login + resaltado no-OK. Vista pasiva consumiendo CT-04.5 simple. |
| HU-12 | Explicación modo degradado | Must | 5 | 0 | 5 | 6 CAs: texto compuesto 3 elementos (disparador/fallback/capacidad perdida) + actualización al cambio + retorno normal + fallback genérico + robustez + login. Catálogo plantillas (4-6) + componente UI. |

### Bloque D — Administrador

| Código | Título corto | MoSCoW | SP total | SP ejec | SP rest | Justificación |
|---|---|---|---|---|---|---|
| HU-13 | Salud técnica componentes (Admin) | Must | 5 | 1 | 4 | 7 CAs: 7 campos por componente (nombre/id/estado/latencia/fallos/2 timestamps) + auto-update + resaltado no-OK + robustez monitor + manejo "sin datos" + RBAC 403 + login. Vista técnica. SP ejec=1: mockup visual en AdminView segunda card (Delta-12) parcialmente reusable como scaffolding. |
| HU-14 | Métricas modelo predictivo | Must | 13 | 0 | 13 | 14 CAs: sustrato inglobado persistencia predicciones (CA-14.1) + asociación con observación al vencer horizonte + cálculo 4 métricas (MAE/RMSE/accuracy/matriz 6×6) + ventana temporal + vista 4 paneles + tooltips ayuda + matriz convención filas=real + toggle absolutos/% + diagonal visual + comunicación datos insuf + DHU-005 B + RBAC + login. Densidad comparable a HU-08+HU-15. AnalyticsView actual es mock irrelevante. |
| HU-15 | Configuración parámetros operativos | Must | 13 | 0 | 13 | 13 CAs: sustrato inglobado persistencia + modificación con auditoría + efecto ≤30s sin redeploy + defaults seguros + presentación 3 familias + validación tiempo real + confirmación con preservación + historial cambios + restaurar defaults + indisponibilidad + concurrencia last-write-wins con advertencia + RBAC + login. HU compleja con concurrencia. Cero existe. |

### Bloque F — Gerente

| Código | Título corto | MoSCoW | SP total | SP ejec | SP rest | Justificación |
|---|---|---|---|---|---|---|
| HU-16 | KPIs Gerente periodo seleccionable | Must | 13 | 1 | 12 | 21 CAs: sustrato F30 inglobada (granularidad 30s flujo/cola/velocidad/densidad) + retención sin política + 5 modos selector + validaciones fechas duales + convenciones presets ISO + definiciones operacionales 4 KPIs + 4 cards con gráficos adaptativos (hora/día/semana) + toggle dirección + tooltips + casos degenerados + DHU-005 B + RBAC + login. HU MASIVA con persistencia inglobada. SP ejec=1: AnalyticsView aporta scaffolding mínimo. |
| HU-17 | Comparativa entre periodos | Must | 8 | 0 | 8 | 16 CAs: selector compartido con HU-16 + definición periodo previo equivalente + 4 paneles con 2 series superpuestas + semántica mejora/empeoramiento triple (signo/flecha/color) + datos insuficientes por panel + DHU-005 B + RBAC + login. Reutiliza sustrato HU-16. No existe vista. |

### MVP2

| Código | Título corto | MoSCoW | SP total | SP ejec | SP rest | Justificación |
|---|---|---|---|---|---|---|
| HU-09 | Notas e incidencias turno | Should | 3 | 0 | 3 | **Ancla baja.** 6 CAs: creación con persistencia + listado paginado + filtros por fechas/autor + edición 24h + resiliencia + login. CRUD estándar autocontenido. Sin dependencias con otras HUs. |
| HU-18 | Drill-down periodo (Gerente) | Could | 13 | 0 | 13 | 21 CAs: acceso desde HU-16/HU-17 + selector con distinción local/global + 3 carriles temporales integrados (tráfico con zoom a 30s + eventos motor con marcadores activables + estado operativo con bandas coloreadas) + casos degenerados independientes por carril + RBAC + login. Vista muy rica consumiendo 3 registros distintos. |
| HU-19 | Exportación PDF/Excel | Could | 13 | 0 | 13 | 28 CAs: invocación desde HU-16/HU-17 + estructura PDF de HU-16 con disgregación obligatoria + estructura PDF de HU-17 con semántica visual + estructura Excel 2 hojas + política conservadora ante fuente caída + entrega archivo con nombre autosuficiente + RBAC + login. Librerías PDF/Excel + plantillas + composición datos. ReportModal Gemini (Delta-13) no reutilizable. |
| HU-20 | Comparativa modelos predictor | Could | 13 | 0 | 13 | 20 CAs: ejecución paralela del modelo de respaldo + persistencia paralela con identificador modelo + asociación misma observación + cálculo sobre mismos eventos + 4 paneles comparativos con valores lado a lado + semántica métrica + tolerancia empate + 2 matrices 6×6 lado a lado con toggle + tooltips + casos degenerados + RBAC. Sustrato paralelo + presentación rica. |
| HU-21 | Escalamiento incidentes Op→Admin | Could | 13 | 0 | 13 | 37 CAs (HU más densa del backlog): botones en HU-10/HU-12 + captura automática contexto desde CT-04.4/04.5 + modal de confirmación con texto libre + sustrato registro append-only nuevo + vista Admin con paginación/filtros + transición Atendido irreversible + badge pendientes en navegación + recuperación automática preserva incidente + vista Operador con propios + casos degenerados + RBAC dual (botón solo Operador, vista Admin solo Admin). |

**Subtotal HUs (Must+Should+Could):** Must=121 + Should=3 + Could=65 = 189 SP total. SP ejec=7. SP restante=182.

### Revisión cíclica HUs (Paso §4.2 cada 5 elementos)

Tras HU-01 a HU-05: HU-02 con SP=8 vs HU-03 con SP=5 — ambas tiempo real pero HU-02 requiere construir realtime infra desde cero (más caro), HU-03 reusa esa infra. Coherente.

Tras HU-06 a HU-10: HU-10 con SP=13 es coherente con TTH-04 (también 13) — ambas subsistemas de robustez densos. HU-07/HU-08 SP=5/8 — HU-08 tiene persistencia inglobada nueva, justifica mayor.

Tras HU-11 a HU-15: HU-14 y HU-15 ambas SP=13 — comparables (sustratos inglobados + presentaciones complejas con concurrencia/cálculo). HU-13 SP=5 es coherente con HU-11 SP=5 (ambas vistas técnicas pasivas con la diferencia ortogonal de 4 campos extra técnicos).

Tras HU-16/17: HU-16 SP=13 es comparable a HU-14/HU-15 (HU más densa del Bloque F con sustrato F30 inglobado). HU-17 SP=8 reutiliza HU-16, no duplica.

Tras HU-18 a HU-21 (MVP2): los 4 Could son SP=13 — todos densos en CAs (21/28/20/37). HU-09 SP=3 es el ancla baja, contraste claro con el resto del MVP2. Coherente.

## 4. RNF transversales estimables por separado (Pasada 3)

Per protocolo §5.3, solo los RNF cuyo esfuerzo NO está contenido en una HU específica.

| Código RNF | Descripción corta | MoSCoW | HUs que toca | SP total | SP ejec | SP rest | Justificación |
|---|---|---|---|---|---|---|---|
| RNF-INT-02 | Accesibilidad WCAG 2.1 AA | Should | 12+ HUs (todas las vistas) | 5 | 0 | 5 | Transversal: color+ícono+texto en estados, controles activables con teclado, elementos dinámicos para lectores de pantalla. Verificación axe-core. Frontend actual no cumple sistemáticamente. |
| RNF-INT-04 | Coherencia visual y textual entre vistas relacionadas | Must | HU-16↔17, HU-18↔10, HU-20↔14, HU-06↔11↔12 | 3 | 0 | 3 | Style guide + revisión cruzada catálogos plantillas + orden indicadores consistente + códigos visuales unificados. Trabajo de auditoría visual post-implementación. |
| RNF-MNT-01 | Extensibilidad catálogos plantillas como datos | Should | HU-06, HU-11, HU-12 | 3 | 0 | 3 | Modelar catálogos como YAML/JSON cargado al arranque, no strings hardcoded. Permite agregar plantilla nueva sin redeploy. Estructura de soporte estándar. |

**Subtotal RNF transversales:** 11 SP total, 0 SP ejecutado, 11 SP restante.

Otros RNF (PERF, SEC, REL, SAF, FLX, COM, FUN, otros INT/MNT) están contenidos en HUs específicas y no se estiman por separado (evitar doble conteo per §5.3).

## 5. Resumen agregado

### 5.1 SP total por categoría MoSCoW (excluye Won't)

| Categoría | TTH | HUs | RNF transv | Total |
|---|---|---|---|---|
| Must | 85 | 121 | 3 | 209 |
| Should | 5 | 3 | 8 | 16 |
| Could | 0 | 52 | 0 | 52 |
| **Total** | **90** | **176** | **11** | **277** |

### 5.2 SP ejecutado (consumido en sprints 1-3)

| Categoría | TTH | HUs | RNF transv | Total |
|---|---|---|---|---|
| Must | 23 | 7 | 0 | 30 |
| Should | 0 | 0 | 0 | 0 |
| Could | 0 | 0 | 0 | 0 |
| **Total** | **23** | **7** | **0** | **30** |

Velocity histórica estimada: 30 SP en 3 sprints = **10 SP/sprint promedio** para un estudiante individual. Realista.

### 5.3 SP restante para sprint 4

| Categoría | TTH | HUs | RNF transv | Total |
|---|---|---|---|---|
| Must | 62 | 114 | 3 | **179** |
| Should | 5 | 3 | 8 | 16 |
| Could | 0 | 52 | 0 | 52 |
| **Total** | **67** | **169** | **11** | **247** |

**SP restante solo Must (alcance mínimo del sprint 4): 179 SP.**

## 6. Alerta crítica: loop MoSCoW ⇄ Planning Poker requerido

Per protocolo §6.3, si la suma de SP restante Must excede la capacidad del sprint 4, dispara loop.

**Cálculo de capacidad sprint 4:**
- Velocity histórica: 10 SP/sprint.
- Sprint 4 puede asumirse de duración estándar o ampliada (decisión humana).
- Aun con sprint 4 = 2× duración estándar (~20 SP), **179 SP Must restante excede 9× la capacidad realista**.

**Conclusión:** el sprint 4 NO puede absorber los 179 SP Must restante en cualquier escenario realista. El loop MoSCoW⇄Planning del Paso 7 de [`PROTOCOLO_DISTRIBUCION_SPRINTS.md`](PROTOCOLO_DISTRIBUCION_SPRINTS.md) se disparará en Fase 4.4. Esta auditoría/estimación confirma la brecha estructural ya señalada por la auditoría (§1 de [`AUDITORIA_HU_CODIGO.md`](AUDITORIA_HU_CODIGO.md)): cobertura efectiva ≈25%.

**Candidatos previsibles a descenso de Must (referencia §8.2 de MOSCOW_RATIFICADA.md):**

- **TTH-07 (SUMO, 13 SP)** — descender a Should si la defensa acepta validación cualitativa.
- **HU-14 (13 SP) + HU-15 (13 SP)** — descender a Should si HU-13 (5 SP) cubre la visibilidad mínima Admin.
- **HU-16 (13 SP) + HU-17 (8 SP)** — descender Bloque F completo si la defensa académica no exige reportería Gerente para MVP1.
- **TTH-11 (5 SP)** — ya está Should; podría bajar a Could si se acepta documento parcial.

Movimientos hipotéticos para alcanzar capacidad realista del sprint 4 (~20-30 SP):

| Escenario | Acción | SP Must restante resultante |
|---|---|---|
| Optimista (mantener todo Must) | — | 179 (inviable) |
| Bajar TTH-07 a Should | -13 | 166 (sigue inviable) |
| Bajar HU-14/15/16/17 a Should | -46 | 120 (sigue inviable) |
| Bajar también HU-08/10 a Should | -21 | 99 (sigue inviable) |
| **Reducir alcance Must a "demo demostrable" mínimo** (HU-01, HU-02, HU-05, HU-11 + TTH-01/02/03/10) | de 209 a ~50 | ~25 SP restante (viable) |

Esta decisión de reducción es **metodológica con estudiante** en Fase 4.4 o Fase 4.5 (no aquí). La Planning Poker reporta honestamente las estimaciones; la negociación de alcance es responsabilidad de la distribución de sprints.

## 7. Elementos marcados `?` (Pasada 5)

Ninguno. Las 32 HUs/TTH se estimaron con confianza razonable a partir de la auditoría y de los CAs/CTs documentados.

## 8. Notas de coherencia detectadas en revisión global (Pasada 4)

### 8.1 Patrón ✓ verificado: SP iguales son comparables

- TTH-04, TTH-07, TTH-08, TTH-09, TTH-10, HU-10, HU-14, HU-15, HU-16, HU-18, HU-19, HU-20, HU-21 todos SP=13. Lista coherente: cada uno es subsistema central o HU de alta densidad (≥10 CAs).
- TTH-01, TTH-02, TTH-03, TTH-05, TTH-11, HU-01, HU-03, HU-04, HU-05, HU-06, HU-07, HU-11, HU-12, HU-13 todos SP=5. Lista coherente: cada uno es vista pasiva o TTH de infra/integración estándar.
- HU-02, HU-08, HU-17 SP=8. Coherente: cada uno tiene un componente de complejidad agregada (realtime para HU-02, persistencia inglobada para HU-08, reutilización con composición de 2 series para HU-17) sin ser subsistema central.

### 8.2 Patrón ✓ verificado: SP restante ≤ SP total siempre

Ninguna inconsistencia. SP ejecutado siempre ≤ SP total.

### 8.3 Patrón ✓ verificado: HUs/TTH con código tienen SP ejecutado > 0

- TTH-02 (Completo): SP ejec = SP total = 5 ✓
- TTH-03/08/10 (Parcial): SP ejec = 3/5/8 (>0) ✓
- HU-05/06/13 (Parcial): SP ejec = 2/1/1 (>0) ✓
- TTH-01/09, HU-02/03/16 (No iniciado con andamiaje mínimo): SP ejec = 1 ✓

### 8.4 Estimación retrospectiva por sprint (proyección informal para Fase 4.4)

Los 30 SP ejecutados se distribuyen tentativamente así para análisis posterior:

| Sprint hipotético | SP ejec estimado | Elementos probables |
|---|---|---|
| Sprint 1 (~10 SP) | 10 | TTH-02 (5) + parte TTH-08 (3) + TTH-01 (1) + TTH-09 (1) |
| Sprint 2 (~10 SP) | 10 | resto TTH-08 (2) + parte TTH-10 (4) + TTH-03 (3) + HU-03 (1) |
| Sprint 3 (~10 SP) | 10 | resto TTH-10 (4) + HU-05 (2) + HU-06 (1) + HU-13 (1) + HU-02 (1) + HU-16 (1) |

Esta proyección es informal — la asignación rigurosa por sprint requiere `git log` y se resuelve en Fase 4.4 (forense de fronteras).
