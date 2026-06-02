# Feature Specification: CerebroVial — Control adaptativo de semáforos (MVP1 + MVP2)

**Feature Branch**: `feature/SDD`

**Created**: 2026-05-20

**Status**: Draft (mapeo brownfield del backlog curado)

**Input**: Product Backlog en `documentation/lean-inception/` (21 HU, 11 TTH, 22 RF, 53 RNF).

> **Nota (2026-06-02, DHU-028):** el Product Backlog se amplió posteriormente a **22 HU + 12 TTH** (HU-22: mapa de congestión de la red; TTH-12: infraestructura de datos de congestión por arista). Esta especificación y su inventario de tareas (`tasks.md`) siguen reflejando el backlog de 21 HU + 11 TTH del que se derivaron y cuyos elementos están estimados en SP; la propagación de HU-22/TTH-12 a los artefactos Spec Kit, con su estimación, queda pendiente de una re-derivación.

> **Adopción brownfield (DHU-021).** Esta especificación **mapea** el Product Backlog ya cerrado
> (2026-05-16) al formato de Spec Kit; no se genera con `/speckit-specify`. Se conservan los IDs
> nativos (`HU-xx`, `RF-xxx`, `RNF-XXX-NN`) para preservar la trazabilidad. **Los criterios de
> aceptación Given-When-Then NO se reproducen aquí**: viven completos en los archivos
> `HU_BLOQUE_*.md` / `HU_MVP2.md`, enlazados por HU. La redacción de cada HU sigue los principios
> metodológicos del **Título II de `.specify/memory/constitution.md`**.

## Personas y objetivos del producto

**Las 4 Personas** (fuente: `BACKLOG_OVERVIEW.md`):
- **Operador de Tráfico Municipal** — vigila el control del semáforo en tiempo real por turnos.
- **Gerente de Tránsito Municipal** — consulta desempeño histórico para reportar y sustentar.
- **Administrador del Sistema** — responsable técnico: salud del sistema, modelo y parámetros.
- **Usuario del sistema** — sujeto compuesto (cualquiera de los tres); se usa solo en HU-01.

**Los 4 Objetivos del producto:**
1. Reducir tiempos de espera en la intersección de Miraflores frente al control fijo actual.
2. Sustentar la decisión técnica del control adaptativo con métricas comparativas auditables.
3. Garantizar continuidad operativa ante fallos de componentes.
4. Generar evidencia gerencial que permita extender el producto a otras intersecciones.

Una sola feature, `001-cerebrovial-mvp`, engloba las 21 HU. **Mapeo de prioridad:** Must → P1,
Should → P2, Could → P3 (clasificación MoSCoW ratificada en `MOSCOW_RATIFICADA.md`).

## User Scenarios & Testing *(mandatory)*

Cada HU se lista con: Persona, resumen ejecutivo (1 línea, de `BACKLOG_OVERVIEW.md`), RF que la
cubre, RNF asociados clave, prioridad MoSCoW y enlace al archivo fuente de sus CAs. Los Gherkin
completos están en el archivo enlazado.

### Sección 1 — Acceso al sistema (Bloque A) · P1

Fuente de CAs: [HU_BLOQUE_A.md](../../documentation/lean-inception/2-backlog/HU_BLOQUE_A.md)

- **HU-01 — Acceso diferenciado por rol** · *Usuario (los 3 roles)* · Must/P1.
  El sistema reconoce al usuario autenticado y le muestra las vistas correspondientes a su rol;
  Operador, Gerente y Administrador tienen interfaces distintas.
  RF: RF-001, RF-002. · RNF: RNF-SEC-02 (bcrypt+JWT), RNF-SEC-03 (RBAC backend), RNF-SEC-04,
  RNF-INT-07 (ocultación de rutas). · Habilitada por TTH-01 (autenticación).

### Sección 2 — Operador: monitoreo en tiempo real (Bloque B) · P1

Fuente de CAs: [HU_BLOQUE_B.md](../../documentation/lean-inception/2-backlog/HU_BLOQUE_B.md)

- **HU-02 — Monitoreo del estado actual de la intersección** · *Operador* · Must/P1.
  Ve flujo vehicular y longitud de cola por acceso, actualizado automáticamente.
  RF: RF-003. · RNF: RNF-PERF-01 (≤5 s), RNF-REL-01 Caso A ("desactualizado"), RNF-INT-01.
- **HU-03 — Predicción de congestión a corto plazo** · *Operador* · Must/P1.
  Ve el nivel de congestión predicho (escala 0-5) para los próximos minutos.
  RF: RF-004. · RNF: RNF-PERF-01, RNF-REL-01 Caso B ("no confirmado"), RNF-COM-02.
- **HU-04 — Vista combinada del estado actual y la predicción** · *Operador* · Must/P1.
  Ve estado presente y predicción en una sola vista integrada.
  RF: RF-005. · RNF: RNF-REL-01 (Casos A+B independientes), RNF-INT-01.
- **HU-05 — Visualización de la estrategia de control activa** · *Operador* · Must/P1.
  Ve qué estrategia del motor está vigente, incluyendo la indicación de cuándo la decisión
  proviene del respaldo normativo. *Vista pasiva* (DHU-020, cierre de Delta-08). RF: RF-006.
  · RNF: RNF-REL-01 Caso B, RNF-INT-05.
- **HU-06 — Explicación de la razón de selección de estrategia** · *Operador* · Must/P1.
  Entiende, en lenguaje legible, por qué el motor eligió la estrategia activa.
  RF: RF-007. · RNF: RNF-FUN-04 (catálogo de plantillas), RNF-MNT-01, RNF-INT-05.
- **HU-07 — Notificación de cambios de estrategia del motor** · *Operador* · Must/P1.
  Es notificado pasivamente cuando el motor cambia de estrategia.
  RF: RF-008. · RNF: RNF-PERF-01, RNF-MNT-02 (tiempos parametrizables), RNF-REL-01.
- **HU-08 — Consulta del historial de decisiones del motor** · *Operador* · Must/P1.
  Consulta cronológicamente las decisiones del motor, con razón y parámetros (auditable).
  RF: RF-009. · RNF: RNF-REL-03/04 (durabilidad append-only), RNF-SEC-01 (inmutabilidad).
- **HU-09 — Registro de notas e incidencias del turno** · *Operador* · **Should/P2 (MVP2)**.
  Registra notas de turno y las consulta; todos los Operadores ven todas las notas.
  RF: RF-021. · RNF: RNF-SEC-01 (excepción: editable en ventana corta), RNF-REL-05, RNF-PERF-02.
  *(Vive físicamente en `HU_BLOQUE_B.md`, conforme a DHU-017; ver también Sección 6.)*

### Sección 3 — Operador: operación degradada (Bloque C) · P1

Fuente de CAs: [HU_BLOQUE_C.md](../../documentation/lean-inception/2-backlog/HU_BLOQUE_C.md)

- **HU-10 — Alerta activa transversal del estado operativo** · *Operador* · Must/P1.
  Alerta visible en todas las vistas cuando el sistema entra en modo degradado o falla total.
  RF: RF-019. · RNF: RNF-REL-02 (disponibilidad transversal), RNF-REL-08 (atomicidad), RNF-INT-04.
  *(Distinción componente caído / degradado / fallback: Art. 16; marca pasiva vs activa: Art. 17.)*
- **HU-11 — Vista del estado operativo de los componentes** · *Operador* · Must/P1.
  Vista simplificada de qué componentes funcionan y cuáles no (sin detalles técnicos).
  RF: RF-018. · RNF: RNF-FUN-04, RNF-MNT-01, RNF-REL-09.
- **HU-12 — Explicación del modo degradado activo** · *Operador* · Must/P1.
  Entiende qué capacidad se perdió, qué respaldo se usa y qué debe hacer.
  RF: RF-020. · RNF: RNF-FUN-04, RNF-INT-05, RNF-SAF-01 (fail-safe a degradado nivel 3).

### Sección 4 — Administrador: soporte técnico (Bloque D) · P1

Fuente de CAs: [HU_BLOQUE_D.md](../../documentation/lean-inception/2-backlog/HU_BLOQUE_D.md)

- **HU-13 — Vista técnica de salud de los componentes** · *Administrador* · Must/P1.
  Estado técnico detallado por componente: latencia, errores recientes, timestamps.
  RF: RF-010. · RNF: RNF-PERF-01, RNF-SEC-05 (segregación de presentación), RNF-REL-09.
- **HU-14 — Métricas de desempeño del modelo predictivo** · *Administrador* · Must/P1.
  Evalúa el modelo principal sobre datos recientes (MAE, RMSE, accuracy, matriz de confusión 6×6).
  RF: RF-012. · RNF: RNF-FUN-02 (calidad del modelo), RNF-PERF-09, RNF-INT-03 (tooltips).
- **HU-15 — Configuración de parámetros operativos** · *Administrador* · Must/P1.
  Ajusta umbrales, ventanas temporales y frecuencias; cambios auditables.
  RF: RF-011. · RNF: RNF-PERF-08 (efecto sin redeploy), RNF-MNT-02, RNF-REL-07 (concurrencia), RNF-SAF-03.

### Sección 5 — Gerente: reportería (Bloque F) · P1

Fuente de CAs: [HU_BLOQUE_F.md](../../documentation/lean-inception/2-backlog/HU_BLOQUE_F.md)

- **HU-16 — Consulta de KPIs operativos sobre periodo seleccionable** · *Gerente* · Must/P1.
  Ve los 4 KPIs principales (tiempo de espera, cola, throughput, demora) sobre un periodo elegible.
  RF: RF-014. · RNF: RNF-PERF-04/05, RNF-FUN-01 (datos faltantes), RNF-INT-03/04.
- **HU-17 — Vista comparativa entre periodos** · *Gerente* · Must/P1.
  Compara el periodo actual con el previo equivalente, con indicadores de mejora/empeoramiento.
  RF: RF-015. · RNF: RNF-PERF-12 (paralelización), RNF-FUN-01, RNF-INT-04.

### Sección 6 — MVP2: extensiones condicionales a holgura · P2/P3

Fuente de CAs: HU-09 en [HU_BLOQUE_B.md](../../documentation/lean-inception/2-backlog/HU_BLOQUE_B.md);
HU-18…21 en [HU_MVP2.md](../../documentation/lean-inception/2-backlog/HU_MVP2.md).

- **HU-09 — Registro de notas e incidencias del turno** · *Operador* · Should/P2. (Detallada en Sección 2.)
- **HU-18 — Vista detallada de periodo específico (drill-down)** · *Gerente* · **Could/P3**.
  Investiga un periodo con tres carriles temporales: tráfico, decisiones del motor, estado operativo.
  RF: RF-016. · RNF: RNF-PERF-06, RNF-PERF-12, RNF-INT-04.
- **HU-19 — Exportación de reportes a PDF o Excel** · *Gerente* · **Could/P3**.
  Descarga reportes de HU-16/HU-17 en PDF presentable o Excel con datos crudos.
  RF: RF-017. · RNF: RNF-PERF-07, RNF-FUN-05, RNF-REL-06, RNF-INT-06, RNF-SEC-07.
- **HU-20 — Comparativa modelo principal vs respaldo** · *Administrador* · **Could/P3**.
  Compara métricas de los dos modelos sobre los mismos eventos operacionales.
  RF: RF-013. · RNF: RNF-FUN-03 (comparabilidad rigurosa), RNF-MNT-03, RNF-PERF-13, RNF-FLX-02.
- **HU-21 — Escalamiento de incidentes del Operador al Administrador** · *Operador → Administrador* · **Could/P3**.
  Escala incidentes con captura automática del contexto, trazabilidad y badge de pendientes.
  RF: RF-022. · RNF: RNF-FUN-06 (independencia de dimensiones), RNF-PERF-10, RNF-SEC-01 (excepción).

> **Nota de alcance.** El Bloque E (TTH-07…TTH-11: SUMO, visión, GRU, motor adaptativo, spike de
> hiperparámetros) NO contiene HU operativas; su sustrato técnico se refleja en `plan.md` y en las
> tareas de `tasks.md`. Las features F02 (dashboard) y F31 (persistencia de decisiones) están
> inglobadas como criterios de aceptación, no son HU propias.

### Edge Cases

El comportamiento ante interrupción de fuente está normado por el **Artículo 13** de la constitución
(DHU-005) y por **RNF-REL-01**: Caso A (fuente de medición → "desactualizado") y Caso B (componente
interno → "no confirmado"), con la excepción conservadora de HU-19/HU-21 (rechazo en vez de marca
pasiva). Los CAs específicos de cada caso límite viven en el archivo fuente de cada HU. Otros casos
límite normados: datos faltantes en cálculos (RNF-FUN-01), concurrencia en configuración (RNF-REL-07),
fallo del propio detector de salud (RNF-REL-09), atomicidad de transiciones de estado (RNF-REL-08).

## Requirements *(mandatory)*

### Functional Requirements

Los **22 RF** (7 familias) están catalogados en
[RF_RNF_LITE.md](../../documentation/lean-inception/3-requisitos/RF_RNF_LITE.md) (lectura) y en
`REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md` (normativo, con trazabilidad CA→RF). Se referencian por
ID, no se reproducen:

- **F1 — Control de acceso:** RF-001, RF-002.
- **F2 — Monitoreo operativo en tiempo real:** RF-003, RF-004, RF-005.
- **F3 — Decisiones del motor adaptativo:** RF-006, RF-007, RF-008, RF-009.
- **F4 — Predicción de tráfico:** cubierta por RF-004/005/013 (sin RF dedicados; sustrato en TTH).
- **F5 — Soporte técnico y configuración:** RF-010, RF-011, RF-012, RF-013 *(RF-013 Could)*.
- **F6 — Reportería ejecutiva:** RF-014, RF-015, RF-016 *(Could)*, RF-017 *(Could)*.
- **F7 — Soporte al Operador y trazabilidad:** RF-018, RF-019, RF-020, RF-021 *(Should)*, RF-022 *(Could)*.

### Key Entities

El modelo de datos (entidades heredadas + `motor_decisions` y `engine_active_state` de diseño) está
en [data-model.md](data-model.md) y en `SDD_CEREBROVIAL.md` §4. No se reproduce aquí.

## Success Criteria *(mandatory)*

### Measurable Outcomes

Criterios medibles, derivados de los 4 objetivos y de RNF cuantificados (referenciados por ID; los
umbrales viven en `RF_RNF_LITE.md`):

- **SC-001 (Obj. 1):** el motor adaptativo reduce el tiempo de espera frente al control fijo actual,
  demostrado por comparación cuantitativa en SUMO (con sistema vs Webster fijo). Medición vía HU-16/17.
- **SC-002 (Obj. 2):** las decisiones del motor son auditables e inmutables (RF-009, RNF-SEC-01,
  RNF-REL-04); las métricas del modelo se reportan honestamente (RNF-FUN-02, Art. 5).
- **SC-003 (Obj. 3):** ante caída de cualquier componente, el sistema permanece operativo en modo
  fail-safe (degradado nivel 3, RNF-SAF-01) y lo comunica de forma transversal (RNF-REL-02) en ≤5 s
  (RNF-PERF-01).
- **SC-004 (Obj. 4):** el Gerente obtiene KPIs y comparativas de periodo (HU-16/17) que sustentan la
  extensión del producto, con vistas que abren en ≤3 s (RNF-PERF-04).
- **SC-005 (transversal):** cumplimiento normativo MTC en cada decisión aplicada al semáforo
  (RNF-SAF-02) y despliegue reproducible con un comando (RNF-FLX-01).

> **Alcance de validación dentro del Sprint 4.** El Sprint 4 vigente (ver `tasks.md` §"Sprint 4
> vigente") cierra 5 elementos (TTH-01, HU-01, TTH-10, HU-05, TTH-03) que sustentan parcialmente
> SC-002 (motor + auditabilidad) y SC-005 (capa MTC, deploy reproducible vía TTH-02 ya completo).
> **SC-001 y SC-003 se validan post-Sprint-4**: SC-001 requiere TTH-07 (Integración SUMO) y
> TTH-09 (GRU servido), ambos en Trabajos Futuros (`REPORTE_PLANIFICACION_SPRINT_4.md` §7); SC-003
> requiere TTH-04 (Fallback en cascada), TTH-05 (Tiempos degradado nivel 3) y HU-10 (Alerta
> transversal), también en Trabajos Futuros. **SC-004** depende de HU-16/17 (Gerente), también
> postergadas. Esta brecha está documentada y argumentada como R4 del reporte de Sprint 4.

## Assumptions

- La validación cuantitativa se hace en simulación SUMO con particiones independientes; no hay datos
  reales de Lima en el ciclo académico (D-008, Art. 7).
- El modelo predictivo es GRU univariado por intersección; el RandomForest es respaldo conmutable
  (D-002/D-006, Art. 2). La escalabilidad multi-intersección es trabajo futuro (RNF-FLX-03 = Won't).
- El módulo de visión se valida de forma independiente y no participa del loop cuantitativo (D-007,
  Art. 6); SUMO provee las métricas de estado para validación.
- MVP1 = 16 HU (HU-01…08, HU-10…17) + 11 TTH. MVP2 (HU-09, HU-18…21) entra solo con holgura de
  cronograma.
- El alcance ejecutable comprometido lo fija `REPORTE_PLANIFICACION_SPRINT_4.md`; ver `tasks.md`.

## Trazabilidad HU ↔ RF ↔ MoSCoW (resumen)

> Matriz fina (CA→RF→RNF) en `REQUISITOS_FUNCIONALES_Y_NO_FUNCIONALES.md`. Las DHU que rigen la
> redacción se enlazan en el Título II de `constitution.md`; no se copian aquí.

| HU | RF | MoSCoW/Prioridad | MVP | Archivo de CAs |
|---|---|---|---|---|
| HU-01 | RF-001, RF-002 | Must / P1 | MVP1 | HU_BLOQUE_A.md |
| HU-02 | RF-003 | Must / P1 | MVP1 | HU_BLOQUE_B.md |
| HU-03 | RF-004 | Must / P1 | MVP1 | HU_BLOQUE_B.md |
| HU-04 | RF-005 | Must / P1 | MVP1 | HU_BLOQUE_B.md |
| HU-05 | RF-006 | Must / P1 | MVP1 | HU_BLOQUE_B.md |
| HU-06 | RF-007 | Must / P1 | MVP1 | HU_BLOQUE_B.md |
| HU-07 | RF-008 | Must / P1 | MVP1 | HU_BLOQUE_B.md |
| HU-08 | RF-009 | Must / P1 | MVP1 | HU_BLOQUE_B.md |
| HU-09 | RF-021 | Should / P2 | MVP2 | HU_BLOQUE_B.md |
| HU-10 | RF-019 | Must / P1 | MVP1 | HU_BLOQUE_C.md |
| HU-11 | RF-018 | Must / P1 | MVP1 | HU_BLOQUE_C.md |
| HU-12 | RF-020 | Must / P1 | MVP1 | HU_BLOQUE_C.md |
| HU-13 | RF-010 | Must / P1 | MVP1 | HU_BLOQUE_D.md |
| HU-14 | RF-012 | Must / P1 | MVP1 | HU_BLOQUE_D.md |
| HU-15 | RF-011 | Must / P1 | MVP1 | HU_BLOQUE_D.md |
| HU-16 | RF-014 | Must / P1 | MVP1 | HU_BLOQUE_F.md |
| HU-17 | RF-015 | Must / P1 | MVP1 | HU_BLOQUE_F.md |
| HU-18 | RF-016 | Could / P3 | MVP2 | HU_MVP2.md |
| HU-19 | RF-017 | Could / P3 | MVP2 | HU_MVP2.md |
| HU-20 | RF-013 | Could / P3 | MVP2 | HU_MVP2.md |
| HU-21 | RF-022 | Could / P3 | MVP2 | HU_MVP2.md |
