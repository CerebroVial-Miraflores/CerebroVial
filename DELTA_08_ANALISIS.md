# Análisis de Delta-08 — Semántica de ControlView (insumo para DHU-020)

> **Naturaleza del documento:** reporte de evidencia, no decisión. Presenta el
> detalle concreto del delta entre HU-05 y el código actual de ControlView para
> que la decisión metodológica DHU-020 se cierre en sesión aparte. No se propone
> resolución; las tres opciones de la §5 se enuncian con su costo, no se eligen.

## Preámbulo — ajustes de ruta respecto a la consigna

La estructura real difiere de las rutas indicadas en la tarea. Rutas efectivas
usadas (todas relativas a la raíz del repo `CerebroVial/`, que es donde vive
`CLAUDE.md`; el directorio de trabajo `/Users/rasec/Tesis/` contiene además
carpetas hermanas `Frontend/`, `Backend/`, `Predictor/`, `CerebroVial-old/` que
**no** son el proyecto vivo):

| Documento esperado en consigna | Ruta real |
|---|---|
| `documentation/lean-inception/HU_BLOQUE_B.md` | `documentation/lean-inception/2-backlog/HU_BLOQUE_B.md` |
| `documentation/lean-inception/AUDITORIA_HU_CODIGO.md` | `documentation/lean-inception/planificacion/AUDITORIA_HU_CODIGO.md` |
| `documentation/lean-inception/REPORTE_PLANIFICACION_SPRINT_4.md` | `documentation/lean-inception/planificacion/REPORTE_PLANIFICACION_SPRINT_4.md` |
| `documentation/lean-inception/DECISIONS_HU.md` | `documentation/lean-inception/4-decisiones/DECISIONS_HU.md` |
| `frontend/src/views/ControlView` | `frontend_ui/src/components/views/control/ControlView.tsx` |

Este archivo se deposita en `CerebroVial/DELTA_08_ANALISIS.md` (raíz del repo del
proyecto vivo). No se modificó código ni documento existente alguno.

---

## 1. Qué dice HU-05 exactamente

Transcripción literal de la cabecera y los criterios desde
[HU_BLOQUE_B.md](documentation/lean-inception/2-backlog/HU_BLOQUE_B.md), líneas 220–273.

### Cabecera (Como / Quiero / Para)

| Campo | Contenido |
|---|---|
| **Como** | Operador de Tráfico Municipal |
| **Quiero** | visualizar cuál es la estrategia de control que el sistema está aplicando actualmente en la intersección y los parámetros activos de esa estrategia |
| **Para** | entender qué decisión de control automático está vigente en cada momento y poder evaluar si es coherente con el estado del tráfico que observo |

**Tipo:** HU de Persona (Operador).
**Feature(s) origen:** F07 (Panel del motor adaptativo — estrategia activa).

### Descripción (literal)

> El sistema selecciona automáticamente entre múltiples estrategias de control
> semafórico según el estado predicho y observado del tráfico (esta es la
> naturaleza adaptativa del sistema, núcleo del Objetivo 3 del producto). Pero el
> Operador necesita saber **qué estrategia se está aplicando en este momento** y
> **con qué parámetros**, por dos razones operativas:
>
> 1. **Trazabilidad de la operación:** si reporta un comportamiento anómalo del
>    sistema, debe poder decir "en ese momento se estaba aplicando la estrategia X
>    con tiempos Y".
> 2. **Coherencia percibida:** si el Operador ve colas largas pero la estrategia
>    activa es la que se usa para flujo libre, hay una incoherencia que debe poder
>    identificar.
>
> El panel muestra el nombre de la estrategia vigente (sin exponer detalles
> internos del motor) y los parámetros activos: tiempos de verde asignados a cada
> acceso de la intersección. El panel también muestra cuándo se aplicó la
> estrategia actual, para que el Operador sepa cuánto tiempo lleva vigente.

### Criterios de aceptación (literal)

- **CA-05.1:** Dado que el Operador ha iniciado sesión y el sistema tiene una
  estrategia de control vigente, cuando ingresa al panel de la estrategia activa,
  entonces el sistema muestra el nombre de la estrategia actualmente aplicada y
  los tiempos de verde asignados a cada acceso de la intersección.

- **CA-05.2:** Dado que el Operador está observando el panel de la estrategia
  activa, cuando el sistema muestra la estrategia vigente, entonces también indica
  el timestamp en que esa estrategia se activó, permitiendo al Operador conocer
  cuánto tiempo lleva aplicándose.

- **CA-05.3:** Dado que el Operador tiene el panel abierto, cuando el sistema
  cambia la estrategia activa o ajusta sus parámetros, entonces los valores
  mostrados se actualizan automáticamente sin necesidad de recargar la página, con
  una latencia máxima de 5 segundos desde que el cambio se produce.

- **CA-05.4:** Dado que el sistema no puede determinar la estrategia vigente por
  cualquier causa (por ejemplo, porque el motor adaptativo no está respondiendo),
  cuando el Operador está observando el panel, entonces el sistema mantiene en
  pantalla la última estrategia conocida, la marca visualmente como "no
  confirmada" e indica el tiempo transcurrido desde la última confirmación
  (DHU-005 Caso B).

- **CA-05.5:** Dado que el Operador no ha iniciado sesión, cuando intenta acceder
  al panel de la estrategia activa, entonces el sistema lo redirige a la pantalla
  de login.

### Semántica que prescribe la HU

**Pasiva, sin ambigüedad.** La HU es explícita y redundante en este punto:

- Resumen ejecutivo (línea 233): *"vista pasiva del motor adaptativo"*.
- Notas clave (línea 239): *"Es vista pasiva: «qué está activo ahora»"*.
- El verbo del "Quiero" es **visualizar**, no configurar/simular/recomendar. El
  Operador es observador de un estado que el sistema produce solo; no hay input
  del usuario que altere la decisión del motor.
- El objeto observado es la **estrategia vigente en producción** ("aplicando
  actualmente", "vigente en cada momento"), no un escenario hipotético propuesto
  por el usuario.
- CA-05.2 (timestamp de activación) y CA-05.3 (auto-update ≤5 s) solo tienen
  sentido sobre un estado que evoluciona por sí mismo en el tiempo — refuerzan la
  semántica pasiva/streaming, incompatibles con un request-response disparado por
  el usuario.

HU-06 (el "por qué", líneas 277–334) y HU-07 (el aviso activo de cambio, líneas
338+) son HUs hermanas que consumen el mismo motor (TTH-10) pero **no** introducen
interactividad: HU-06 sigue siendo lectura pasiva de un texto plantillado.

---

## 2. Qué hace el código actual

ControlView es un **simulador/playground interactivo request-response**, no un
panel pasivo.

### Comportamiento real

[ControlView.tsx](frontend_ui/src/components/views/control/ControlView.tsx) monta
un formulario de edición de un estado de intersección **hipotético** y, al pulsar
un botón, pide una recomendación al motor:

- **Inputs del usuario:**
  - Identificador de intersección — campo de texto libre, default `INT_001`
    ([ControlView.tsx:100](frontend_ui/src/components/views/control/ControlView.tsx#L100), L172–178).
  - "Tiempo perdido por ciclo" — slider 2–16 s
    ([Slider.tsx](frontend_ui/src/components/views/control/Slider.tsx), L180–189).
  - Fases editables (flow, saturation_flow, queue, has_pedestrian) vía
    [PhaseEditor.tsx](frontend_ui/src/components/views/control/PhaseEditor.tsx) (L192).
  - Presets que rellenan el formulario con escenarios de demo ("Off-peak típico",
    "Peak normal", "Peak saturado", "Webster infeasible") —
    [PresetButtons.tsx](frontend_ui/src/components/views/control/PresetButtons.tsx) +
    [controlTypes.ts:14-51](frontend_ui/src/components/views/control/controlTypes.ts#L14-L51).
  - Botón **"Recomendar"** con ícono `Play`
    ([ControlView.tsx:194-202](frontend_ui/src/components/views/control/ControlView.tsx#L194-L202)).

- **Endpoint consumido:** un único `POST /control/recommend`, disparado por el
  click del usuario, enviando el estado editado en el body
  ([controlService.ts:56-64](frontend_ui/src/services/controlService.ts#L56-L64);
  backend en [routes.py:81-99](core_management_api/src/control/presentation/api/routes.py#L81-L99)).
  No existe ningún endpoint de "estrategia vigente"; el motor se invoca contra el
  estado que el usuario inventa, no contra el estado real de una intersección
  operativa.

- **Qué muestra:** [RecommendationPanel.tsx](frontend_ui/src/components/views/control/RecommendationPanel.tsx)
  renderiza el output del cálculo: modo elegido (`webster` / `max_pressure`),
  ciclo en segundos, métricas derivadas del formulario (flow_total, Y, PEAK/OFF-PEAK),
  ciclo semafórico animado ([TrafficLightCycle.tsx](frontend_ui/src/components/views/control/TrafficLightCycle.tsx)),
  tiempos por fase ([TimingBar.tsx](frontend_ui/src/components/views/control/TimingBar.tsx)),
  una explicación narrativa, un **"Log técnico (para operador C4)"** con el
  `reasoning` crudo del motor (L188–192), y los ajustes MTC aplicados. El caso
  patológico `webster_infeasible` se presenta con una tarjeta pedagógica
  ([Pedagogical422Card.tsx](frontend_ui/src/components/views/control/Pedagogical422Card.tsx)).

- **Efecto sobre el motor adaptativo:** ninguno persistente. `/control/recommend`
  es una función pura: recibe un estado, devuelve una recomendación, no guarda
  nada ni cambia qué corre "en producción". El motor es un singleton en memoria
  ([routes.py:29-40](core_management_api/src/control/presentation/api/routes.py#L29-L40))
  que solo calcula sobre el payload recibido. No hay noción de "estrategia
  vigente" en ninguna intersección real.

- **Frecuencia de actualización:** ninguna automática. El panel solo cambia cuando
  el usuario vuelve a pulsar "Recomendar". No hay polling, SSE ni WebSocket (esto
  coincide con Delta-07 de la auditoría: cero infraestructura realtime).

### Archivos involucrados

**Frontend** (`frontend_ui/src/`):

| Archivo | Rol |
|---|---|
| `components/views/control/ControlView.tsx` | Orquesta formulario + invocación del endpoint |
| `components/views/control/PhaseEditor.tsx` | Edición de fases (input del usuario) |
| `components/views/control/Slider.tsx` | Input del tiempo perdido |
| `components/views/control/PresetButtons.tsx` | Escenarios de demo |
| `components/views/control/controlTypes.ts` | PRESETS, etiquetas de modo, subtítulos de fase |
| `components/views/control/RecommendationPanel.tsx` | Render del output del cálculo |
| `components/views/control/TrafficLightCycle.tsx` | Animación del ciclo semafórico |
| `components/views/control/TimingBar.tsx` | Barra de tiempos por fase |
| `components/views/control/ModeSelector.tsx` | Selector/visualización de modo |
| `components/views/control/Pedagogical422Card.tsx` | Tarjeta pedagógica `webster_infeasible` |
| `services/controlService.ts` | Cliente HTTP de `/control/recommend` + parse de errores |
| `App.tsx` | Cableado: `activeTab === 'control'` renderiza `<ControlView />` (L11, L52) |
| `components/layout/Sidebar.tsx` | Entrada de navegación que activa el tab `control` |

**Backend** (`core_management_api/src/control/`):

| Archivo | Rol |
|---|---|
| `presentation/api/routes.py` | Endpoint `POST /control/recommend` |
| `presentation/api/schemas.py` | Schemas Pydantic (IntersectionState, ControlRecommendation, ErrorDetail…) |
| `application/adaptive_engine.py` | Motor que produce la recomendación + `reasoning` |
| `application/webster.py` | Cálculo Webster (lanza `WebsterInfeasible`) |
| `application/max_pressure.py` *(referenciado por tests)* | Estrategia Max Pressure |
| `application/mtc.py` *(referenciado por tests)* | Ajustes normativos MTC |

---

## 3. Delta concreto

| Aspecto | Lo que dice HU-05 | Lo que hace el código |
|---|---|---|
| **Dirección del flujo** | Pasivo: el sistema empuja el estado vigente; el Operador observa. | Activo / request-response: el usuario edita un estado y pide una recomendación pulsando "Recomendar". |
| **Inputs del usuario** | Ninguno sobre la decisión del motor (solo abrir el panel, ya autenticado). | Muchos: intersection_id (texto libre), slider de tiempo perdido, editor de fases (flow/saturation/queue/peatón), presets de demo. |
| **Efecto sobre el motor** | Ninguno: el motor decide solo en producción; la vista no lo altera. | Ninguno persistente tampoco — **pero por motivo distinto**: el motor se invoca *bajo demanda del usuario* sobre un estado inventado; no existe "motor corriendo en producción" al que la vista se asome. |
| **Fuente de datos mostrados** | Estado real de la intersección operativa ("aplicando actualmente"). | Estado hipotético tecleado por el usuario en el formulario / preset. |
| **Persistencia de decisiones** | Implícita: existe una estrategia vigente con timestamp de activación (CA-05.2) y, transversalmente, registro histórico en HU-08. | Nula: `/control/recommend` es función pura, sin estado, sin timestamp de activación, sin historial. |
| **Frecuencia de actualización** | Auto-update ≤ 5 s ante cambio de estrategia o parámetros (CA-05.3). | Solo al re-pulsar "Recomendar". Sin polling/SSE/WebSocket (concordante con Delta-07). |
| **Nombre de la estrategia** | Etiqueta legible/autoexplicativa para el Operador, agnóstica al motor (DHU-006). | Etiquetas `Webster (off-peak)` / `Max Pressure (peak)` — nombres técnicos del algoritmo, no del dominio operativo. |
| **Explicación** | (HU-06) texto plantillado, lenguaje del dominio, sin jerga técnica. | Narrativa + **"Log técnico (para operador C4)"** con `reasoning` crudo — lenguaje técnico (cubierto por Delta-09). |
| **Autenticación** | CA-05.5: redirección a login si no autenticado. | No implementada (sin gate de auth en la vista). |
| **Robustez (motor caído)** | CA-05.4: mantener última estrategia conocida, marcarla "no confirmada", mostrar tiempo desde última confirmación (DHU-005 Caso B). | Manejo de errores `webster_infeasible` / `invalid_state` / genérico con botón "Reintentar" — semántica de *fallo de cálculo del request*, no de *fuente vigente no confirmada*. |

### Capacidades del código que la HU no menciona

Declaradas explícitamente, porque son las que generan el conflicto (no son
simplemente "extras"):

1. **Playground de simulación:** edición libre de un estado de intersección
   arbitrario para explorar qué recomendaría el motor. La HU-05 no contempla que
   el usuario proponga estados.
2. **Presets pedagógicos:** escenarios de demostración ("Off-peak típico", "Peak
   normal", "Peak saturado", "Webster infeasible") diseñados para enseñar el
   comportamiento del motor, incluido el caso patológico que devuelve 422.
3. **Tarjeta pedagógica de `webster_infeasible`** ([Pedagogical422Card.tsx](frontend_ui/src/components/views/control/Pedagogical422Card.tsx)):
   material explicativo del caso degenerado de Webster. Valor docente (contexto
   tesis), ajeno a la operación del Operador.
4. **Métricas instructivas en vivo** (flow_total, Y = Σ flow/sat, umbral 1500,
   badge PEAK/OFF-PEAK): muestran *cómo* el motor decide entre estrategias —
   pedagogía del algoritmo, no estado operativo.

> En síntesis: el código y la HU coinciden en *qué se muestra* (modo + tiempos por
> fase) pero divergen radicalmente en *de dónde sale el dato* (estado inventado vs.
> estado vigente) y *quién dispara la actualización* (el usuario vs. el sistema).
> El delta no es de presentación; es de **semántica de la fuente y la dirección
> del flujo**.

---

## 4. Alcance del refactor (alinear el código a la semántica pasiva de HU-05)

Estimación cualitativa del trabajo necesario para llevar ControlView a "vista
pasiva del estado vigente". Pensado como inventario de impacto, no como plan.

### Frontend

| Archivo | Impacto |
|---|---|
| `ControlView.tsx` | **Reescritura mayor.** Eliminar todo el estado de formulario (`intersectionId`, `lostTime`, `phases`), el hook `useRecommendControl` (mutate/submit), validación, presets y el botón "Recomendar". Sustituir por un fetch/suscripción de solo lectura a la estrategia vigente. Probablemente conviene renombrar el componente (p.ej. `ActiveStrategyView`) por claridad semántica. |
| `PhaseEditor.tsx` | **Eliminar** (o reusar como display de solo lectura). Es input del usuario; en vista pasiva no hay edición. |
| `Slider.tsx` | **Eliminar** del flujo de ControlView (input del usuario). |
| `PresetButtons.tsx` + PRESETS en `controlTypes.ts` | **Eliminar** (escenarios de demo, ajenos a la operación). |
| `RecommendationPanel.tsx` | **Reescritura media.** Reutilizable para mostrar modo + tiempos por fase, pero hay que: quitar estados `idle`/`loading`/`error` ligados al request del usuario; agregar timestamp de activación (CA-05.2); cambiar el "Log técnico (para operador C4)" por lenguaje de dominio (esto es Delta-09/HU-06, queda fuera del estricto Delta-08 pero se toca el mismo archivo). |
| `TrafficLightCycle.tsx`, `TimingBar.tsx`, `ModeSelector.tsx` | **Reutilizables casi tal cual** — visualizan tiempos/modo, agnósticos al origen del dato. Ajuste menor de props. |
| `Pedagogical422Card.tsx` | **Eliminar** del flujo operativo (material pedagógico del caso 422). |
| `controlTypes.ts` | **Edición.** Quitar `PRESETS` y `PlaybackSpeed`; conservar `MODE_LABEL`/`PHASE_SUBTITLES` (probablemente remapear a etiquetas de dominio, DHU-006). |
| `services/controlService.ts` | **Edición media.** Sustituir `recommend(state)` por un getter de estrategia vigente (p.ej. `getActiveStrategy(intersectionId)`); y/o un canal realtime para CA-05.3. |
| `App.tsx`, `Sidebar.tsx` | **Edición menor.** Si se renombra el componente, actualizar import y, eventualmente, el label del tab. |

### Backend

| Archivo | Impacto |
|---|---|
| `presentation/api/routes.py` | **Adición / cambio de contrato.** Se necesita un endpoint de lectura del estado vigente, p.ej. `GET /control/strategy/active` (o `/current`), que devuelva estrategia + tiempos + timestamp de activación. Decisión abierta: ¿se conserva `POST /control/recommend` (para playground/admin) o se elimina? |
| `application/adaptive_engine.py` | **Cambio estructural posible.** Hoy el motor es función pura sin estado. Una "estrategia vigente con timestamp de activación" implica que algo *mantiene y persiste* el estado vigente por intersección. Esto es trabajo nuevo, no refactor de lo existente, y roza la advertencia de CLAUDE.md sobre cambios estructurales/modelo de BD → **parar y preguntar**. |
| `presentation/api/schemas.py` | **Edición.** Nuevo schema de respuesta para la estrategia vigente (incluye `activated_at`). |
| Infra realtime (no existe) | **Trabajo nuevo (Delta-07).** CA-05.3 (≤5 s) exige polling o SSE/WebSocket, hoy inexistente. Es dependencia transversal del sprint, no exclusiva de Delta-08. |

### Tests que cambiarían

- **Frontend:** no existen tests de control en `frontend_ui` (búsqueda de
  `*.test.*`/`*.spec.*` sobre control/recommend/webster: cero resultados). El
  refactor no rompe tests de UI porque no los hay — pero sí deja sin cobertura un
  cambio grande.
- **Backend:** `core_management_api/tests/control/` contiene `test_engine.py`,
  `test_webster.py`, `test_mtc.py`, `test_max_pressure.py`, `conftest.py`. Estos
  ejercitan la **matemática** del motor (Webster, Max Pressure, MTC), no la
  semántica del endpoint. Si `/control/recommend` se conserva, **no cambian**. Si
  se agrega persistencia de estrategia vigente, harían falta tests **nuevos** para
  el getter/estado vigente; los existentes seguirían válidos.

### Dependencias externas a ControlView que tocaría el refactor

- **`App.tsx` (L11, L52) y `Sidebar.tsx`:** únicos consumidores del componente en
  el frontend. `ControlView` no es importado por ninguna otra vista (búsqueda:
  solo `App.tsx` y el propio archivo lo referencian).
- **`controlService.ts`** es consumido solo por la familia control; cambiar su
  contrato no afecta otras vistas.
- **`POST /control/recommend`:** su único cliente es `controlService.recommend`.
  No hay otros llamadores en el frontend, así que eliminarlo o conservarlo es una
  decisión sin efecto colateral fuera de la familia control.

---

## 5. Opciones de resolución

Enunciadas, no elegidas. Esfuerzo relativo: S (pequeño) / M (medio) / L (grande).

### Opción A — Refactor a vista pasiva pura (eliminar playground)

- **Archivos afectados:**
  - *Eliminar:* `PhaseEditor.tsx`, `Slider.tsx`, `PresetButtons.tsx`,
    `Pedagogical422Card.tsx`; PRESETS en `controlTypes.ts`.
  - *Reescribir:* `ControlView.tsx` (→ vista de solo lectura), `RecommendationPanel.tsx`,
    `controlService.ts`.
  - *Reutilizar con ajuste:* `TrafficLightCycle.tsx`, `TimingBar.tsx`, `ModeSelector.tsx`.
  - *Backend:* nuevo `GET /control/strategy/active` + schema + persistencia de
    estrategia vigente en `adaptive_engine.py`; posible eliminación de
    `POST /control/recommend`. Infra realtime para CA-05.3.
  - *Wiring:* `App.tsx`, `Sidebar.tsx`.
- **Esfuerzo:** **L.** Es el más caro: implica trabajo nuevo de backend
  (estado vigente persistido + endpoint + realtime) además de borrar UI.
- **Implicancia sobre HU-05:** **mantener** la HU tal cual. El código se alinea a
  la HU; se pierde toda la capacidad pedagógica/demo (costo para el contexto tesis).

### Opción B — Vista pasiva + playground preservado como vista interna separada (debugging / Administrador)

- **Archivos afectados:**
  - *Crear:* nueva vista pasiva (p.ej. `ActiveStrategyView.tsx`) reutilizando
    `TrafficLightCycle`/`TimingBar`/`ModeSelector` y una variante de
    `RecommendationPanel`.
  - *Conservar:* el `ControlView.tsx` actual y todos sus subcomponentes
    (PhaseEditor, Slider, Presets, Pedagogical422Card) movidos a un tab/área de
    Administrador o de debugging, posiblemente bajo `AdminView`.
  - *Backend:* se conserva `POST /control/recommend` (lo usa el playground) **y**
    se agrega `GET /control/strategy/active` + persistencia + realtime para la
    vista pasiva.
  - *Wiring:* `App.tsx`/`Sidebar.tsx` ganan separación entre tab operativo
    (Operador) y tab playground (Administrador); idealmente gateado por rol.
- **Esfuerzo:** **M–L.** Más superficie total que A (se mantienen dos cosas) pero
  con menos destrucción y menor riesgo: el backend nuevo es el mismo que en A,
  pero no hay que borrar ni reescribir el playground existente. Coincide con la
  recomendación informal anotada en R1 del Sprint 4 ("vista pasiva + tab admin
  oculto con el playground actual").
- **Implicancia sobre HU-05:** **mantener** la HU. El playground queda fuera del
  alcance de HU-05 (lo cubriría una HU/TTH de herramienta administrativa o de
  validación de tesis, a redactar aparte).

### Opción C — Enmendar HU-05 para legitimar la semántica playground actual

- **Archivos afectados:**
  - *Documentación:* `HU_BLOQUE_B.md` (reescritura de cabecera, descripción y CAs
    de HU-05), `AUDITORIA_HU_CODIGO.md` (cerrar Delta-08), `DECISIONS_HU.md`
    (registrar DHU-020), `REPORTE_PLANIFICACION_SPRINT_4.md` (R1).
  - *Código:* mínimo o nulo (el código ya hace lo que la HU enmendada describiría);
    a lo sumo limpieza de lenguaje técnico (Delta-09) y auth (CA-05.5).
- **Esfuerzo:** **S** en código, **M** en documentación y coherencia (hay que
  recablear las dependencias de HU-05 con HU-06/HU-07/HU-08, que asumen vista
  pasiva, y revisar el "núcleo del Objetivo 3: naturaleza adaptativa").
- **Implicancia sobre HU-05:** **enmendar.** Riesgo metodológico alto: HU-05 dejaría
  de describir la supervisión operativa del motor en producción (su justificación
  declarada: trazabilidad + coherencia percibida del Operador) y pasaría a describir
  una herramienta de simulación. CA-05.2 (timestamp de activación) y CA-05.3
  (auto-update) perderían sentido o habría que reescribirlos. Afecta la coherencia
  del Bloque B completo.

---

## 6. Hallazgos inesperados

1. **Rutas distintas a la consigna.** Los documentos viven en subcarpetas
   numeradas (`2-backlog/`, `planificacion/`, `4-decisiones/`) y el frontend es
   `frontend_ui/` (no `frontend/`), con ControlView en
   `components/views/control/`, no `views/`. Detallado en el preámbulo.

2. **Directorio de trabajo con repos hermanos.** `/Users/rasec/Tesis/` contiene
   `Frontend/`, `Backend/`, `Predictor/` y `CerebroVial-old/` además del proyecto
   vivo `CerebroVial/`. Son herencia del periodo de 3 repos (confirmado por
   CLAUDE.md: "Las carpetas separadas son herencia de cuando había 3 repos"). El
   análisis se circunscribió a `CerebroVial/`, el monolito vivo. **Conviene
   confirmar** que el frontend operativo es `CerebroVial/frontend_ui` y no
   `Tesis/Frontend`, para no analizar código muerto.

3. **El delta es más profundo que "pasivo vs. interactivo".** No es solo que el
   código tenga botones de más: es que **no existe en el backend la noción de
   "estrategia vigente en producción"**. `/control/recommend` es una función pura
   sin estado. Por tanto, "alinear a HU-05" no es solo quitar UI: requiere
   *construir* el concepto de estado vigente persistido por intersección con
   timestamp de activación. Esto cruza la advertencia de CLAUDE.md sobre cambios
   estructurales y de modelo de BD ("parar y preguntar al usuario"). DHU-020 debería
   ser consciente de que la Opción A y la B comparten ese costo de backend nuevo.

4. **Delta-08 viaja acompañado de Delta-07 y Delta-09.** CA-05.3 (auto-update ≤5 s)
   es imposible sin la infraestructura realtime ausente (Delta-07), y el "Log
   técnico (para operador C4)" choca con el lenguaje de dominio que pide HU-05/HU-06
   (Delta-09). Cualquier refactor que cierre Delta-08 toca los mismos archivos que
   esos dos deltas; conviene que DHU-020 declare explícitamente si los aborda en
   conjunto o los mantiene separados.

5. **Cobertura de tests asimétrica.** El motor tiene tests sólidos
   (`test_engine`, `test_webster`, `test_mtc`, `test_max_pressure`) pero **no hay
   un solo test de la vista de control ni del endpoint `/control/recommend`** desde
   el frontend. El refactor más grande del sprint (HU-05) partiría sin red de
   seguridad en la capa que más cambia.

6. **El propio backlog ya marca la tensión.** HU-07 (línea 355) define HU-05
   literalmente como *"vista pasiva del estado actual"* y la contrasta con el aviso
   activo. La semántica pasiva no es interpretación de este análisis: está
   ratificada de forma cruzada por las HUs vecinas. La Opción C tendría que
   deshacer esas referencias cruzadas.
