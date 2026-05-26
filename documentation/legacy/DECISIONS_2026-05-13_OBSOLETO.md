> ⚠️ DOCUMENTO HISTÓRICO — NO VIGENTE (archivado 2026-05-25)
> Copia parcial de DECISIONS.md (D-001…D-008, **sin D-009**; última edición 2026-05-13).
> El canónico vigente vive en `documentation/lean-inception/4-decisiones/DECISIONS.md`
> (D-001…D-009). Conservado por trazabilidad histórica.

---

# DECISIONS — CerebroVial

> Registro de decisiones técnicas y de proyecto. Formato ADR ligero. Las decisiones cerradas afectan el código y la documentación; las pendientes (`D-PENDING-*`) son cuestiones abiertas que requieren resolución antes de avanzar a las fases que dependen de ellas.

## Índice

| ID | Estado | Fecha | Título |
|---|---|---|---|
| D-001 | Cerrada | 2026-04-30 | Arquitectura: monolito modular |
| D-002 | Cerrada | 2026-04-30 | Modelo predictivo: RNN |
| D-003 | Cerrada | 2026-04-30 | Deploy: Docker local |
| D-004 | Cerrada | 2026-04-30 | Pi física: demostración conceptual, no entrega |
| D-005 | Cerrada | 2026-04-30 | Números de tesis: actualizar tras validación real |
| D-006 | Cerrada | 2026-05-11 | Modelo predictivo: GRU univariado por intersección |
| D-007 | Cerrada | 2026-05-11 | Módulo de visión: componente demostrable, no en loop de validación |
| D-008 | Cerrada | 2026-05-11 | SUMO end-to-end: datos sintéticos para entrenamiento y validación |
| D-PENDING-001 | **Resuelta por D-006** | — | Modelo: reutilizar `time_then_space.py` o GRU desde cero |

---

## D-001 — Arquitectura: monolito modular
**Fecha:** 2026-04-30 · **Estado:** Cerrada

**Decisión:** El sistema se organiza como un **monolito modular**, no como microservicios. Las carpetas `core_management_api/`, `edge_device/`, `ia_prediction_service/` se entienden como módulos del mismo sistema. Sprint 1 consolida la base común en `shared/` instalable como paquete pip local.

**Justificación:** El refactor del commit `7b26edab` separó el código en carpetas que sugieren microservicios, pero (a) los `common/` de `core_management_api` y `edge_device` son byte-idénticos, (b) no existe API real entre ellos, (c) `ia_prediction_service` es pipeline ML offline, no servicio HTTP. Mantener tres servicios desplegables independientes agregaría complejidad sin valor en un proyecto de tesis con un equipo de dos.

**Impacto:** El docker-compose final tiene `db`, `core_management_api` (incluye prediction + control + vision-consumer), `edge_device` y `frontend_ui`. `ia_prediction_service` queda como herramienta de entrenamiento offline.

---

## D-002 — Modelo predictivo: RNN
**Fecha:** 2026-04-30 · **Estado:** Cerrada (refinada por D-006)

**Decisión:** El modelo predictivo del sistema es una **RNN** (alineado al documento de tesis). El `RandomForestPredictor` actual queda como fallback temporal con flag de configuración hasta que la RNN esté servida.

**Justificación:** El documento de tesis declara una arquitectura RNN. Mantener el RandomForest como fallback evita que una falla de carga del modelo neuronal rompa el endpoint de predicción.

**Impacto:** F3 del TODO implementa la RNN. Ver `D-006` para la materialización concreta (GRU univariado).

**Nota:** D-006 refina esta decisión especificando GRU como la familia de RNN a utilizar, descartando arquitecturas espacio-temporales (STGNN) por estar fuera del alcance.

---

## D-003 — Deploy: Docker local
**Fecha:** 2026-04-30 · **Estado:** Cerrada

**Decisión:** El sistema se despliega localmente con `docker compose up`. No se usa Azure ni ningún cloud por ahora.

**Justificación:** El alcance de la tesis no incluye productivización. Los recursos disponibles (tiempo + cuenta cloud + presupuesto) no justifican el deploy en Azure. La "arquitectura desplegable en Pi/cloud" se demuestra arquitectónicamente y se documenta como plan de productivización en el README final.

**Impacto:** El README de quickstart asume `docker compose up` + `npm run dev`. El Bloque K (defensa) prueba el sistema en máquina limpia, no en cloud.

---

## D-004 — Pi física: demostración conceptual, no entrega
**Fecha:** 2026-04-30 · **Estado:** Cerrada · **Sujeta a confirmación con asesor (A2)**

**Decisión:** No se entrega una Raspberry Pi física en la defensa. Se demuestra que la arquitectura **es desplegable** en Pi (separación de `edge_device` con dependencias mínimas, contenerización separada, comunicación por SSE/HTTP) sin entregar el hardware.

**Justificación:** El proyecto se evalúa por la arquitectura predictiva y la integridad del sistema, no por hardware. La demostración conceptual cubre el espíritu del IoT del documento sin agregar riesgo de hardware roto en la defensa.

**Impacto:** El demo final corre todo en una laptop. El documento de tesis y el video explican qué módulos correrían en Pi (edge_device) y cuáles en servidor central (core_management_api + frontend + db).

**Pendiente:** Confirmar con asesor en llamada A2 del TODO.

---

## D-005 — Números de tesis: actualizar tras validación real
**Fecha:** 2026-04-30 · **Estado:** Cerrada · **Sujeta a confirmación con asesor (A2)**

**Decisión:** Los números declarados en el documento de tesis (88.2% accuracy de detección, 81.3% accuracy del predictor, latencia <2s) se **actualizan a los valores reales** medidos en el Bloque J de validación. Si la realidad es peor, se reporta la realidad.

**Justificación:** Integridad académica. Reportar números que no se pueden reproducir en el demo es riesgo alto en preguntas de defensa. La tesis se defiende mejor con honestidad sobre limitaciones que con marketing inflado.

**Impacto:** Bloque J4 del TODO marca explícitamente la actualización del documento. Si los números reales son peores, el README documenta limitaciones del demo (datos sintéticos, validación parcial, etc.).

**Pendiente:** Confirmar con asesor en llamada A2 del TODO.

---

## D-006 — Modelo predictivo: GRU univariado por intersección
**Fecha:** 2026-05-11 · **Estado:** Cerrada · **Resuelve:** D-PENDING-001 · **Sujeta a confirmación con asesor**

**Decisión:** Se adopta **GRU univariado por intersección** como modelo predictivo de congestión. Se descarta `time_then_space.py` (RNN + DiffConv espacial / STGNN) y los checkpoints asociados (`epoch=79-step=30800.ckpt` y otros). La incorporación de dependencia espacial entre intersecciones (arquitecturas tipo STGNN, vecindad) se declara como **trabajo futuro**.

**Justificación:**

1. **Alcance de validación.** El sistema se valida sobre **una sola intersección** de Miraflores. Una arquitectura espacio-temporal requiere múltiples nodos interrelacionados; no aplica al problema definido.
2. **Cronograma realista.** Con 9 semanas hasta entrega final y dependencias pesadas (SUMO end-to-end, cierre Fase 2, integración completa), no hay margen para definir grafos espaciales, debuggear pipelines tsl/PyTorch Lightning y entrenar STGNN.
3. **Aporte central de tesis.** La contribución es el **sistema integrado** (predicción + control adaptativo + visión + validación cuantitativa), no la sofisticación arquitectónica del predictor aislado.
4. **Defensa académica.** GRU univariado es estándar de la literatura para predicción de serie temporal de tráfico por sensor/intersección. Es justificable y reproducible.

**Impacto:**

- Bloque F del TODO se reescribe: crear `ia_prediction_service/src/models/gru_model.py` (GRU desde cero, simple).
- `time_then_space.py` se mueve a `ia_prediction_service/src/models/legacy/` o se elimina (decisión en limpieza de repo).
- Checkpoints en `ia_prediction_service/notebooks/logs/` se archivan o eliminan.
- METR-LA deja de ser referencia de dataset; los datos vienen de SUMO (ver D-008).
- El capítulo de modelo predictivo de la tesis se reescribe para reflejar GRU univariado y declarar STGNN como trabajo futuro.

**Justificación para la tesis (texto sugerido):**

> *"Se selecciona GRU univariado por intersección dado que la validación del sistema se realiza sobre intersecciones tratadas independientemente. La incorporación de dependencia espacial entre intersecciones (mediante arquitecturas espacio-temporales tipo STGNN o atención sobre vecinos) se identifica como una extensión natural del trabajo y se declara como trabajo futuro, condicionada a validación a escala de red urbana."*

**Pendiente:** Confirmar con asesor (reunión 2026-05-12).

---

## D-007 — Módulo de visión: componente demostrable, no en loop de validación
**Fecha:** 2026-05-11 · **Estado:** Cerrada · **Sujeta a confirmación con asesor**

**Decisión:** El módulo de visión computacional se implementa como **componente funcional demostrable** del sistema, con validación independiente mediante métricas estándar de detección (precisión, recall, mAP de YOLO sobre un dataset etiquetado representativo). **No participa en el loop de validación cuantitativa del sistema integrado**; en su lugar, SUMO provee directamente las métricas de estado (flujo, cola, densidad) que el módulo de visión proveería en producción.

El rol del módulo en la arquitectura del sistema es de **sensor de estado en tiempo real** que alimenta al motor adaptativo con observación del tráfico observado por cámara. La idea original de "ajuste fino del motor mediante visión" se descarta por requerir literatura adicional fuera del cronograma y por no haber input real confiable (los streams de YouTube usados hoy no son fiables a largo plazo).

**Justificación:**

1. **Input no controlable.** Los streams de YouTube actuales pueden apagarse, no son específicos de Miraflores y no proveen ground truth para validación cuantitativa. Depender de ellos para la validación final es un riesgo de demo evitable.
2. **Consistencia metodológica.** Si la validación se hace en SUMO, las métricas de estado deben venir de SUMO. Mezclar simulación con observación real introduce confusión sobre qué se está validando.
3. **Defensa académica.** Tener un módulo de visión con su propia validación acotada (métricas de detección sobre dataset etiquetado) es metodológicamente más limpio que un módulo cuya validación está acoplada al sistema completo.
4. **Alcance temporal.** "Cómo usar visión para ajuste fino de un motor de control adaptativo" es un tema de investigación completo; no cabe en 9 semanas.

**Impacto:**

- El módulo de visión (`edge_device/src/vision/`) se mantiene y se completa para demostración.
- Validación del módulo: dataset etiquetado pequeño (≥200 frames), métricas de detección reportadas.
- En el loop de validación cuantitativa (HU-17 comparación con/sin sistema), las métricas de estado las provee SUMO.
- El video de demo muestra el módulo de visión operando sobre un stream/video grabado, sin ser parte del experimento cuantitativo.
- El capítulo de validación de la tesis separa explícitamente "validación del módulo de visión" (métricas de detección) y "validación del sistema integrado" (KPIs SUMO).

**Justificación para la tesis (texto sugerido):**

> *"El módulo de visión computacional se implementa como sensor de estado en tiempo real del sistema. Su validación se realiza mediante métricas estándar de detección (precisión, recall, mAP) sobre un dataset etiquetado representativo. Para la validación cuantitativa del sistema integrado (predicción + control adaptativo), se utiliza simulación SUMO que provee directamente las métricas de estado que el módulo de visión proveería en operación. Esta separación asegura consistencia metodológica y aísla las fuentes de error del sistema integrado de las fuentes de error del módulo de detección."*

**Pendiente:** Confirmar con asesor (reunión 2026-05-12).

---

## D-008 — SUMO end-to-end: datos sintéticos para entrenamiento y validación
**Fecha:** 2026-05-11 · **Estado:** Cerrada · **Sujeta a confirmación con asesor**

**Decisión:** SUMO (Simulation of Urban MObility) es la **columna vertebral del sistema de datos** del proyecto. Se utiliza para:

1. **Generación del dataset de entrenamiento** del modelo GRU (series temporales sintéticas de flujo/velocidad/ocupación por intersección, bajo distintos patrones de demanda).
2. **Validación cuantitativa del sistema integrado**: comparación "con sistema (GRU + motor adaptativo) vs sin sistema (Webster fijo)" mediante KPIs estándar (tiempo de viaje, longitud de cola, demoras).

Las particiones de entrenamiento y validación son **escenarios SUMO distintos** (distintos seeds, patrones de demanda, eventos) para evitar fuga de información. No se utilizan datos reales de Waze ni datasets públicos (PeMS, METR-LA) como fuente principal de entrenamiento. La incorporación de datos reales de tráfico de Lima (vía acuerdo con la municipalidad) se declara como **trabajo futuro** o como **bono académico** si se obtienen antes de la entrega.

**Justificación:**

1. **No hay acceso a datos reales hoy.** No se tiene API key de Waze ni acuerdo con la municipalidad. Depender de obtenerlo en 9 semanas es riesgo terminal.
2. **Consistencia metodológica fuerte.** Entrenar y validar en el mismo mundo simulado evita el problema de transferibilidad entre datasets distintos. La tesis declara explícitamente este alcance.
3. **Control experimental.** SUMO permite generar dataset ilimitado y controlable: días laborales, fines de semana, hora pico, valle, eventos. Calidad y variabilidad están bajo control del tesista.
4. **Eliminación de dependencias externas.** No hay riesgo de que un servicio público apague endpoints, que un acuerdo se caiga, que un dataset cambie.
5. **Defendible académicamente.** Múltiples tesis de control de tráfico operan en este modo. La limitación se declara explícitamente.

**Riesgo conocido:** El jurado puede objetar "se entrena y valida en el mismo simulador". **Respuesta:** se usan particiones distintas (escenarios, seeds), se declara el límite metodológico explícitamente en el capítulo de alcance, y la generalización a datos reales se identifica como trabajo futuro.

**Impacto:**

- HU-16 (SUMO) sube de "validación al final" a **columna vertebral del sistema**. Empieza en semana 6.
- Cronograma: 1-2 semanas para topología de Miraflores en SUMO + escenarios de demanda + generación de dataset.
- El modelo GRU se entrena sobre el dataset SUMO generado, no sobre METR-LA ni Waze.
- El capítulo de alcance de la tesis declara explícitamente la naturaleza simulada de la validación.
- Si el PO de la municipalidad provee datos antes de la entrega: se usan como **validación adicional** ("el modelo entrenado en simulación se evalúa también sobre datos reales de X periodo, mostrando degradación de Y%"), no como reemplazo del flujo principal.

**Justificación para la tesis (texto sugerido):**

> *"La validación del sistema propuesto se realiza mediante simulación SUMO calibrada con la topología de la intersección de estudio en Miraflores. Tanto el dataset de entrenamiento del modelo predictivo como los escenarios de validación cuantitativa se generan a partir de SUMO con particiones independientes (distintos seeds y patrones de demanda) para evitar fuga de información. La obtención de datos de tráfico reales de la ciudad de Lima se identifica como una limitación reconocida del trabajo, y la calibración del modelo con datos reales mediante acuerdo con entidades municipales se declara como trabajo futuro."*

**Pendiente:** Confirmar con asesor (reunión 2026-05-12).

---

## D-PENDING-001 — Modelo: reutilizar `time_then_space.py` o GRU desde cero
**Estado:** **Resuelta por D-006** (2026-05-11)

**Contexto histórico:** El archivo `ia_prediction_service/src/models/time_then_space.py` implementa una arquitectura **Time-then-Space**: encoder lineal + RNN(cell='gru') temporal + DiffConv espacial + MLPDecoder. La celda recurrente ya era GRU por defecto. Existían 5 checkpoints entrenados en `ia_prediction_service/notebooks/logs/`.

**Resolución:** Ver D-006. Se descarta `time_then_space.py` por exceder el alcance temporal y metodológico del trabajo. Se implementa GRU univariado desde cero.

**Acción de archivo:** Esta entrada se mantiene como traza histórica de la decisión. No mover.
