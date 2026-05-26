# Historias de Usuario — Bloque E (Cerrado)

> Quinta entrega del Product Backlog del proyecto CerebroVial.
>
> **Estado:** Bloque E cerrado y aprobado. Bloques A, B, C, D y F del MVP1 cerrados, y MVP2 también cerrado el 2026-05-16 (DHU-017). **Con el cierre del MVP2, la redacción del Product Backlog del proyecto queda completa en su componente funcional: 21 HUs operativas (HU-01 a HU-21) + 11 TTH (TTH-01 a TTH-11).** Pendiente: documento RF/RNF (DHU-007), Planning Poker, MoSCoW, implementación SCRUM del MVP1. Las 5 TTH de este bloque (TTH-07 SUMO, TTH-08 visión, TTH-09 GRU, TTH-10 motor adaptativo, TTH-11 spike) son consumidas por HUs del MVP1 y del MVP2 sin modificación: HU-20 del MVP2 consume TTH-09/CT-09.5 extendido con persistencia paralela del modelo de respaldo (extensión inglobada como CA-20.1 a CA-20.4 de HU-20, sin TTH nueva).
>
> **Fecha de cierre:** 2026-05-15

---

## Contexto

Este documento contiene el cierre del **Bloque E — Componentes centrales del sistema** del Sequencer del Lean Inception (ver `LEAN_INCEPTION_CEREBROVIAL.md`, sección 9, y `FEATURE_BACKLOG_DETALLADO.md`, Bloque E).

A diferencia de los Bloques A, B, C y D, el Bloque E **no contiene Historias de Usuario operativas**. Las cuatro features que componen el bloque (F32, F33, F34, F35) son componentes técnicos internos del sistema sin Persona del producto beneficiaria directa. Aplicando los criterios de DHU-004, las cuatro se modelan como **Tareas Técnicas Habilitadoras (TTH)** documentadas en `TAREAS_TECNICAS_HABILITADORAS.md`. Durante la redacción se identificó la necesidad de una quinta TTH (TTH-11, spike de investigación de hiperparámetros temporales del modelo predictivo) que se incorporó al Bloque E formalizando la actualización de DHU-015.

Las decisiones que sustentan este cierre, tomadas durante la redacción del bloque, son:

- **DHU-015** (decisión metodológica del Bloque E, en `DECISIONS_HU.md`): clasificación HU/TTH de las features F32, F33, F34, F35 mediante aplicación de DHU-004. Resultado: 5 TTH operativas (TTH-07 a TTH-11), 0 HUs operativas. Ver `DECISIONS_HU.md` para fundamentación completa.
- **D-005** (registro técnico): los umbrales aspiracionales del modelo predictivo (≥80% accuracy sobre nivel discreto, TTH-09 CT-09.7) y del módulo de visión (≥80% sobre dataset etiquetado, TTH-08 CT-08.9) se establecen como objetivos, no como criterios bloqueantes. Si la realidad medida es peor, se reporta honestamente conforme a D-005.
- **D-006** (técnica): GRU univariado por intersección, descarta STGNN. Trabajado en TTH-09.
- **D-007** (técnica): visión como componente demostrable, no en loop de validación cuantitativa. Trabajado en TTH-08.
- **D-008** (técnica): SUMO como columna vertebral de datos. Trabajado en TTH-07.
- **D-009** (técnica): jam level 0-5 de Waze como variable predicha. Materializado en TTH-07 (mapeo) y TTH-09 (output del modelo).

Las TTH del Bloque E siguen las reglas metodológicas establecidas en bloques previos:

- **DHU-003** (sujetos válidos): las TTH no tienen sujeto Persona; declaran imperativamente el trabajo técnico.
- **DHU-004** (TTH como categoría separada): cada TTH cumple al menos uno de los cuatro criterios para clasificar como TTH (en el caso del Bloque E, los cuatro componentes cumplen los cuatro criterios).
- **DHU-006** (HUs agnósticas a la implementación): las HUs ya cerradas que consumen los componentes del Bloque E (HU-02, HU-03, HU-04, HU-05, HU-06, HU-07, HU-08, HU-14, HU-15) son agnósticas a las tecnologías concretas. Las TTH del Bloque E sí mencionan las tecnologías por su naturaleza técnica (SUMO, YOLO, GRU, Webster, Max Pressure, MTC).
- **DHU-007** (RNF declarados): las TTH no requieren sección "Candidatos a RNF" porque sus criterios técnicos de terminado ya incluyen requisitos no funcionales como criterios de Done.

Ver `DECISIONS_HU.md` para fundamentación completa de todas las decisiones metodológicas.

---

## Mapeo de features del Bloque E

Las 4 features del Bloque E (F32, F33, F34, F35) se mapearon a 4 TTH operativas, conforme a DHU-015. Durante la redacción se identificó la necesidad de una quinta TTH (TTH-11) que se incorporó al bloque formalizando la actualización de DHU-015 (4 → 5 TTH).

| TTH | Título | Feature origen |
|---|---|---|
| TTH-07 | Integración con SUMO para simulación del entorno | F32 |
| TTH-08 | Módulo de visión computacional que produce métricas de estado | F33 |
| TTH-09 | Modelo predictivo GRU servido vía API | F34 |
| TTH-10 | Motor adaptativo de control semafórico | F35 |
| TTH-11 | Spike de calibración de hiperparámetros temporales del modelo predictivo | (TTH derivada de TTH-09, no asociada a feature) |

**Total Bloque E:** 0 HUs operativas + 5 TTH nuevas.

**Por qué 0 HUs operativas (resumen de DHU-015):** Los cuatro componentes del Bloque E (F32 SUMO, F33 visión, F34 predictor, F35 motor) son sustratos técnicos del sistema sin Persona del producto beneficiaria directa. El valor al usuario llega vía las HUs ya cerradas en bloques previos: HU-02/03/04 (predicciones y monitoreo del Operador), HU-05/06/07/08 (motor adaptativo del Operador), HU-14 (métricas del modelo para el Administrador), HU-15 (configuración de parámetros). Las cuatro features cumplen los cuatro criterios de DHU-004 simultáneamente. La aplicación de DHU-004 al Bloque E es el patrón establecido (DHU-010 en Bloque C, DHU-013 en Bloque D).

**Por qué se agregó TTH-11 durante la redacción:** Al cerrar la decisión arquitectónica del modelo predictivo (multi-output, lookback, horizonte) se identificó que estos hiperparámetros temporales merecen sustentación bibliográfica y empírica explícita para defensa académica. Se abrió TTH-11 como spike de investigación independiente con entregable documental, prerrequisito documental de TTH-09. El documento entregable de TTH-11 se ubicará en `documentation/docs/` (sugerencia de nombre: `INVESTIGACION_HIPERPARAMETROS_TEMPORALES.md`); no se redacta en esta sesión, queda como entregable pendiente de implementación.

---

## Orden de redacción aplicado

Por dependencias técnicas declaradas en DHU-015:

1. **TTH-07 (SUMO).** Prerrequisito de TTH-09 (dataset) y de la validación que TTH-10 consume.
2. **TTH-11 (Spike de hiperparámetros temporales).** Prerrequisito documental de TTH-09; agregada durante la redacción del bloque.
3. **TTH-09 (GRU).** Consume dataset de TTH-07 y sustentación de TTH-11.
4. **TTH-10 (Motor adaptativo).** Consume predicciones de TTH-09 y estado observado de TTH-07 en validación.
5. **TTH-08 (Visión).** Independiente del eje crítico; redactada al final por dependencia menor en MVP1.

---

## Mapa de consumidores

| TTH | Consumidores en HUs cerradas | Otras TTH que dependen |
|---|---|---|
| TTH-07 | (ninguna HU directa; alimenta validación cuantitativa y otras TTH) | TTH-09 (dataset de entrenamiento), TTH-10 (entorno de validación) |
| TTH-08 | (ninguna HU consume operativamente en MVP1; HU-02 es agnóstica a fuente y en MVP1 consume de SUMO) | TTH-04 (Nivel 1 de cascada, contractualmente preservado) |
| TTH-09 | HU-03 (predicción), HU-04 (vista combinada), HU-14 (métricas del modelo) | TTH-10 (consume predicciones), TTH-04 (Nivel 2 invoca al RandomForest baseline preservado por TTH-09) |
| TTH-10 | HU-05 (estrategia activa), HU-06 (explicación), HU-07 (notificación), HU-08 (historial), HU-15 (parámetros configurables) | TTH-04 (Nivel 3 invoca tiempos preconfigurados de TTH-05 cuando TTH-10 cae) |
| TTH-11 | (ninguna HU directa; alimenta sustentación documental) | TTH-09 (consume hiperparámetros temporales recomendados) |

---

## Validaciones declaradas por TTH

| TTH | Tipo de validación | Detalle |
|---|---|---|
| TTH-07 | Funcional | Topología cargada + simulación end-to-end vía TraCI + dataset generado + escenarios reproducibles + integración con motor adaptativo demostrable end-to-end (CT-07.5). |
| TTH-08 | Independiente, métricas de detección | Precisión, recall, mAP sobre dataset etiquetado propio ≥200 frames (CT-08.9). Objetivo aspiracional accuracy ≥ 80%. NO entra al loop de validación cuantitativa del sistema integrado (D-007). |
| TTH-09 | Funcional + cuantitativo de modelo | Endpoint sirviendo predicciones + MAE/RMSE sobre ratio continuo + accuracy sobre nivel discreto 0-5 + matriz de confusión 6×6 (CT-09.6). Objetivo aspiracional accuracy ≥ 80% (CT-09.7). |
| TTH-10 | Funcional | Las dos estrategias adaptativas operan correctamente, AdaptiveEngine selecciona según criterios, MTC aplica reglas duras documentadas, integración con TTH-09/TTH-07/TTH-04 funciona end-to-end. La validación cuantitativa del sistema (mejora vs control fijo) pertenece al capítulo de validación de la tesis, no al Done de TTH-10. |
| TTH-11 | Documental | Documento entregable en `documentation/docs/` con revisión bibliográfica (mínimo 5 fuentes), exploración empírica mínima (3 combinaciones), recomendación final consolidada, y limitaciones declaradas (CT-11.1 a CT-11.8). |

---

## Decisiones tomadas durante la redacción del Bloque E

Las siguientes decisiones se cerraron durante el diálogo de redacción y están consolidadas en DHU-015 o son notas operativas internas:

1. **Arquitectura del motor adaptativo (TTH-10).** Conforme a `CONTROL.md` y a la clarificación arquitectónica registrada durante la redacción: el motor es una **pipeline de dos etapas** (selección entre Webster y Max Pressure como estrategias adaptativas; aplicación de MTC como capa de reglas duras post-procesamiento), no un selector tripartita entre tres estrategias. Esta clarificación afecta nominalmente la descripción de F35 en `FEATURE_BACKLOG_DETALLADO.md` (título "Motor adaptativo (Webster + MaxPressure + MTC)" se preserva por trazabilidad histórica pero la descripción se ajusta) y la Fase 3 de `EVOLUCION_TESIS.md`.

2. **Arquitectura del modelo predictivo (TTH-09).** Multi-output (un GRU univariado por dirección, cada uno produce un vector de predicciones a múltiples horizontes en una sola inferencia). Esto materializa el UX del slider de HU-03 sin reabrir esa HU: una sola llamada al backend devuelve todas las predicciones que el slider del Operador recorre. El detalle vive en TTH-09 CT-09.2.

3. **Hiperparámetros temporales (TTH-11).** Cuatro hiperparámetros temporales identificados como acoplados: paso de muestreo (Δt_in), ventana de entrada (lookback), horizonte de predicción, frecuencia de re-inferencia del endpoint. Sustentación bibliográfica y empírica en documento dedicado (TTH-11). Valores provisionales para arranque de implementación de TTH-09 si TTH-11 no ha cerrado: Δt_in = 60 segundos, lookback = 30 minutos, horizonte = 60 minutos.

4. **Anclaje de Δt en TTH-07.** La nota técnica de TTH-07 sobre granularidad del dataset (originalmente "30-60s referenciales") se ancla a 60 segundos provisionales, con la cláusula de que el valor definitivo lo cierra TTH-11.

5. **Refactor del módulo de visión (TTH-08).** El módulo se reconstruye desde cero como parte del refactor del Bloque E. El código predecesor en `edge_device/src/vision/` queda como exploración descartada arquitectónicamente; sirve como referencia histórica pero no se preserva. Patrón análogo a `time_then_space.py` movido a `legacy/` por D-006.

6. **Asignación direccional en visión.** Resuelta mediante polígonos ROI configurables por intersección al desplegar (Opción A de la literatura estándar de la industria). Sin UI dedicada en MVP1; configuración vía archivo del backend.

7. **Output visual del módulo de visión.** Declarado como CT explícito (CT-08.8): stream de frames procesados con bounding boxes etiquetadas con clase y velocidad estimada por tracker. Materializa el rol del módulo como "componente demostrable" declarado en D-007.

8. **RandomForest baseline preservado.** No se rediseña ni se elimina. Continúa como predictor de respaldo invocado por TTH-04 en Nivel 2 de la cascada de fallback. TTH-09 introduce el GRU como modelo principal sin reemplazar al RandomForest. Coordinación a nivel de cascada de TTH-04, no de TTH-09.

9. **Rastreo de valores del documento de tesis.** Las cifras "81.3% accuracy del predictor" y "88.2% accuracy de detección" declaradas en el documento de tesis se rastrean durante la implementación de TTH-09 y TTH-08 respectivamente, conforme a D-005. Si son cifras de relleno se sustituyen por las métricas reales medidas.

10. **Topologías de cuatro direcciones como alcance MVP1.** Las intersecciones genéricas de TTH-07 (CT-07.1) y los modelos de TTH-09 están dimensionados para 4 accesos. Topologías con más direcciones o múltiples intersecciones interrelacionadas son trabajo futuro: en visión es nota técnica de TTH-08; en predicción la extensión a red urbana es F37 (arquitecturas espacio-temporales tipo STGNN) con `legacy/time_then_space.py` como punto de partida documentado.

---

## Resumen del Bloque E

| TTH | Título | Tipo | Feature origen | Clasificación MVP |
|---|---|---|---|---|
| TTH-07 | Integración con SUMO para simulación del entorno | TTH del Bloque E | F32 | MVP1 |
| TTH-08 | Módulo de visión computacional que produce métricas de estado | TTH del Bloque E | F33 | MVP1 |
| TTH-09 | Modelo predictivo GRU servido vía API | TTH del Bloque E | F34 | MVP1 |
| TTH-10 | Motor adaptativo de control semafórico | TTH del Bloque E | F35 | MVP1 |
| TTH-11 | Spike de calibración de hiperparámetros temporales del modelo predictivo | TTH derivada (no asociada a feature) | — | MVP1 |

**Total Bloque E: 0 HUs operativas + 5 TTH nuevas.**

La numeración de HUs del backlog no avanza con el Bloque E: la última HU operativa cerrada es HU-15 (Bloque D). La próxima HU operativa será HU-16 en el Bloque F (Gerente) o en sesión MVP2 dedicada.

---

## Cambios aplicados a documentos previos como consecuencia del Bloque E

Durante la redacción del Bloque E, las siguientes modificaciones se aplicaron a documentos previos:

1. **DHU-015 agregada a `DECISIONS_HU.md`**: clasificación HU/TTH de F32, F33, F34, F35 (4 → 5 TTH tras incorporación de TTH-11 durante la redacción).

2. **TTH-07, TTH-08, TTH-09, TTH-10, TTH-11 agregadas a `TAREAS_TECNICAS_HABILITADORAS.md`**: cinco TTH nuevas, índice actualizado, referencias cruzadas con TTH-04 actualizadas (Nivel 1 ante caída de TTH-08, Nivel 2 invoca RandomForest preservado por TTH-09, Nivel 3 invoca TTH-05 cuando TTH-10 cae).

3. **Fichas F32, F33, F34, F35 en `FEATURE_BACKLOG_DETALLADO.md`** actualizan su columna "Modelado" para apuntar a las TTH respectivas (estaban como "A determinar al redactar el Bloque E"). F32 además actualiza estado a "🆕 Por construir desde cero, exploración previa no entra como entregable conforme a TTH-07 estado actual".

4. **`EVOLUCION_TESIS.md` Fase 3 actualizada**: la descripción del motor adaptativo se ajusta para reflejar la arquitectura real (2 estrategias adaptativas + 1 capa de reglas duras), no "3 estrategias de control". Ajuste de coherencia documental tras clarificación arquitectónica registrada durante la redacción de TTH-10.

5. **"Próximos pasos" actualizados en `HU_BLOQUE_A.md`, `HU_BLOQUE_B.md`, `HU_BLOQUE_C.md`, `HU_BLOQUE_D.md`**: Bloque E cerrado, restan Bloque F y MVP2.

6. **`LEAN_INCEPTION_CEREBROVIAL.md` documentos relacionados**: referencia a `HU_BLOQUE_E.md` agregada.

---

## Entregables pendientes asociados al Bloque E

Estos entregables están declarados en las TTH del Bloque E pero no se producen en esta sesión:

1. **`documentation/docs/INVESTIGACION_HIPERPARAMETROS_TEMPORALES.md`** (entregable de TTH-11): documento de sustentación bibliográfica y empírica de hiperparámetros temporales del modelo predictivo. CT-11.1 a CT-11.8 declaran su contenido esperado.

2. **Topología SUMO genérica de 4 accesos + escenarios de demanda + script de generación de dataset** (entregable de TTH-07).

3. **Modelos GRU univariados entrenados** (entregable de TTH-09): cuatro modelos persistidos en disco con métricas reportadas.

4. **Refactor del módulo de visión** (entregable de TTH-08) con detección + tracking + asignación direccional por polígonos ROI + persistencia + endpoints expuestos.

5. **Integración funcional del motor adaptativo con TTH-09, TTH-07 y TTH-04** (entregable de TTH-10) con persistencia ampliada de decisiones.

---

## Decisiones que aplicaron a este bloque

Durante la redacción del Bloque E se cerró la siguiente decisión formal (en `DECISIONS_HU.md`):

- **DHU-015:** clasificación HU/TTH de las features del Bloque E (F32, F33, F34, F35). Resultado: 5 TTH operativas (TTH-07 a TTH-11), 0 HUs operativas. La numeración del backlog de HUs no avanza con el Bloque E.

---

## Próximos pasos

Esta sesión cerró el Bloque E. A la fecha actual, el Bloque F del MVP1 también está cerrado (el 2026-05-16 por DHU-016, con 2 HUs operativas HU-16 y HU-17) y el MVP2 está cerrado (el 2026-05-16 por DHU-017, con 4 HUs operativas adicionales HU-18 a HU-21 + HU-09 ya anticipada en el Bloque B). **Con el cierre del MVP2, la redacción del Product Backlog del proyecto queda completa en su componente funcional: 21 HUs operativas (HU-01 a HU-21) y 11 TTH (TTH-01 a TTH-11).** Los siguientes pasos del proyecto, fuera del alcance del Product Backlog funcional, en sesiones futuras:

1. **Documento de Requisitos Funcionales y No Funcionales (RF/RNF).** Pendiente desde DHU-007: consolidar los "Candidatos a RNF" de todas las HUs (HU-01 a HU-21) en un documento único aprobado, numerando cada RNF y reemplazando los umbrales hardcodeados en las HUs por referencias al documento formal.

2. **Ceremonias de estimación (Planning Poker) y priorización (MoSCoW)** sobre el backlog completo (MVP1 + MVP2 + RF/RNF).

3. **Implementación SCRUM del MVP1**, conforme al cronograma del Bloque 7 del MVP Canvas del Inception. El MVP2 (5 HUs adicionales) entra al sprint si hay holgura de cronograma, conforme a la semántica refinada por DHU-012.

4. **SDD (Software Design Document)**, siguiente entregable académico mayor del proyecto.

---

## Documentos relacionados

- `HU_BLOQUE_A.md` — Bloque A del Product Backlog (acceso al sistema, 1 HU).
- `HU_BLOQUE_B.md` — Bloque B del Product Backlog (8 HUs: HU-02 a HU-09).
- `HU_BLOQUE_C.md` — Bloque C del Product Backlog (3 HUs operativas: HU-10, HU-11, HU-12).
- `HU_BLOQUE_D.md` — Bloque D del Product Backlog (3 HUs operativas: HU-13, HU-14, HU-15).
- `HU_BLOQUE_F.md` — Bloque F del Product Backlog (2 HUs operativas: HU-16, HU-17; F30 inglobada como CAs).
- `HU_MVP2.md` — MVP2 del Product Backlog (HU-18, HU-19, HU-20, HU-21; HU-09 reside en `HU_BLOQUE_B.md`). HU-20 extiende CT-09.5 de TTH-09 inglobando persistencia paralela del modelo de respaldo, sin TTH nueva.
- `DECISIONS_HU.md` — Decisiones metodológicas sobre HUs (DHU-001 a DHU-019).
- `DECISIONS.md` — Decisiones técnicas del producto (D-001 a D-009). Particularmente D-006, D-007, D-008, D-009 fundamentan el Bloque E.
- `TAREAS_TECNICAS_HABILITADORAS.md` — TTH-01 a TTH-11 (incluye las 5 TTH del Bloque E).
- `LEAN_INCEPTION_CEREBROVIAL.md` — Inception completo aplicado al proyecto.
- `FEATURE_BACKLOG_DETALLADO.md` — Detalle completo de las 41 features identificadas (29 MVP1 + 5 MVP2 + 7 Trabajos Futuros).
- `EVOLUCION_TESIS.md` — Narrativa de las 4 fases del proyecto.
- `CONTROL.md` — Sustentación teórica del motor adaptativo (consumido por TTH-10).
- `documentation/docs/INVESTIGACION_HIPERPARAMETROS_TEMPORALES.md` — Documento pendiente entregable de TTH-11.
