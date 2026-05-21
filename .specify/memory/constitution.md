# CerebroVial Constitution

> **Adopción brownfield (DHU-021).** Esta constitución **mapea** las decisiones ya tomadas del
> proyecto al formato de Spec Kit; no se genera con `/speckit-constitution`. Cada artículo destila
> un principio en lenguaje normativo y **enlaza su fuente por ID y archivo**; NO reproduce la
> justificación, el impacto ni el texto sugerido para la tesis, que viven en el documento fuente.
> Las fuentes canónicas son `documentation/docs/DECISIONS.md` (decisiones técnicas, `D-xxx`) y
> `documentation/lean-inception/4-decisiones/DECISIONS_HU.md` (decisiones metodológicas, `DHU-xxx`).
> El `SDD_CEREBROVIAL.md` sigue siendo la fuente canónica de arquitectura.

## Core Principles

Los principios rectores se organizan en dos Títulos: el **Título I** deriva de las decisiones
técnicas del producto (D-001…D-009); el **Título II**, de las decisiones metodológicas del backlog
(DHU-001…DHU-022). Cada artículo es vinculante para las HU, TTH, RF/RNF y artefactos derivados.

## Título I — Principios técnicos del producto

### Artículo 1 — Arquitectura monolito modular
El sistema es un monolito modular, no microservicios; las carpetas `core_management_api/`,
`edge_device/`, `ia_prediction_service/` y `frontend_ui/` son módulos de un mismo sistema, y la base
común se consolida en `shared/` como paquete pip instalable. → Fuente: D-001 (DECISIONS.md)

### Artículo 2 — Modelo predictivo RNN → GRU univariado por intersección
El predictor es un GRU univariado por intersección, alineado al documento de tesis; el RandomForest
queda como respaldo temporal conmutado por flag de configuración hasta que el GRU esté servido. La
dependencia espacial entre intersecciones (arquitecturas tipo STGNN) es trabajo futuro. → Fuente:
D-002, D-006 (DECISIONS.md; D-006 refina D-002 y resuelve D-PENDING-001)

### Artículo 3 — Despliegue local con contenedores
El sistema se despliega localmente con `docker compose`; no se usa cloud en el ciclo académico. La
desplegabilidad en otros entornos se demuestra arquitectónicamente y se documenta como plan de
productivización. → Fuente: D-003 (DECISIONS.md)

### Artículo 4 — Pi física como demostración conceptual
No se entrega hardware en la defensa; se demuestra que la arquitectura es desplegable en Raspberry Pi
por la separación de `edge_device` con dependencias mínimas y comunicación por SSE/HTTP. *(Sujeta a
confirmación con asesor.)* → Fuente: D-004 (DECISIONS.md)

### Artículo 5 — Integridad de los números de tesis
Los números declarados en el documento de tesis se actualizan a los valores reales medidos en la
validación; si la realidad es peor, se reporta la realidad. *(Sujeta a confirmación con asesor.)*
→ Fuente: D-005 (DECISIONS.md)

### Artículo 6 — Visión como componente demostrable
El módulo de visión es un sensor de estado en tiempo real con validación independiente (métricas de
detección: precisión, recall, mAP sobre dataset etiquetado); NO participa en el loop de validación
cuantitativa del sistema integrado, donde las métricas de estado las provee SUMO. → Fuente: D-007
(DECISIONS.md)

### Artículo 7 — SUMO como columna vertebral de datos
SUMO genera el dataset de entrenamiento del modelo y los escenarios de validación cuantitativa, con
particiones independientes (distintos seeds y patrones de demanda) para evitar fuga de información;
los datos reales de tráfico de Lima son trabajo futuro. → Fuente: D-008 (DECISIONS.md)

### Artículo 8 — Variable de estado: jam level ordinal 0-5
La variable de estado y objetivo del modelo es el nivel de congestión en escala ordinal 0-5
(constructo derivado de Waze); este desacople de la fuente permite intercambiar visión, SUMO o Waze
sin reentrenar ni alterar predicción y control. → Fuente: D-009 (DECISIONS.md)

> **Nota.** D-PENDING-001 no genera artículo: quedó resuelta por D-006 y se conserva solo como traza
> histórica en `DECISIONS.md`.

## Título II — Principios metodológicos del backlog

### Artículo 9 — El login no es una HU
La autenticación se modela como Tarea Técnica Habilitadora (TTH-01), no como Historia de Usuario; los
requisitos de autenticación que afectan a HU operativas se ingloban como criterios de aceptación de
esas HU. → Fuente: DHU-001 (DECISIONS_HU.md)

### Artículo 10 — Valor cognitivo del acceso por rol
El acceso diferenciado por rol se justifica por reducción de carga cognitiva (cada Persona se
concentra en su contexto de trabajo), no por segregación defensiva de permisos. → Fuente: DHU-002
(DECISIONS_HU.md)

### Artículo 11 — Sujetos válidos en HU
Solo las tres Personas del producto (Operador de Tráfico Municipal, Gerente de Tránsito Municipal,
Administrador del Sistema), o su enumeración explícita, son sujetos válidos en una HU; "el sistema",
"Equipo de Desarrollo" y "Usuario" genérico no lo son. → Fuente: DHU-003 (DECISIONS_HU.md)

### Artículo 12 — TTH como categoría separada del Product Backlog
El trabajo técnico habilitador se documenta como TTH (enunciado imperativo + criterios técnicos de
"terminado", sin Given-When-Then) en `TAREAS_TECNICAS_HABILITADORAS.md`, separado del Product Backlog
de HU y priorizado con criterios distintos. → Fuente: DHU-004 (DECISIONS_HU.md)

### Artículo 13 — Robustez ante interrupción de fuente
Toda HU operativa marca pasivamente su propio panel ante caída de su fuente: Caso A (fuente externa
de medición → último valor "desactualizado" con antigüedad) y Caso B (componente interno de decisión
→ última decisión "no confirmada" con antigüedad). La marca pasiva es responsabilidad de cada vista;
la alerta activa transversal es del Bloque C. → Fuente: DHU-005 (DECISIONS_HU.md)

### Artículo 14 — HU agnósticas a la implementación
Las HU describen el qué observable, no el cómo; no nombran tecnologías, componentes ni constructos
técnicos (visión, SUMO, GRU, Webster/MaxPressure/MTC, Waze). Única excepción: la escala ordinal 0-5,
ya autónoma del sistema. → Fuente: DHU-006 (DECISIONS_HU.md)

### Artículo 15 — RNF declarados como tales
Los requisitos no funcionales se declaran en documento propio bajo la taxonomía ISO/IEC 25010:2023
(9 características); las HU solo los referencian por ID `RNF-XXX-NN`. La metodología de redacción del
catálogo RF/RNF (plantillas, derivación de RF por composición, política aditiva) está fijada y es
vinculante. → Fuente: DHU-007, DHU-019 (DECISIONS_HU.md)

### Artículo 16 — Componente caído vs. modo degradado vs. lógica de fallback
Se distinguen arquitectónicamente tres conceptos: componente caído (hecho técnico binario), modo
degradado del sistema (estado operativo, p. ej. "degradado nivel 3") y lógica de fallback en cascada
(mecanismo interno, modelado como TTH). → Fuente: DHU-008 (DECISIONS_HU.md)

### Artículo 17 — Marca pasiva (Bloque B) vs. alerta activa (Bloque C)
La marca pasiva contextual de cada panel (Bloque B) y la alerta activa transversal del estado del
sistema completo (Bloque C, HU-10) son responsabilidades complementarias, no duplicadas; el trabajo
del Bloque C que no tiene Persona beneficiaria directa se clasifica como TTH. → Fuente: DHU-009,
DHU-010 (DECISIONS_HU.md)

### Artículo 18 — Decisiones de redacción por bloque
La clasificación HU/TTH y la redacción de cada bloque del backlog (C/D/E/F y MVP2) se rigen por sus
actas de redacción registradas en `DECISIONS_HU.md`; estas gobiernan la consistencia del backlog
(numeración, inglobamientos, ampliación de TTH, sustratos) y no se reproducen aquí. → Fuente:
DHU-011, DHU-013, DHU-014, DHU-015, DHU-016, DHU-017 (DECISIONS_HU.md)

### Artículo 19 — Coherencia documental y patrones transversales
La coherencia del corpus —semántica de MVP, eliminación de MVP3, conteos, alineación de vocabulario y
el patrón "Resumen ejecutivo" al inicio de cada HU— se mantiene por las decisiones de auditoría
documental. → Fuente: DHU-012, DHU-018 (DECISIONS_HU.md)

### Artículo 20 — Alineación especificación ↔ código
Cuando el código diverge de la especificación, prevalece la semántica especificada y el código se
alinea a ella (caso ControlView / HU-05, Delta-08); los cambios estructurales necesarios para esa
alineación (p. ej. persistencia del estado vigente del motor) se autorizan deliberadamente conforme a
la guardia de `CLAUDE.md`. → Fuente: DHU-020 (DECISIONS_HU.md)

### Artículo 21 — El SDD como fuente canónica de arquitectura
El `SDD_CEREBROVIAL.md`, verificado contra el repositorio vivo, es la fuente canónica de la
arquitectura y se redacta según las decisiones metodológicas registradas; la integración Gemini queda
fuera de la arquitectura objetivo (su remoción es saneamiento diferido). → Fuente: DHU-021
(DECISIONS_HU.md)

### Artículo 22 — Nomenclatura de roles
`operator`, `manager` y `admin` son los claims técnicos canónicos de rol; el frontend muestra labels
en español mapeadas a esos claims (cierre de Delta-02). → Fuente: DHU-022 (DECISIONS_HU.md)

## Governance

Esta constitución consolida y enlaza las decisiones rectoras del proyecto; **no las reemplaza**. La
autoridad sustantiva reside en los documentos fuente:

- Las decisiones técnicas (`D-xxx`) viven en `documentation/docs/DECISIONS.md`; las metodológicas
  (`DHU-xxx`) en `documentation/lean-inception/4-decisiones/DECISIONS_HU.md`. La arquitectura es
  canónica en `documentation/sdd/SDD_CEREBROVIAL.md`.
- **Enmiendas:** toda modificación de un principio se hace primero como nueva entrada o refinamiento
  `D-xxx` / `DHU-xxx` en su documento fuente; luego se refleja aquí actualizando el artículo
  correspondiente, su cita y el pie de versión. Los artículos no se editan sin una decisión fuente
  que los respalde.
- **Cumplimiento:** `spec.md` y `tasks.md` referencian estos principios por número de artículo; los
  cambios estructurales (mover carpetas, renombrar paquetes, cambiar el modelo de BD) se detienen y
  consultan según la guardia de `CLAUDE.md` (Artículo 20).
- **Trazabilidad del mapeo:** la correspondencia artefacto Spec Kit ↔ documento fuente se mantiene en
  `documentation/sdd/SPECKIT_MAPPING.md`.

> **Cobertura de las 22 DHU.** El Título II tiene 14 artículos (Arts. 9–22) que cubren las 22 DHU por
> fusión y agrupación: el Art. 15 cubre DHU-007 + DHU-019; el Art. 17, DHU-009 + DHU-010; el Art. 18
> agrupa DHU-011/013/014/015/016/017; el Art. 19, DHU-012 + DHU-018. Las restantes son 1:1. El Título
> I tiene 8 artículos (D-002 y D-006 fusionados en el Art. 2).

**Version**: 1.0.0 | **Ratified**: 2026-05-20 | **Last Amended**: 2026-05-20
