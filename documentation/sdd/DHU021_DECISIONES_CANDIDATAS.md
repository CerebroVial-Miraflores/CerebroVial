# Decisiones candidatas a DHU-021 — registro completo

> **Propósito:** preservar el contenido completo de las decisiones metodológicas
> de redacción del SDD que se acumularon durante la sesión de diseño, para
> consolidarlas en la DHU-021 al cerrar el SDD. NO se redacta DHU-021 todavía
> (se acordó hacerlo al final). Este archivo es el insumo de esa redacción.
>
> Formato de cada entrada: decisión + justificación + (si aplica) alternativas
> descartadas, reproduciendo el razonamiento acordado en sesión.

---

## #1 — Conciliación As-designed / matriz rica confinada a §10

**Decisión:** El SDD adopta postura **As-designed** en el cuerpo (§1–§9): describe
la arquitectura objetivo sin teñir la prosa con el estado de construcción. La
trazabilidad es una matriz bidireccional rica (HU/TTH ↔ componente ↔ estado ↔
delta) **confinada a §10**, y la confrontación con el ~25% construido vive en §11.

**Justificación:** Honra simultáneamente las tres elecciones del usuario (As-designed
+ matriz bidireccional con estado/delta + DHU-021 al final) sin que ninguna pise a
la otra. Es además la forma de **menor trabajo y menor confusión para Claude Code**:
una descripción de diseño limpia + una tabla de punto-de-partida, en lugar de dos
relatos paralelos (as-built vs as-designed) que habría que reconciliar. La tensión
detectada era que "estado" y "delta" en la matriz SON la auditoría, lo que choca con
As-designed puro; se resuelve confinando ese material a §10/§11 en vez de derramarlo
en la prosa de cada componente.

**Alternativas descartadas:** (a) matriz sin columnas estado/delta — contradice la
elección explícita del usuario; (b) As-built/As-designed etiquetado en el cuerpo —
el usuario lo rechazó y produce dos fuentes a reconciliar.

---

## #2 — Formato Markdown, construcción incremental y convenciones de cita

**Decisión:** El SDD se redacta como un único archivo Markdown construido sección
por sección, con verificación de coherencia antes de avanzar (mismo patrón que los
bloques previos del proyecto). Convenciones de cita: decisiones técnicas `D-00N`,
metodológicas `DHU-0NN`, historias `HU-NN`, tareas técnicas `TTH-NN`, requisitos
`RF-0NN` / `RNF-XXX-NN`, criterios `CA-NN.N` (HU) y `CT-NN.N` (TTH), deltas
`Delta-NN`. Nombres de archivos/módulos/rutas en `estilo de código`.

**Justificación:** Trazabilidad uniforme con el resto del corpus documental y
facilidad de versionado/cita en la tesis.

---

## #3 — Proceso Spec Kit + estructura interna híbrida 4+1 / ISO 25010 / ADR

**Decisión:** Proceso de trabajo: se adopta **GitHub Spec Kit** como toolkit de
Spec-Driven Development. Estructura interna del artefacto de diseño (la fase Plan
de Spec Kit): **híbrido 4+1 de Kruchten (vistas) + ISO/IEC 25010:2023 (atributos
de calidad) + ADR ligero (decisiones)**.

**Justificación:** Spec Kit y un marco arquitectónico operan en capas distintas y
complementarias: Spec Kit gobierna el *proceso* (Spec→Plan→Tasks→Implement); el
híbrido gobierna la *estructura interna* del documento. Spec Kit es deliberadamente
agnóstico al marco arquitectónico interno (su meta declarada es ser independiente
de tecnologías/frameworks), así que 4+1 llena ese hueco. ISO 25010 da coherencia
con DHU-019 (que ya lo adoptó para los RNF); ADR da coherencia con `DECISIONS.md`.
El SDD corresponde al artefacto `plan.md` + `data-model.md` de Spec Kit.

**Alternativas descartadas:** 4+1 puro (sin hogar para calidad/decisiones); arc42
(reorganiza el índice ya aprobado, menos citado en academia); ISO/IEC/IEEE 42010
(maquinaria de viewpoints/concerns sobredimensionada para proyecto individual).

---

## #4 — Adopción brownfield de Spec Kit (mapear, no regenerar)

**Decisión:** Spec Kit se adopta en modo **Iterative Enhancement / Brownfield** (no
greenfield), instalado en el repo existente vía `specify init --here`. El contenido
ya curado (HU/TTH/RF/RNF/DHU) se **mapea** a las plantillas de Spec Kit (`spec.md`,
`plan.md`, `data-model.md`) en lugar de **regenerarse** con `/speckit-specify`,
preservando la trazabilidad fina y las 20 DHU. El SDD redactado corresponde a
`plan.md` + `data-model.md`; su estructura interna sigue el híbrido 4+1.

**Justificación:** El proyecto está en Sprint 4 con ~25% construido y backlog
formal cerrado; no es zero-to-one. Regenerar specs desde cero arriesgaría perder
matices de las 20 DHU y la trazabilidad. Spec Kit contempla el modo brownfield
explícitamente. Spec Kit aporta su andamiaje (CLI, plantillas, comandos); el
contenido proviene del corpus.

---

## #5 — `ARCHITECTURE_TARGET.md` archivado en legacy, no citado

**Decisión:** El archivo `documentation/docs/ARCHITECTURE_TARGET.md` se mueve a
`documentation/docs/legacy/` con nombre que refleje su naturaleza histórica, y
**no se cita ni se menciona en el SDD**. El SDD nuevo es la única arquitectura
objetivo vigente.

**Justificación:** `ARCHITECTURE_TARGET.md` es una versión pre-Lean-Inception (su
título interno es de hecho `# CLAUDE.md`) que describe una arquitectura ya
descartada: híbrido Edge-Cloud con **Azure**, **microservicios** (API Gateway,
servicios de predicción/control/datos separados), **YOLOv8** (el repo usa YOLO11n),
**motor de reglas** if-then (el real es Webster/MaxPressure/MTC), **MongoDB +
Grafana + Blob Storage** (no presentes), y números de validación reportados como
obtenidos (D-005 dice que se actualizan tras validación real). Contradice D-001
(monolito), D-003 (Docker local, sin Azure), TTH-10 y la auditoría. El usuario optó
por cortarlo limpio en lugar de citarlo; la narrativa de evolución, si se requiere,
vive en `EVOLUCION_TESIS.md`.

**Acción asociada:** corregir la fila de `plan.md` en `SPECKIT_MAPPING.md` (hoy aún
apunta a `ARCHITECTURE_TARGET.md`) para que apunte solo al SDD nuevo. Verificar al
archivar que no se confunda con el `CLAUDE.md` operativo vivo (comparten título).

---

## #6 — Colisión de identificadores D en el modelo de datos

**Decisión:** El SDD adopta la numeración del `DECISIONS.md` canónico. Las
decisiones del `DATA_MODEL_AUDIT.md` viejo se citan por contenido y fecha, no por
su ID-D, para evitar ambigüedad.

**Justificación:** `DATA_MODEL_AUDIT.md` (2026-05-03) define D-006/D-007/D-008 sobre
persistencia de visión, tablas vacías y dataset sintético. El `DECISIONS.md`
canónico (2026-05-11) usa D-006/D-007/D-008 para GRU univariado, visión-fuera-de-loop
y SUMO. Son decisiones distintas con identificadores colisionantes (las distingue
solo la fecha). El SDD no puede usar ambas series con el mismo ID sin confundir al
lector. (Pendiente menor de higiene documental: renumerar las del audit o marcarlas
explícitamente como serie distinta.)

---

## #7 — Notación C4 reservada al informe; SDD no la usa internamente

**Decisión:** El SDD usa el formato del híbrido 4+1 (prosa + tablas + diagramas
propios), **sin notación C4**. C4 se reserva para el informe de tesis / sustentación,
donde el usuario lo usará para documentar.

**Justificación:** Decisión del usuario. 4+1 es un marco de *vistas*; C4 es una
*notación de diagramas* — se pueden combinar, pero el usuario prefirió mantenerlos
separados por documento (SDD en 4+1, informe en C4). El viejo `ARCHITECTURE_TARGET.md`
usaba C4 con una descomposición microservicios desactualizada que NO debe reutilizarse.

---

## #8 — SDD como fuente canónica de componentes; el C4 del informe deriva de él

**Decisión:** El SDD (marco 4+1) es la **fuente de verdad** sobre la descomposición
en componentes: qué existe, nombres canónicos, responsabilidades. Si el informe de
tesis usa C4 para los mismos elementos, debe **derivar del SDD** (mismos nombres,
misma descomposición), tratando C4 como una vista visual del modelo del SDD, no como
un modelo independiente.

**Justificación:** Evita divergencia entre el SDD y el informe (el riesgo de tener
dos representaciones de la misma arquitectura que se desincronizan). Coherente con
la política de fuente única que la sesión ya aplicó a `DECISIONS.md` y a
`ARCHITECTURE_TARGET.md`. Salvaguarda que el usuario pidió explícitamente vigilar.

---

## #9 — Profundidad de §3 a dos niveles; detalle DDD a la vista de desarrollo

**Decisión:** La vista de componentes (§3) se descompone en dos niveles:
contenedores (5 servicios) y componentes internos del `core_management_api`
(prediction/control/vision-consumer). La estructura DDD interna de cada módulo
(`domain/application/infrastructure/presentation`, evidenciada por la auditoría)
se documenta en la **vista de desarrollo** (sección de organización del código),
no en §3.

**Justificación:** Dos niveles es el punto justo para una vista de componentes de
tesis: muestra el sistema y abre la caja donde está la complejidad (el core con el
motor adaptativo) sin perderse en detalle de implementación. Bajar a la estructura
DDD en §3 convertiría la vista arquitectónica en un mapa de carpetas; ese detalle
encaja mejor en la vista de desarrollo del 4+1.

---

## Notas para la redacción de DHU-021 (al cerrar el SDD)

- Estas 9 se consolidan en **una sola DHU-021** siguiendo el patrón de decisiones
  consolidadas del proyecto (como DHU-014, DHU-016, DHU-017, DHU-019).
- DHU-021 registra las decisiones metodológicas de redacción del **propio SDD**
  (meta-decisiones), análogo a cómo DHU-019 registró las del documento RF/RNF.
- El identificador DHU-021 estaba libre al cierre de la sesión previa (última
  cerrada: DHU-020). Verificar que siga libre antes de asignarlo.
- Es probable que durante la redacción de §4–§12 surjan decisiones adicionales
  (p. ej. mecanismo realtime SSE vs polling en §5; diseño concreto de la persistencia
  de estado vigente en §4; tratamiento del playground huérfano de DHU-020 §C;
  tratamiento de features huérfanas Delta-13). Agregarlas a esta lista conforme
  aparezcan, para que DHU-021 las incluya.
