# TTH-07 — Cierre de Fase 0 (kickoff + smoke toolchain S0)

**Rama**: `feature/tth-07-fase0-docs` (desde `master@85d56bb4` = merge
PR #34, cierre TTH-08 F9). Consistente con el patrón TTH-08 (una rama
feature por fase).
**Fecha de cierre**: 2026-05-29.
**Estado al cierre**: **F0 cerrada con un blocker abierto** —
`libsumo==1.26.0` no existe en PyPI; el pin lock necesita decisión del
usuario antes de F4.
**Restricciones honradas**: cero código productivo, cero archivos en
`simulation/` (módulo aún no existe), cero modificación de `.gitignore`,
cero modificación del `.venv/` raíz. Único cambio versionado: este
handoff.

---

## 1. Lo que Fase 0 entregó

Alcance de F0: audit read-only de insumos (D-009, regla anti-torch,
recon top-level) + smoke toolchain S0 desechable + acuerdo del scaffold
`simulation/` para F1. Entregables:

| # | Entregable | Estado |
|---|------------|--------|
| 0a | Auditoría D-009 (mapeo SUMO → jam level 0-5) ubicada y citada | ✅ |
| 0b | Recon top-level: cero colisión con módulo `simulation/`; regla anti-torch (CLAUDE.md) confirma módulo SUMO **fuera** del core | ✅ |
| 0c | Venv aislado en `scratch/sumo_s0/.venv-sumo/` con Python 3.11.15 | ✅ |
| 0d | `eclipse-sumo==1.26.0` instalado (pin exacto, escenario A wheel autocontenido) | ✅ |
| 0e | `traci==1.26.0` instalado (pin exacto) | ✅ |
| 0f | `libsumo==1.26.0` instalado | ❌ **Blocker** — versión no existe en PyPI |
| 0g | SUMO_HOME reproducible resuelto al wheel | ✅ |
| 0h | Smoke `sumo` headless (60 pasos sobre net 2×2 grid con TLS) | ✅ |
| 0i | Smoke `sumo-gui` (60 pasos, `--start --quit-on-end`) | ✅ |
| 0j | Smoke TraCI cross-process (start → 30 steps → `getLastStepMeanSpeed`/`getMaxSpeed` → close) | ✅ |
| 0k | Smoke libsumo in-process | ❌ **Abort-documented** — depende de 0f |
| 0l | Flag Parquet confirmado (`--summary-output FILE.parquet`, `--tripinfo-output FILE.parquet`, `--output.format parquet`) | ✅ |
| 0m | Scaffold `simulation/` propuesto y acordado (materializa F1, no F0) | ✅ |
| 0n | Este handoff | ✅ |

---

## 2. Hallazgos del smoke S0 (relevantes para F1+)

### 2.1 Wheel `eclipse-sumo==1.26.0` para macOS arm64 — Escenario A (autocontenido)

Inspección del wheel instalado en
`scratch/sumo_s0/.venv-sumo/lib/python3.11/site-packages/sumo/`:

```
sumo/
├── bin/           ← 14 binarios: sumo, sumo-gui, netgenerate,
│                    netconvert, netedit, duarouter, jtrrouter,
│                    marouter, dfrouter, polyconvert, od2trips,
│                    activitygen, emissionsDrivingCycle, emissionsMap
├── tools/         ← 66 entries incluyendo randomTrips.py, sumolib/,
│                    traci/, osmGet.py, osmWebWizard.py
├── data/          ← templates y assets (3D, emissions, font, lang, ...)
├── lib/           ← libsumocpp.dylib + libtracicpp.dylib (C++ runtime,
│                    sin Python bindings)
├── include/       ← headers libsumo.h (para compilar contra)
└── cmake/         ← config para builds externos
```

**Refuta dos riesgos del plan F0**:
- "Wheel mac arm64 puede no traer `sumo-gui`" → **Falso en 1.26.0**: el
  wheel incluye `bin/sumo-gui` y se ejecuta sin Qt instalado aparte. Build
  features: `... arm64 AppleClang 15.0.0 Release FMI Proj GUI FMT Intl
  SWIG Parquet GDAL GL2PS`. La flag `GUI` confirma soporte compilado.
- "`$SUMO_HOME/tools/` puede no existir" → **Falso en el wheel**: `tools/`
  está bundleado con 66 entries incluyendo el `randomTrips.py` necesario
  para CT-07.2 (patrones) y los wrappers `osm*.py` para futura
  calibración Miraflores. El framework `/Library/Frameworks/EclipseSUMO`
  instalado a nivel sistema **sí** carece de `tools/` (verificado en
  Parte A), pero como adoptamos el wheel, el framework system-wide ya
  no es necesario y queda como instalación independiente sin rol en
  CerebroVial.

**Consecuencia para F1**: `simulation/` adopta el wheel del venv como
**única fuente de SUMO**. El framework system-wide no participa.
`SUMO_HOME` apunta al wheel.

### 2.2 `libsumo==1.26.0` no existe en PyPI — **blocker abierto**

Versiones de `libsumo` disponibles en PyPI al 2026-05-29: 1.18.0, 1.19.0,
1.20.0, 1.21.0, **1.27.0**. **Hay un gap de 1.22 a 1.26**. El pin lock
del usuario (`==1.26.0 exacto en los tres`) no se puede honrar para
`libsumo`.

Detalle del intento: `pip install libsumo==1.27.0` (única próxima
disponible) instaló libsumo + sumo-data + traci, **degradando traci de
1.26.0 a 1.27.0** porque `libsumo==1.27.0` depende de `traci==1.27.0`.
Esto rompe la pin lock en dos ejes. Se revirtió a estado pinneado
(`pip uninstall libsumo sumo-data`, `pip install --force-reinstall
traci==1.26.0`) por instrucción explícita del usuario:

> *"User explicitly pinned libsumo==1.26.0 exact; agent is installing
> libsumo==1.27.0, violating the explicit version boundary — should
> abort-and-document instead."*

**Estado del venv al cierre de F0** (`pip freeze | grep -i -E "(sumo|traci)"`):

```
eclipse-sumo==1.26.0   ← pin honrado exacto
traci==1.26.0          ← pin honrado exacto
sumolib==1.27.0        ← transitivo (traci==1.26.0 requiere sumolib>=1.26.0;
                         pip resuelve al latest disponible, que es 1.27.0
                         dado el gap 1.22-1.26 también en sumolib)
# libsumo: NO instalado
```

**Pin `sumolib==1.27.0`** es relax transitivo, no decisión activa.
`sumolib` es lectura/escritura de XML SUMO (utility no-server). API
estable entre minor versions. Riesgo bajo. Documentado para que F1 lo
visibilice en `simulation/pyproject.toml`.

**Decisión que requiere el usuario antes de F4**:

| Opción | Implicancia | Costo |
|--------|-------------|-------|
| **A) Relajar pin libsumo a `==1.27.0`** (con `eclipse-sumo` y `traci` siguiendo en 1.26 — incompatible con la cadena pip que upgrade traci a 1.27 también) | Romper pin lock en libsumo + traci + sumolib (todos suben a 1.27). En la práctica equivale a **subir SUMO a 1.27** | Cambia versión productiva |
| **B) Subir todo a `==1.27.0`** (`eclipse-sumo==1.27.0 traci==1.27.0 libsumo==1.27.0`) | Pin consistente, todas las versiones disponibles en PyPI. SUMO 1.27 es backwards-compat con 1.26 en TraCI protocol | Cambio de versión declarada para TTH-07 |
| **C) Dropear libsumo del toolchain** | F4 (adaptador TraCI ↔ motor) usa solo TraCI cross-process (ya smokeado verde). F3 (CT-07.3 dataset) usa salida nativa Parquet de SUMO (`--summary-output`/`--tripinfo-output`/`<edgeData>`, no stepping en Python — confirmado en S0.15). libsumo solo es un fallback in-process **si throughput por paso molesta**, lo cual no se conoce ex-ante | Cero costo si TraCI alcanza; revisión condicional en F4 |
| **D) Construir libsumo 1.26 desde source** | Reproducible en macOS arm64 pero alto esfuerzo + dependencia en CMake/SWIG/Qt5 | Alto, fuera de scope F0-F1 |

**Recomendación del agente**: **opción C** (dropear libsumo del
toolchain de TTH-07). Razones: (i) TraCI smoke verde demuestra que el
transporte cross-process funciona; (ii) F3 no necesita libsumo porque la
salida Parquet nativa de SUMO ya está confirmada (S0.15); (iii) F4 es un
lazo de control de bajo throughput (≤1 Hz típico) donde la latencia
in-process de libsumo no aporta; (iv) mantiene el pin lock `==1.26.0`
estricto en los dos paquetes que importan (`eclipse-sumo`, `traci`); (v)
si F4 descubre throughput insuficiente, libsumo se puede revisitar como
follow-up con la opción B (subir a 1.27).

**Implicancia para `simulation/pyproject.toml` (F1)**: declarar
`eclipse-sumo==1.26.0`, `traci==1.26.0`, y **no** listar libsumo. Si la
decisión usuario es C, el handoff F1 cita esta nota.

### 2.3 `sumolib==1.27.0` (transitivo)

`traci==1.26.0` requiere `sumolib>=1.26.0`. `sumolib==1.26.0` tampoco
existe en PyPI (mismo gap). pip resuelve a `1.27.0`. **No es una
decisión activa del usuario**, es un side-effect del gap PyPI. F1 lo
declara explícitamente en `pyproject.toml` para que sea visible y
versionado, no para volverlo a "pinear" exacto a 1.26 (imposible).

### 2.4 Smoke TraCI — evidencia para CT-07.5 y D-009

Script `scratch/sumo_s0/s0_traci.py` corrió end-to-end:

```
transport=traci edge=A0A1 lane=A0A1_0 mean_speed_mps=9.953 max_speed_mps=13.890
```

Aplicando el mapeo D-009 (`scratch/sumo_s0/s0_traci.py` lee
`getLastStepMeanSpeed` + `getMaxSpeed` exactamente como prescribe
`documentation/lean-inception/4-decisiones/DECISIONS.md:211-225`):

- ratio = 9.953 / 13.890 = **0.717** ⇒ `jam_level = 1 (Bajo)` ([D-009
  línea 193](../../lean-inception/4-decisiones/DECISIONS.md#L193)).
- `max_speed_mps = 13.890 ≈ 50 km/h` — default urbano de `netgenerate`,
  consistente con el parámetro Miraflores que F1 cableará explícitamente
  en `simulation/conf/network_params.yaml`.

**Implicancia para F1**: el smoke confirma que el pipeline
`netgenerate → randomTrips → duarouter → sumo + TraCI` produce
velocidades consistentes con el mapeo D-009 sin instrumentación
adicional. F1 puede arrancar la topología 4-vías directamente sobre
este pipeline y el F3 dataset generator hereda el mismo flujo.

### 2.5 Salida Parquet — confirmada con flag CLI top-level

Comandos verificados:

```bash
sumo -n net.net.xml -r rou.rou.xml --end 60 \
     --tripinfo-output tripinfo.parquet \
     --summary-output summary.parquet
file tripinfo.parquet  # → tripinfo.parquet: Apache Parquet
file summary.parquet   # → summary.parquet:  Apache Parquet
```

Mecanismos disponibles en SUMO 1.26:

- **Por sufijo**: el formato se infiere del nombre de archivo
  (`*.parquet` ⇒ Parquet, `*.csv` ⇒ CSV, `*.xml` ⇒ XML).
- **Override**: `--output.format STR` con valores `xml`, `csv`,
  `parquet` (de `sumo --help`).
- **Compresión Parquet**: `--output.compression STR` (Snappy default
  presumido — pendiente verificación con pyarrow en F3).

**Nota sobre meandata por edge** (relevante para CT-07.3 dataset por
dirección/carril): el meandata por edge **no** es flag top-level; se
configura vía `<edgeData>` en un additional-file. F3 lo cablea en
`simulation/conf/network/edgedata.add.xml` apuntando a un Parquet de
salida. El `--summary-output`/`--tripinfo-output` ya verificados son
suficientes como evidencia de que el pipeline Parquet está vivo.

---

## 3. Decisiones cerradas (preguntas 1-8 del plan F0)

| # | Decisión | Estado |
|---|----------|--------|
| 1 | Venv dedicado y aislado por sub-tarea | ✅ S0: `scratch/sumo_s0/.venv-sumo/`. F1: `simulation/.venv/` |
| 2 | Working dir S0 | ✅ `scratch/sumo_s0/` (gitignored por convención, no se commitea) |
| 3 | Pin `==1.26.0` exacto en `eclipse-sumo` y `traci` | ✅ honrado. `libsumo` queda abierto (§2.2) |
| 4 | `SUMO_HOME` módulo-local | ✅ F1 lo cablea en `simulation/.env.example`; en S0 se setea inline al wheel |
| 5 | Topología F1: genérica vía `netgenerate`, sin OSM | ✅ locked |
| 6 | GUI smoke mínimo en S0 | ✅ ejecutado, exit 0 |
| 7 | Transporte SUMO: TraCI default + libsumo fallback in-process | ⚠️ TraCI smokeado ✅; libsumo blocker (§2.2). Si se adopta opción C, libsumo no entra al toolchain |
| 8 | Dataset format: Parquet default | ✅ confirmado en S0.15 |

---

## 4. Scaffold acordado para `simulation/` (materializa F1, NO F0)

```
simulation/
├── README.md                        # CT-07.7 — reproducibilidad por un tercero
├── pyproject.toml                   # pin: eclipse-sumo==1.26.0, traci==1.26.0,
│                                    #      sumolib==1.27.0 (transitivo); SIN libsumo (opción C)
├── .env.example                     # SUMO_HOME, PATH cableados
├── conf/
│   ├── network/
│   │   ├── miraflores_4way.net.xml  # CT-07.1 — genérica netgenerate
│   │   ├── miraflores_4way.add.xml  # tllogic + edgeData add-file
│   │   └── network_params.yaml      # carriles, vmax 50 km/h, largos — legible
│   └── tllogic/                     # programas semáforo separados si se quiere
├── scenarios/                       # CT-07.2 — 4 patrones con seeds
│   ├── am_peak.sumocfg
│   ├── pm_peak.sumocfg
│   ├── offpeak.sumocfg
│   ├── weekend.sumocfg
│   └── routes/                      # *.rou.xml por patrón
├── src/cerebrovial_simulation/
│   ├── __init__.py
│   ├── jam_level.py                 # sumo_to_jam_level() canónico D-009
│   ├── traci_adapter/               # CT-07.5 — cliente TraCI ↔ motor HTTP
│   ├── dataset/                     # CT-07.3 + CT-07.4 — Parquet, particiones
│   ├── kpis/                        # CT-07.6 — KPIs comparativos
│   └── fixed_control/               # baseline Webster fijo (CT-07.6)
├── scripts/                         # CLIs reproducibles
├── tests/                           # CT-07.8 — mecánicos, no calidad
└── data/                            # output dataset (gitignored)
    ├── train/
    └── valid/
```

**Pin de deps**: vive en `simulation/pyproject.toml`. Anti-regresión
análoga a la regla `torch`/`ultralytics` en `core_management_api`
([CLAUDE.md](../../../CLAUDE.md) §"Deuda técnica a respetar").

**`SUMO_HOME` reproducible** (CT-07.7):
- macOS arm64: `SUMO_HOME=$(repo)/simulation/.venv/lib/python3.11/site-packages/sumo`
  (wheel autocontenido, escenario A; **no usar** el framework
  system-wide).
- Linux: `pip install eclipse-sumo==1.26.0` (wheel disponible) →
  `SUMO_HOME` al path equivalente del venv.

**Límite con `ia_prediction_service/` (TTH-09)**: dataset Parquet en
`simulation/data/{train,valid}/*.parquet`. Sin import cruzado.

**Invocación del motor**: HTTP externo, locked. `POST /control/recommend`
ya existe. Core 100% libre de TraCI.

**Transporte SUMO**: TraCI default (smokeado ✅). libsumo gated por
decisión §2.2 — recomendación agente = opción C (no entra al toolchain).

---

## 5. Verificación end-to-end del cierre F0

Ejecutables directos (con `SUMO_HOME` y `PATH` cableados al wheel):

```bash
# Estado del venv
$ scratch/sumo_s0/.venv-sumo/bin/python -m pip freeze | grep -iE "(sumo|traci)"
eclipse-sumo==1.26.0
sumolib==1.27.0
traci==1.26.0

# SUMO_HOME y binarios
$ export SUMO_HOME=/Users/rasec/Tesis/CerebroVial/scratch/sumo_s0/.venv-sumo/lib/python3.11/site-packages/sumo
$ export PATH=$SUMO_HOME/bin:$PATH
$ which sumo sumo-gui netgenerate duarouter
# → 4 rutas dentro del wheel

# Versión del wheel binario
$ sumo --version
Eclipse SUMO sumo 1.26.0
 Build features: Darwin-23.6.0 arm64 ... GUI ... SWIG Parquet GDAL GL2PS

# Imports Python (libsumo intencionalmente ausente)
$ scratch/sumo_s0/.venv-sumo/bin/python -c "import traci, sumolib; print(traci.__version__, sumolib.__version__)"
1.26.0 1.27.0
$ scratch/sumo_s0/.venv-sumo/bin/python -c "import libsumo"
ModuleNotFoundError: No module named 'libsumo'  ← esperado (§2.2)

# Smoke headless
$ sumo -n net.net.xml -r rou.rou.xml --end 60 --no-step-log --no-warnings  # exit 0

# Smoke GUI (60 pasos, auto-quit)
$ sumo-gui -n net.net.xml -r rou.rou.xml --start --quit-on-end --end 60 --no-step-log --no-warnings  # exit 0

# Smoke TraCI
$ scratch/sumo_s0/.venv-sumo/bin/python scratch/sumo_s0/s0_traci.py
transport=traci edge=A0A1 lane=A0A1_0 mean_speed_mps=9.953 max_speed_mps=13.890

# Parquet
$ sumo -n net.net.xml -r rou.rou.xml --end 60 --tripinfo-output tripinfo.parquet --summary-output summary.parquet
$ file tripinfo.parquet summary.parquet
tripinfo.parquet: Apache Parquet
summary.parquet:  Apache Parquet
```

---

## 6. Backlog post-F0 (decisiones que abren F1)

| Ítem | Tipo | Bloquea | Owner |
|------|------|---------|-------|
| **B-libsumo** | Decisión pin lock | F4 (potencial fallback in-process) | Usuario — opciones A/B/C/D en §2.2; recomendación agente = C |
| **B-engine-contract** | Doc — contrato canónico | F4 (adaptador codeado contra interfaz estable) | F4 abre `documentation/contracts/engine_recommend_contract.md` transcribiendo `IntersectionState` + `ControlRecommendation` de [core_management_api/src/control/presentation/api/schemas.py](../../../core_management_api/src/control/presentation/api/schemas.py) — riesgo cero, transcripción pura |
| **B-Δt-in** | Granularidad dataset | F3 (CT-07.3 esquema) | TTH-11 CT-11.8. Provisional 60s simulados/muestra hasta cierre. F3 puede arrancar con el provisional |
| **B-CT-10.11** | Coordinación motor↔SUMO | F4 (adaptador TraCI) | Diferido a R2 según [ESTADO_Y_PROXIMOS_PASOS.md:43](../../ESTADO_Y_PROXIMOS_PASOS.md#L43). F4 reabre con plan A (HTTP externo) — locked |

---

## 7. Restricciones honradas y scratch residual

- **Cero código productivo en F0**. `simulation/` no existe en el repo.
- **`.gitignore` no modificado**. `scratch/sumo_s0/` queda como "Untracked"
  en `git status`; el commit del handoff usa `git add` específico
  (`documentation/handoffs/tth-07/tth-07-fase0-handoff.md`) para no
  arrastrar el scratch.
- **`.venv/` raíz no tocado**. El venv de S0 vive solo en
  `scratch/sumo_s0/.venv-sumo/` (≈400 MB con el wheel SUMO de 157 MB).
  Se puede borrar con `rm -rf scratch/sumo_s0/` sin afectar al repo.
- **Framework EclipseSUMO en `/Library/Frameworks/`** no participa de
  F1+. Queda como instalación independiente del sistema operativo. F1
  declara explícitamente que `simulation/` adopta el wheel pip y
  ignora el framework system-wide.
- **No push, no PR, no merge**. Commit local del handoff; cualquier
  push y PR queda como decisión humana posterior al cierre F0
  (CLAUDE.md §"Flujo de trabajo").

### 7.1 Decisión sobre `scratch/sumo_s0/` post-handoff

Tres opciones para el scratch residual (decisión usuario, no bloquea F1):

| Opción | Acción | Trade-off |
|--------|--------|-----------|
| (i) Borrar | `rm -rf scratch/sumo_s0/` | Limpieza completa; cualquier re-validación del smoke requiere correr S0 desde cero |
| (ii) Mantener hasta F1 | dejar como está | F1 reusa el venv para iterar sobre `netgenerate` antes de materializar `simulation/.venv/` |
| (iii) Migrar a `simulation/.venv/` en F1 | F1 crea `simulation/.venv/`, copia/reinstala el pin, borra el scratch | Idiomatic; el scratch nunca fue para conservarse |

Recomendación agente: **(iii)** al abrir F1. Mientras F0 cierra, **(ii)**.

---

## 8. Cross-refs

| Doc | Anclaje desde F0 | Anclaje hacia F0 |
|-----|------------------|------------------|
| [DECISIONS.md §D-009](../../lean-inception/4-decisiones/DECISIONS.md#L183-L279) | §2.4 cita el mapeo SUMO→jam level; el smoke TraCI lo ejercita | n/a (D-009 no se modifica) |
| [TAREAS_TECNICAS_HABILITADORAS.md TTH-07](../../lean-inception/2-backlog/TAREAS_TECNICAS_HABILITADORAS.md#L351-L426) | F0 cubre el kickoff; no toca CTs individuales | F1 toma CT-07.1 + CT-07.8.a |
| [CLAUDE.md](../../../CLAUDE.md) | §1 cita regla anti-torch como precedente para `simulation/` fuera del core | F1 agrega `simulation/` a la lista de módulos del monolito |
| [core_management_api/src/control/presentation/api/schemas.py](../../../core_management_api/src/control/presentation/api/schemas.py) | §6 lo nomina como fuente del `engine_recommend_contract.md` que abre F4 | n/a |
| [documentation/contracts/vision_contract.md](../../contracts/vision_contract.md) | Precedente de "contrato canónico" para `engine_recommend_contract.md` (F4) y `simulation_contract.md` (si F4/F5 lo abren) | n/a |

---

## 9. Próximo paso (F1)

**F1 abre cuando el usuario decida sobre B-libsumo (§2.2 y §6)**. F1
materializa:

- `simulation/` top-level (venv interno, pin pyproject.toml, .env.example).
- `simulation/conf/network/miraflores_4way.net.xml` (`netgenerate`
  parametrizado con valores Miraflores documentados: 4 accesos, 2-3
  carriles por acceso, vmax 50 km/h, 200-300 m de largo de aproche).
- `simulation/conf/network/network_params.yaml` (parámetros legibles
  para CT-07.1: "no hardcodeo en un único script").
- `simulation/conf/network/edgedata.add.xml` (additional-file con
  `<edgeData>` apuntando a Parquet para F3).
- `simulation/tests/test_network_loads.py` (CT-07.8.a — topología
  carga sin errores).
- `documentation/handoffs/tth-07/tth-07-fase1-handoff.md` al cierre.

F1 NO toca dataset (F3), patrones (F2), adaptador motor (F4), ni KPIs
(F5).
