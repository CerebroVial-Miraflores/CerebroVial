# Regeneración de la demanda B2 (scale 1.1, gate D-014)

Nota de regeneración. **No se versiona el `.rou.xml`**: es un artefacto regenerable
determinísticamente desde insumos ya versionados. Acá queda documentado el cómo y el
porqué, para cerrar la deuda de tenerlo sólo en `scratch/` (gitignoreado) sin
versionar un binario de ~12 MB.

## Qué produce

La demanda vehicular de **Miraflores completo** a **scale 1.1** — la del gate **D-014**:

- ~33 557 vehículos.
- Curva horaria laborable de **10 fases** (doble spike AM 07–09 / PM 18–20, meseta
  diurna, valles nocturnos), heredada de la forma `laborable`.
- Ruteada sobre `simulation/conf/network/miraflores.net.xml` (net completo).

## Por qué NO se versiona

Es **regenerable determinística** desde artefactos ya versionados (Escenario A,
confirmado por auditoría 2026-06-04). Todo lo que define la demanda está fijado en el repo:

- **Seed fija (42)**, con doble respaldo: `sweep24.sh` la pasa explícita
  (`SEED="${SEED:-42}"`) y `generate_b1_demand.py` la tiene como default
  (`--seed`, default 42). `randomTrips.py` se invoca siempre con `--seed`.
- **Parámetros hardcodeados** en `generate_b1_demand.py`: curva `LABORABLE_PHASES`
  (10 fases), `ARTERIAL_TYPES` (clases viales con peso alto), `ratio=5` (arteria:resto),
  `scale=1.1`, OD por pesos de clase vial.
- **Decisión de scale** documentada en
  `simulation/data/datasets/miraflores_laborable_60d/calibracion/`
  (`DRENAJE_GATE_60D_RESULTS.md`, `SWEEP_C1_RESULTS.md`, `SWEEP_C2_RESULTS.md`,
  `SWEEP_C3_RESULTS.md` — todos versionados).

Versionar el `.rou.xml` (~12 MB) violaría la regla *"los regenerables no se versionan
salvo input cross-session"* sin necesidad: no hay ningún input cross-session atrapado
en `scratch/`. Todo insumo vive ya en el repo.

## Comando de regeneración

Dos caminos, ambos verificados contra los scripts reales del repo.

### Camino completo (vía `sweep24.sh`, incluye la corrida 24h del gate)

```bash
cd /Users/rasec/Tesis/CerebroVial
export SUMO_HOME="$PWD/simulation/.venv/lib/python3.11/site-packages/sumo"
OUTDIR=/tmp/b2_regen \
NET="$PWD/simulation/conf/network/miraflores.net.xml" \
SEED=42 \
  simulation/scripts/sweep24.sh 1.1
# → /tmp/b2_regen/sweep24_s11/rou_s11.rou.xml
```

### Solo el `.rou.xml` (sin la corrida 24h)

```bash
cd /Users/rasec/Tesis/CerebroVial
export SUMO_HOME="$PWD/simulation/.venv/lib/python3.11/site-packages/sumo"
NET="$PWD/simulation/conf/network/miraflores.net.xml"
python3 simulation/scripts/generate_b1_demand.py --net "$NET" \
        --ratio 5 --scale 1.1 --seed 42 --outdir /tmp/b2_regen
( cd /tmp/b2_regen && duarouter -n "$NET" \
    --route-files "$(cat trip_files.txt)" -o rou_s11.rou.xml --seed 42 --ignore-errors )
```

> **Nota sobre este segundo camino (ajuste respecto del comando de la auditoría):**
> - El `export SUMO_HOME` es **obligatorio también acá**: `generate_b1_demand.py`
>   aborta con `sys.exit` si `SUMO_HOME` no está seteado (lo necesita para localizar
>   `randomTrips.py`). El comando de la auditoría lo omitía.
> - `NET` se captura como **ruta absoluta antes del `cd /tmp/b2_regen`**. La versión
>   previa usaba `$PWD/../../simulation/...` *dentro* del `cd`, lo que resuelve a `/`
>   (porque ahí `$PWD` ya es `/tmp/b2_regen`) y rompía. Capturarlo antes lo deja absoluto.

## Salvedades (honestidad de reproducibilidad)

- **"Idéntica" = demanda idéntica, no byte-idéntica.** El header del `.rou.xml` lleva un
  timestamp que cambia en cada corrida; los vehículos (IDs, departs, rutas) sí son
  determinísticos con seed 42 + SUMO 1.26.0. Un diff de **solo-timestamp** no es un fallo
  de regeneración.
- **Dependencia de entorno.** La reproducibilidad exacta exige **SUMO 1.26.0**
  (`eclipse-sumo==1.26.0`) + **Python 3.11** (pin `requires-python = ">=3.11,<3.13"`),
  ambos en `simulation/pyproject.toml`. El `.venv` es gitignoreado pero **reconstruible**
  (`invoke setup-dev` / `pip`), no se rescata de `scratch/`.

## Alcance

Hoy **NO** existe un `.sumocfg` versionado que apunte a esta demanda sobre la red
completa. Cablearlo es trabajo de la simulación **fijo-vs-adaptativo** (fase siguiente
de esta rama), no parte de esta nota.
