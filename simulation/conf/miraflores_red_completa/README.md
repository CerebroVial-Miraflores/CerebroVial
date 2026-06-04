# Miraflores red completa — demanda seed051 + config observable (F1, DHU-030)

Insumos del experimento de validación del control **fijo-vs-adaptativo** sobre la red completa
de Miraflores (`miraflores.net.xml`, net v2, 99 TLS). Esta carpeta arranca con el **día laborable
completo seed051** y su `.sumocfg` observable; las fases siguientes (runner, brazo adaptativo,
KPIs) cuelgan de acá.

## Qué es

- **`routes/miraflores_seed051_laborable.rou.xml`** (~12 MB, 33 557 vehículos) — la demanda del
  **"lunes 8 de junio"** de la demo. Es el día **seed051 = day_idx 9** del dataset de predicción
  `simulation/data/datasets/miraflores_laborable_60d/` (mapeo `2026-06-08 ↔ seed051` por convención
  en `core_management_api/src/congestion/presentation/api/routes.py`). Curva laborable B2 de 10
  fases, scale 1.1 (gate D-014).
- **`miraflores_seed051_laborable.sumocfg`** — config día-completo (24 h) **observable**: corre
  headless para medir y se abre en `sumo-gui` para presentar. Control **fijo** = los `tlLogic` de
  netconvert embebidos en el net (sin motor, sin TraCI).

## Por qué se versiona (excepción consciente)

La regla general del proyecto es **"los regenerables no se versionan"** (ver
`simulation/scripts/REGENERACION_DEMANDA_B2.md`): `gen_day.sh` de hecho **borra** el `.rou.xml`
de cada día tras compactar a Parquet. Este `.rou.xml` se versiona **excepcionalmente** porque es a
la vez (a) el **insumo del experimento** de validación y (b) el **respaldo visual de la
presentación** — el único día que el mapa de congestión del frontend muestra hardcodeado. Tenerlo
versionado evita depender de regenerarlo en vivo el día de la demo. Se versiona como **blob git
normal** (no LFS: `.gitattributes` no captura `*.rou.xml`).

## Cómo abrir en sumo-gui (para observar / presentar)

```bash
cd simulation/conf/miraflores_red_completa
sumo-gui -c miraflores_seed051_laborable.sumocfg
```
El `seed=51` + `time-to-teleport=300` + `collision.action=warn` embebidos reproducen la **misma
dinámica** que generó `day_seed051.parquet` (la congestión que pinta el mapa).

## Cómo regenerar la demanda (Escenario A, determinística)

Son los pasos 1-2 de `simulation/scripts/gen_day.sh` con seed 51 (verificado en auditoría):

```bash
cd <repo-root>
export SUMO_HOME="$PWD/simulation/.venv/lib/python3.11/site-packages/sumo"
NET="$PWD/simulation/conf/network/miraflores.net.xml"
python3 simulation/scripts/generate_b1_demand.py --ratio 5 --scale 1.1 --seed 51 --outdir /tmp/day_seed051
( cd /tmp/day_seed051 && duarouter -n "$NET" --route-files "$(cat trip_files.txt)" \
    -o rou.rou.xml --alternatives-output NUL --seed 51 --ignore-errors --no-step-log )
# luego: cp /tmp/day_seed051/rou.rou.xml routes/miraflores_seed051_laborable.rou.xml
```

### Salvedad de reproducibilidad
- **"Idéntica" = demanda idéntica, no byte-idéntica.** El header del `.rou.xml` lleva un
  timestamp que cambia en cada corrida; los vehículos (IDs, departs, rutas) sí son determinísticos
  con seed 51. Un diff de solo-timestamp **no** es un fallo de regeneración.
- **Exige el entorno pinneado:** **SUMO 1.26.0** + **Python 3.11** (`simulation/pyproject.toml`).
  El `.venv` es gitignoreado pero reconstruible (`invoke setup-dev` / `pip`).

## Outputs

El `.sumocfg` escribe `runs/tripinfo.parquet` + `runs/summary.parquet` (Parquet **nativo** de SUMO
1.26). `runs/` está **gitignoreado**: son regenerables, no se versionan. Solo se versionan la
demanda, el `.sumocfg` y este README.
