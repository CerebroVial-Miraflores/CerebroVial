#!/usr/bin/env bash
# Runner del BRAZO FIJO del experimento de validación fijo-vs-adaptativo sobre Miraflores
# COMPLETO (F2, DHU-030). Parametrizado por SEED, AGNÓSTICO al # de nodos.
#
# Control FIJO = los tlLogic de netconvert embebidos en miraflores.net.xml (sin motor, sin
# TraCI). Reusa los patrones de gen_day.sh (regeneración de demanda, pasos 1-2) y de
# run_corredor_baseline.sh (invocación SUMO headless + KPIs), SIN nada hardcodeado al
# corredor: no hay dirs peak_s_n_*, ni eje S→N, ni los 6 edges instrumentados. Solo
# net + demanda del seed + control fijo + métrica de red (net_timeLoss, vía collect_red_metrics.py).
#
# Demanda: para seed 51 reusa la YA VERSIONADA (conf/miraflores_red_completa/routes/...);
# para cualquier otro seed la regenera (gen_day pasos 1-2, scale 1.1) a un dir de trabajo
# bajo runs/ (gitignoreado, efímero — esas demandas no se versionan).
#
# Salidas: runs/fijo_seed<NNN>/{tripinfo,summary}.parquet + sumo.log (todo gitignoreado).
#
# NO toca el brazo adaptativo (no existe aún para la red completa — es F3/F6). F2 solo el fijo.
#
# Uso:
#   bash simulation/scripts/run_fijo_red.sh <seed>
#   bash simulation/scripts/run_fijo_red.sh 51     # reproduce el baseline de F1 (~115.2 s)
set -euo pipefail

SEED="${1:?uso: run_fijo_red.sh <seed>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(dirname "$SCRIPT_DIR")"                 # scripts/ -> simulation/
VENV_PY="$SIM_ROOT/.venv/bin/python"
SUMO_PKG="$SIM_ROOT/.venv/lib/python3.11/site-packages/sumo"
export SUMO_HOME="$SUMO_PKG"
export PATH="$SUMO_PKG/bin:$PATH"

NET="$SIM_ROOT/conf/network/miraflores.net.xml"
EXP_DIR="$SIM_ROOT/conf/miraflores_red_completa"
VERSIONED_ROU="$EXP_DIR/routes/miraflores_seed051_laborable.rou.xml"
RUNS="$EXP_DIR/runs"
SEEDP="$(printf "%03d" "$SEED")"
OUT="$RUNS/fijo_seed${SEEDP}"
mkdir -p "$OUT"

# 1) Localizar / regenerar la demanda del seed (mismos parámetros que F1: ratio 5, scale 1.1).
if [ "$SEED" -eq 51 ] && [ -f "$VERSIONED_ROU" ]; then
  ROU="$VERSIONED_ROU"
  echo ">> seed 51: reuso la demanda VERSIONADA -> $ROU"
else
  DEMAND="$RUNS/_demand_seed${SEEDP}"
  ROU="$DEMAND/rou.rou.xml"
  if [ -f "$ROU" ]; then
    echo ">> seed $SEED: reuso demanda regenerada previa -> $ROU"
  else
    echo ">> seed $SEED: regenero demanda (gen_day pasos 1-2, scale 1.1, seed $SEED)"
    mkdir -p "$DEMAND"
    python3 "$SCRIPT_DIR/generate_b1_demand.py" --net "$NET" --ratio 5 --scale 1.1 \
      --seed "$SEED" --outdir "$DEMAND" >/dev/null
    ( cd "$DEMAND" && duarouter -n "$NET" --route-files "$(cat trip_files.txt)" \
        -o rou.rou.xml --alternatives-output NUL --seed "$SEED" --ignore-errors --no-step-log )
  fi
fi

# 2) Brazo FIJO sobre el net (tlLogic embebidos), día completo, parámetros que F1 validó.
echo ">> SUMO fijo (seed=$SEED, día completo 0..86400s) -> $OUT"
sumo -n "$NET" -r "$ROU" \
  --begin 0 --end 86400 --step-length 1 \
  --time-to-teleport 300 --collision.action warn --seed "$SEED" \
  --tripinfo-output "$OUT/tripinfo.parquet" --summary-output "$OUT/summary.parquet" \
  --no-step-log --log "$OUT/sumo.log"

# 3) Métrica de red + señales de salud (agnóstico; sin corredor).
echo ">> métrica de red (net_timeLoss + censura/teleports/completados)"
"$VENV_PY" "$SCRIPT_DIR/collect_red_metrics.py" --out "$OUT" --json "$OUT/metrics.json"
