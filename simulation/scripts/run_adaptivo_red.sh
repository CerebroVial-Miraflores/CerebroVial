#!/usr/bin/env bash
# Runner del BRAZO ADAPTATIVO del experimento de validación sobre Miraflores COMPLETO
# (F3b, DHU-030). Per-node puro sobre los TLS de 2 fases (54, o solo Larco con --larco),
# alimentado por el generador node_config_gen (F3a). Mapeo al motor: intersection_id =
# "sumo_"+tls_id (Decisión 2). Mismos parámetros SUMO que el fijo de F1/F2 (tt=300,
# collision warn, seed, 0..end) → comparable apples-to-apples; única diferencia: el control.
#
# Levanta el motor en :8001 si no está, espera a que responda, y lo BAJA al terminar SOLO
# si fue este runner quien lo levantó (no tira abajo un stack que el usuario ya tenía up).
# Reusa collect_red_metrics.py (F2) para la métrica net_timeLoss.
#
# Uso:
#   bash simulation/scripts/run_adaptivo_red.sh <seed> [--end <s>] [--larco]
#   bash simulation/scripts/run_adaptivo_red.sh 51 --end 3600 --larco   # validación Larco 1h
#   bash simulation/scripts/run_adaptivo_red.sh 51 --end 3600           # 54 nodos 1h
#   bash simulation/scripts/run_adaptivo_red.sh 51                       # 54 nodos día completo
set -euo pipefail

SEED="${1:?uso: run_adaptivo_red.sh <seed> [--end <s>] [--larco]}"; shift || true
END=86400; LARCO=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --end) END="$2"; shift 2;;
    --larco) LARCO="--larco"; shift;;
    *) echo "arg desconocido: $1" >&2; exit 2;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$SIM_ROOT")"
VENV_PY="$SIM_ROOT/.venv/bin/python"
SUMO_PKG="$SIM_ROOT/.venv/lib/python3.11/site-packages/sumo"
export SUMO_HOME="$SUMO_PKG"
export PATH="$SUMO_PKG/bin:$PATH"

NET="$SIM_ROOT/conf/network/miraflores.net.xml"
EXP_DIR="$SIM_ROOT/conf/miraflores_red_completa"
VERSIONED_ROU="$EXP_DIR/routes/miraflores_seed051_laborable.rou.xml"
RUNS="$EXP_DIR/runs"
SEEDP="$(printf "%03d" "$SEED")"
TAG="adaptivo_seed${SEEDP}"; [[ -n "$LARCO" ]] && TAG="adaptivo_larco_seed${SEEDP}"
OUT="$RUNS/${TAG}_end${END}"
HEALTH="http://localhost:8001/control/health"
mkdir -p "$OUT"

# --- motor: levantar solo si no está; bajar al final solo si lo levantamos nosotros ---
STARTED_MOTOR=0
cleanup() {
  if [[ "$STARTED_MOTOR" == "1" ]]; then
    echo ">> bajando el motor (lo levantó este runner)"
    ( cd "$REPO_ROOT" && invoke down >/dev/null 2>&1 || true )
  else
    echo ">> motor queda como estaba (no lo levantó este runner)"
  fi
}
trap cleanup EXIT

if curl -sf --max-time 4 "$HEALTH" >/dev/null 2>&1; then
  echo ">> motor ya está arriba en :8001"
else
  echo ">> levantando el motor (invoke up --service=core_management_api)"
  ( cd "$REPO_ROOT" && invoke up --service=core_management_api )
  STARTED_MOTOR=1
  echo ">> esperando a que :8001 responda..."
  for i in $(seq 1 90); do
    if curl -sf --max-time 3 "$HEALTH" >/dev/null 2>&1; then echo "   motor OK (intento $i)"; break; fi
    sleep 2
    if [[ "$i" == "90" ]]; then echo "ERROR: el motor no respondió en ~3 min" >&2; exit 1; fi
  done
fi

# --- demanda del seed: seed 51 = versionada; otros = regenerar (gen_day pasos 1-2) ---
if [[ "$SEED" == "51" && -f "$VERSIONED_ROU" ]]; then
  ROU="$VERSIONED_ROU"; echo ">> seed 51: demanda VERSIONADA -> $ROU"
else
  DEMAND="$RUNS/_demand_seed${SEEDP}"; ROU="$DEMAND/rou.rou.xml"
  if [[ ! -f "$ROU" ]]; then
    echo ">> seed $SEED: regenero demanda (gen_day pasos 1-2, scale 1.1)"
    mkdir -p "$DEMAND"
    python3 "$SCRIPT_DIR/generate_b1_demand.py" --net "$NET" --ratio 5 --scale 1.1 \
      --seed "$SEED" --outdir "$DEMAND" >/dev/null
    ( cd "$DEMAND" && duarouter -n "$NET" --route-files "$(cat trip_files.txt)" \
        -o rou.rou.xml --alternatives-output NUL --seed "$SEED" --ignore-errors --no-step-log )
  fi
fi

# --- brazo adaptativo (TraCI) ---
echo ">> adaptativo per-node (seed=$SEED end=$END ${LARCO:-54-nodos}) -> $OUT"
"$VENV_PY" -m cerebrovial_simulation.traci_adapter.red_adaptive \
  --seed "$SEED" --end "$END" --out "$OUT" --net "$NET" --rou "$ROU" $LARCO

# --- métrica de red (misma que el fijo) ---
echo ">> métrica de red"
"$VENV_PY" "$SCRIPT_DIR/collect_red_metrics.py" --out "$OUT" --json "$OUT/metrics.json"
