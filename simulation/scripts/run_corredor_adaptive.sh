#!/usr/bin/env bash
# ETAPA 2 / IE05 — corrida adaptativa (Max Pressure per-node) del corredor Larco vs fijo.
#
# Mismo escenario que el baseline fijo (peak_s_n.sumocfg, scale=1.0, seed=42, end=1800,
# mismos detectores E2). SOLO cambia el control: Benavides y Schell se controlan por el
# motor (POST :8001/control/recommend); Diez Canseco queda en su control fijo.
#
# FAIL-RÁPIDO: requiere el motor arriba en :8001 (db + core). NO lo levanta solo
# (invoke up arrastra docker pesado con efectos secundarios). Si está caído, aborta con
# la instrucción.
#
# Uso:
#   invoke up          # (en la raíz del repo, en otra terminal) levanta db + core
#   bash scripts/run_corredor_adaptive.sh                  # seed=42 end=1800 warmup=600
#   bash scripts/run_corredor_adaptive.sh --end 300        # smoke

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PY="$SIM_ROOT/.venv/bin/python"
SUMO_PKG="$SIM_ROOT/.venv/lib/python3.11/site-packages/sumo"
export SUMO_HOME="$SUMO_PKG"
export PATH="$SUMO_PKG/bin:$PATH"

ENGINE_HEALTH="${ENGINE_HEALTH:-http://localhost:8001/control/health}"
SEED="${SEED:-42}"; END="${END:-1800}"; WARMUP="${WARMUP:-600}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
    --end) END="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    *) echo "arg desconocido: $1" >&2; exit 2;;
  esac
done

echo ">> health-check del motor en $ENGINE_HEALTH"
if ! curl -sf --max-time 5 "$ENGINE_HEALTH" >/dev/null 2>&1; then
  echo "ERROR: el motor no responde 200 en $ENGINE_HEALTH." >&2
  echo "       Levantalo primero desde la raíz del repo:  invoke up" >&2
  echo "       (el endpoint /control/recommend necesita db + core; no lo levanto solo.)" >&2
  exit 1
fi
echo "   motor OK"

OUT_DIR="$SIM_ROOT/data/corredor_larco/peak_s_n_adaptive_seed${SEED}_end${END}"
FIXED_DIR="$SIM_ROOT/data/corredor_larco/peak_s_n_seed${SEED}_end${END}"

echo ">> generando demanda baseline (scale=1.0, determinista)"
"$VENV_PY" "$SCRIPT_DIR/generate_corredor_routes.py" >/dev/null

echo ">> corriendo adaptativo (seed=$SEED end=$END) → $OUT_DIR"
"$VENV_PY" -m cerebrovial_simulation.traci_adapter.corredor_adaptive \
  --seed "$SEED" --end "$END" --out "$OUT_DIR"

echo ">> KPIs adaptativo"
"$VENV_PY" -m cerebrovial_simulation.kpis.collect_corredor \
  --out "$OUT_DIR" --warmup "$WARMUP" \
  --params "$SIM_ROOT/conf/corredor_larco/demand_params.yaml" \
  --json "$OUT_DIR/kpis.json"

if [[ ! -f "$FIXED_DIR/kpis.json" ]]; then
  echo ">> baseline fijo no encontrado en $FIXED_DIR — corriéndolo para comparar"
  bash "$SCRIPT_DIR/run_corredor_baseline.sh" --seed "$SEED" --end "$END" --warmup "$WARMUP" >/dev/null
fi

echo ">> comparación adaptativo vs fijo (RD%)"
"$VENV_PY" "$SCRIPT_DIR/compare_corredor.py" --fixed "$FIXED_DIR" --adaptive "$OUT_DIR"
