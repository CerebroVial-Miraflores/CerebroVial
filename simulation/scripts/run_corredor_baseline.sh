#!/usr/bin/env bash
# ETAPA 2 — baseline peak S→N corredor Larco: control fijo + KPIs + veredicto spillback.
#
# Corrida headless, semilla fija, sin motor. Genera demanda (determinista), corre sumo
# con los tlLogic estáticos embebidos en el .net.xml y agrega KPIs con collect_corredor.
#
# Patrón de paths (igual a run_comparison.py): los paths del .sumocfg (net, routes)
# resuelven relativos al directorio del cfg; el det.add.xml se copia a out_dir y se pasa
# por --additional-files, así su file="lanearea.xml" cae en out_dir (cwd de sumo).
#
# Uso:
#   bash scripts/run_corredor_baseline.sh                 # end=1800, seed=42, warmup=600
#   bash scripts/run_corredor_baseline.sh --end 300       # smoke corto
#   END=300 SEED=7 bash scripts/run_corredor_baseline.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PY="$SIM_ROOT/.venv/bin/python"
SUMO_PKG="$SIM_ROOT/.venv/lib/python3.11/site-packages/sumo"
export SUMO_HOME="$SUMO_PKG"
export PATH="$SUMO_PKG/bin:$PATH"

CONF_DIR="$SIM_ROOT/conf/corredor_larco"
CFG="$CONF_DIR/peak_s_n.sumocfg"
DET="$CONF_DIR/corredor_larco.det.add.xml"
PARAMS="$CONF_DIR/demand_params.yaml"

# Defaults (override por env o flags simples).
END="${END:-1800}"
SEED="${SEED:-42}"
WARMUP="${WARMUP:-600}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --end) END="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    *) echo "arg desconocido: $1" >&2; exit 2;;
  esac
done

OUT_DIR="$SIM_ROOT/data/corredor_larco/peak_s_n_seed${SEED}_end${END}"
mkdir -p "$OUT_DIR"
cp "$DET" "$OUT_DIR/"

echo ">> generando demanda (determinista)"
"$VENV_PY" "$SCRIPT_DIR/generate_corredor_routes.py"

echo ">> corriendo sumo (end=${END}s seed=${SEED}) → $OUT_DIR"
"$SUMO_PKG/bin/sumo" \
  -c "$CFG" \
  --additional-files "$OUT_DIR/corredor_larco.det.add.xml" \
  --seed "$SEED" \
  --end "$END" \
  --summary-output "$OUT_DIR/summary.parquet" \
  --tripinfo-output "$OUT_DIR/tripinfo.parquet" \
  --no-step-log \
  --log "$OUT_DIR/sumo.log" \
  --message-log "$OUT_DIR/sumo.msg.log" 2> >(tee "$OUT_DIR/sumo.stderr" >&2) || {
    echo "sumo falló — ver $OUT_DIR/sumo.stderr" >&2; exit 1;
  }

echo ">> teleports/gridlock en el log:"
grep -ciE "teleport|gridlock|jam" "$OUT_DIR/sumo.log" 2>/dev/null || true

echo ">> agregando KPIs + veredicto spillback"
"$VENV_PY" -m cerebrovial_simulation.kpis.collect_corredor \
  --out "$OUT_DIR" \
  --warmup "$WARMUP" \
  --params "$PARAMS" \
  --json "$OUT_DIR/kpis.json"
