#!/usr/bin/env bash
# F5 — run_comparison: adaptive (engine HTTP) vs fixed-Webster (offline) × 4 patrones × N seeds.
#
# Requiere motor corriendo para la rama adaptive (`invoke up`).
# La rama fixed-Webster NO requiere motor (Catch B).
#
# Output: simulation/data/kpis/comparison.csv con columnas
#         (mode, pattern, seed, sim_duration_s, mean_travel_time_s,
#          total_delay_s, throughput_veh_per_h, max_queue_m_{N,S,E,W},
#          mean_queue_m_{N,S,E,W}).
#
# Uso:
#   bash scripts/run_comparison.sh            # 3 seeds per (mode, pattern) = 24 corridas
#   bash scripts/run_comparison.sh --quick    # 1 seed per (mode, pattern) = 8 corridas

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PY="$SIM_ROOT/.venv/bin/python"
SUMO_PKG="$SIM_ROOT/.venv/lib/python3.11/site-packages/sumo"
export SUMO_HOME="$SUMO_PKG"
export PATH="$SUMO_PKG/bin:$PATH"

cd "$SIM_ROOT"
"$VENV_PY" -m cerebrovial_simulation.kpis.run_comparison "$@"
