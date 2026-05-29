#!/usr/bin/env bash
# Regenera el dataset de TTH-09 (CT-07.3) reproducible (CT-07.7 punto c).
# Default: full (80 train + 20 valid). Pasar --quick para 1 seed por
# patrón × partición (smoke).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PY="$SIM_ROOT/.venv/bin/python"

# SUMO_HOME al wheel (escenario A — autocontenido).
SUMO_PKG="$SIM_ROOT/.venv/lib/python3.11/site-packages/sumo"
export SUMO_HOME="$SUMO_PKG"
export PATH="$SUMO_PKG/bin:$PATH"

cd "$SIM_ROOT"
echo "Regenerando dataset desde $SIM_ROOT"
echo "  SUMO_HOME=$SUMO_HOME"
echo "  python=$VENV_PY"
"$VENV_PY" -m cerebrovial_simulation.dataset.generate "$@"
echo "OK"
