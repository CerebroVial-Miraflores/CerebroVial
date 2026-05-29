#!/usr/bin/env bash
# E2E con motor adaptativo real (CT-07.5).
#
# Requiere que el motor esté corriendo en otra terminal:
#     invoke up   (desde el root del repo)
#
# Uso:
#     bash scripts/run_e2e_with_engine.sh [--pattern am_peak] [--seed 1] [--end 600]
#
# El script valida primero que ENGINE_URL responde a /control/health.
# Si no, falla rápido con instrucciones.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PY="$SIM_ROOT/.venv/bin/python"
ENGINE_URL="${ENGINE_URL:-http://localhost:8001/control/recommend}"

# health-check URL: derivar de ENGINE_URL → /control/health.
HEALTH_URL="$(echo "$ENGINE_URL" | sed 's|/recommend$|/health|')"
echo "Verificando motor en $HEALTH_URL ..."
if ! curl -s -f -o /dev/null -m 5 "$HEALTH_URL"; then
    echo "ERROR: motor no responde en $HEALTH_URL"
    echo "  Levantá el motor con 'invoke up' desde la raíz del repo."
    echo "  Si el motor corre en otro puerto/host, exportá ENGINE_URL."
    exit 1
fi
echo "OK"

SUMO_PKG="$SIM_ROOT/.venv/lib/python3.11/site-packages/sumo"
export SUMO_HOME="$SUMO_PKG"
export PATH="$SUMO_PKG/bin:$PATH"
export ENGINE_URL

cd "$SIM_ROOT"
echo "Corrida e2e desde $SIM_ROOT"
echo "  ENGINE_URL=$ENGINE_URL"
"$VENV_PY" -m cerebrovial_simulation.traci_adapter.run_e2e "$@"
