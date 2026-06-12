#!/usr/bin/env bash
# Orquesta el benchmark completo de captura de visión (read-only, fuera de producción).
# Corre TODO secuencial (las mediciones de CPU no deben solaparse).
#
# Requiere dos venvs (ver README):
#   PROD_VENV : venv de producción (cv2 + módulo vision + ultralytics) — default ./.venv
#   PYAV_VENV : venv aparte con `av` (+cv2,psutil,ultralytics) — default ~/.venvs/pyav-bench
#
# Override de duraciones: SINGLE_DUR, CONC_DUR (segundos).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
PROD_VENV="${PROD_VENV:-$REPO/.venv}"
PYAV_VENV="${PYAV_VENV:-$HOME/.venvs/pyav-bench}"
PY_PROD="$PROD_VENV/bin/python"
PY_PYAV="$PYAV_VENV/bin/python"
SINGLE_DUR="${SINGLE_DUR:-120}"
CONC_DUR="${CONC_DUR:-80}"

echo "### entorno"; "$PY_PROD" --version; "$PY_PYAV" -c "import av; print('PyAV', av.__version__)"
echo "### Eje 3 — CORS"; "$PY_PROD" "$HERE/cors_check.py"

echo "### Eje 1 — single (dur=$SINGLE_DUR)"
for m in native sleep1 kf_ffmpeg; do "$PY_PROD" "$HERE/run_single.py" --mech $m --dur "$SINGLE_DUR"; done
"$PY_PYAV" "$HERE/run_single.py" --mech kf_pyav --dur "$SINGLE_DUR"

echo "### Eje 1 — concurrencia 8 streams, inferencia 1 fps (dur=$CONC_DUR)"
for m in native sleep1 kf_ffmpeg; do "$PY_PROD" "$HERE/run_concurrency.py" --mech $m --streams 8 --infer-fps 1 --dur "$CONC_DUR"; done
"$PY_PYAV" "$HERE/run_concurrency.py" --mech kf_pyav --streams 8 --infer-fps 1 --dur "$CONC_DUR"

echo "### Eje 2 — cadencia de inferencia sobre kf_ffmpeg, 8 streams (dur=$CONC_DUR)"
for f in 0.5 2; do "$PY_PROD" "$HERE/run_concurrency.py" --mech kf_ffmpeg --streams 8 --infer-fps $f --dur "$CONC_DUR"; done
echo "(infer 1 fps de Eje 2 = fila kf_ffmpeg/infer_fps=1 de Eje 1)"

echo "### FIN"
