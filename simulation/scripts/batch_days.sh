#!/usr/bin/env bash
# Genera+compacta días seeds 42..101 (60 días, perfil laborable scale 1.1). Secuencial.
# Para en el primer fallo (no sigue sobre un bug). DESCARTABLE/scratch. NO commits.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FAILED=""
for s in $(seq 42 101); do
  if bash "$HERE/gen_day.sh" "$s"; then :; else FAILED="$s"; break; fi
done
if [ -n "$FAILED" ]; then echo "### FALLO en seed $FAILED — DETENIDO ###"; exit 1; fi
echo "### TODOS LOS 60 (42..101) OK ###"
