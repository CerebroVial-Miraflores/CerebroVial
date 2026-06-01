#!/usr/bin/env bash
# Genera+compacta días seeds 43..101 (los 59 restantes; el 42 ya está). Secuencial.
# Para en el primer fallo (no sigue sobre un bug). DESCARTABLE/scratch. NO commits.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FAILED=""
for s in $(seq 43 101); do
  if bash "$HERE/gen_day.sh" "$s"; then :; else FAILED="$s"; break; fi
done
if [ -n "$FAILED" ]; then echo "### FALLO en seed $FAILED — DETENIDO ###"; exit 1; fi
echo "### TODOS LOS 59 (43..101) OK ###"
