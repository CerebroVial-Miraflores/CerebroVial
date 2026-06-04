#!/usr/bin/env bash
# Barrido de scale sobre la 24h CONTINUA completa (control fijo, sin TraCI).
# Maquinaria del barrido C2, parametrizada por NET para barrer sobre distintos nets.
# Uso:   sweep24.sh <scale>
# Vars overridables por entorno (con default):
#   NET     ruta al .net.xml          (default: net viejo miraflores.net.xml, relativo al repo)
#   SEED    semilla determinista      (default: 42)
#   OUTDIR  dir base de salida        (default: este dir; la corrida va en OUTDIR/sweep24_s<tag>/)
# Ej. sanity sobre net viejo a scratch: OUTDIR=.../scratch/b2_miraflores sweep24.sh 0.20
set -euo pipefail
SCALE="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
# scripts/ -> simulation/ -> repo-root. Default del net relativo al repo, NO machine-specific.
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
TAG="$(echo "$SCALE" | sed 's/\.//')"      # 0.25 -> 025
NET="${NET:-$REPO_ROOT/simulation/conf/network/miraflores.net.xml}"
SEED="${SEED:-42}"
OUTDIR="${OUTDIR:-$HERE}"
DIR="$OUTDIR/sweep24_s${TAG}"
mkdir -p "$DIR"
ROU="$DIR/rou_s${TAG}.rou.xml"
EDGEOUT="$DIR/edgedata_s${TAG}.xml"

# add-file local: misma config de muestreo (freq=60, excludeEmpty=false), output a scratch.
cat > "$DIR/edgedata_local.add.xml" <<XML
<additional>
  <edgeData id="aggE" freq="60" file="${EDGEOUT}" excludeEmpty="false"/>
</additional>
XML

echo "[$SCALE] generar demanda (net=$NET)"
python3 "$HERE/generate_b1_demand.py" --net "$NET" --ratio 5 --scale "$SCALE" --seed "$SEED" --outdir "$DIR" >/dev/null

echo "[$SCALE] duarouter"
( cd "$DIR" && duarouter -n "$NET" --route-files "$(cat "$DIR/trip_files.txt")" \
    -o "$ROU" --alternatives-output NUL --seed "$SEED" --ignore-errors --no-step-log 2>/dev/null )

echo "[$SCALE] sumo 24h"
sumo -n "$NET" -r "$ROU" --additional-files "$DIR/edgedata_local.add.xml" \
  -b 0 -e 86400 --time-to-teleport 300 --collision.action warn \
  --statistic-output "$DIR/stats_s${TAG}.xml" --duration-log.statistics --no-step-log \
  --log "$DIR/sumo_s${TAG}.log"
echo "[$SCALE] LISTO -> $DIR"
