#!/usr/bin/env bash
# Dataset laborable 60d — genera+rutea+simula+compacta UN día (perfil laborable,
# scale=0.20). seed -> randomTrips, duarouter Y sumo (varía ruteo y dinámica).
# Compacta a Parquet y borra el XML pesado. DESCARTABLE/scratch. NO conf/. NO commits.
# Uso: gen_day.sh <seed> [keepxml]
set -euo pipefail
SEED="$1"; KEEPXML="${2:-no}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# scripts/ -> simulation/ -> repo-root. Todas las rutas relativas al repo-root, nada machine-specific.
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
# Destino del dataset: la ubicación versionada. Override con B1_DS_OUT para regenerar
# a un dir temporal sin pisar el dataset promovido (gate de regeneración).
DS="${B1_DS_OUT:-$REPO_ROOT/simulation/data/datasets/miraflores_laborable_60d}"
NET="$REPO_ROOT/simulation/conf/network/miraflores.net.xml"
VENV="$REPO_ROOT/.venv/bin/python"
SCALE=0.20
SEEDP=$(printf "%03d" "$SEED")
WORK="$DS/_work_seed${SEEDP}"
PARQUET="$DS/day_seed${SEEDP}.parquet"
mkdir -p "$WORK"
EDGEOUT="$WORK/edgedata.xml"
ROU="$WORK/rou.rou.xml"

cat > "$WORK/edgedata_local.add.xml" <<XML
<additional>
  <edgeData id="aggE" freq="60" file="${EDGEOUT}" excludeEmpty="false"/>
</additional>
XML

# 1) demanda (system python3 maneja randomTrips, como en las corridas previas)
python3 "$HERE/generate_b1_demand.py" --ratio 5 --scale "$SCALE" --seed "$SEED" --outdir "$WORK" >/dev/null 2>&1
# 2) ruteo
( cd "$WORK" && duarouter -n "$NET" --route-files "$(cat "$WORK/trip_files.txt")" \
    -o "$ROU" --alternatives-output NUL --seed "$SEED" --ignore-errors --no-step-log >/dev/null 2>&1 )
# 3) sim 24h (seed a sumo para variar la dinámica)
sumo -n "$NET" -r "$ROU" --additional-files "$WORK/edgedata_local.add.xml" \
  -b 0 -e 86400 --seed "$SEED" --time-to-teleport 300 --collision.action warn \
  --statistic-output "$WORK/stats.xml" --duration-log.statistics --no-step-log \
  --log "$WORK/sumo.log" >/dev/null 2>&1
# 4) compactar a Parquet (venv del proyecto: pandas + pyarrow)
"$VENV" "$HERE/compact_day.py" "$EDGEOUT" "$PARQUET" "$NET"
# 5) limpiar: borrar XML pesado + intermedios; conservar XML crudo solo si keepxml
if [ "$KEEPXML" = "keepxml" ]; then
  mv "$EDGEOUT" "$DS/SAMPLE_edgedata_seed${SEEDP}.xml"
  cp "$WORK/stats.xml" "$DS/SAMPLE_stats_seed${SEEDP}.xml"
fi
rm -rf "$WORK"
echo "[seed $SEED] LISTO -> $PARQUET"
