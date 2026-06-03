#!/usr/bin/env bash
# Dataset laborable 60d — genera+rutea+simula+compacta UN día (perfil laborable,
# scale=0.20). seed -> randomTrips, duarouter Y sumo (varía ruteo y dinámica).
# Compacta a Parquet; persiste stats.xml (durable) + edgedata.xml crudo (efímero, insumo
# del evaluador de drenaje D-014). DESCARTABLE/scratch. NO conf/. NO commits.
# Uso: gen_day.sh <seed>
set -euo pipefail
SEED="$1"
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
# 5) persistir insumos del evaluador de drenaje (D-014, B3.2.a) antes de limpiar:
#    stats.xml DURABLE (chico: teleports + duración media + auditoría posterior);
#    edgedata.xml crudo EFÍMERO (señal dip ponderada por sampledSeconds; el evaluador
#    batch lo consume y lo borra). compact_day.py ya leyó el edgedata arriba.
cp "$WORK/stats.xml" "$DS/stats_seed${SEEDP}.xml"
mv "$EDGEOUT" "$DS/edgedata_seed${SEEDP}.xml"
# 6) limpiar intermedios (rou, trips, add-file, log, stats original)
rm -rf "$WORK"
echo "[seed $SEED] LISTO -> $PARQUET"
