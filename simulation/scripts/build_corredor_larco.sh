#!/usr/bin/env bash
# build_corredor_larco.sh — Importación INICIAL de la red base del corredor Av. Larco.
#
# DHU-027 (2026-05-29). Genera la red SUMO de los 3 cruces consecutivos de Av. José
# Larco, Miraflores (Diez Canseco, Schell, Benavides) desde geometría REAL de
# OpenStreetMap. Camino de build PARALELO al de la topología genérica de TTH-07
# (scripts/build_network.py) — no lo reemplaza.
#
# ┌───────────────────────────────────────────────────────────────────────────┐
# │ CICLO DE VIDA DEL .net.xml — LEER ANTES DE RE-EJECUTAR                       │
# │                                                                             │
# │ Este script reproduce SOLO la importación inicial OSM → .net.xml. Una vez   │
# │ que la red se edite en netedit (corrección de semáforos / sentidos /        │
# │ carriles tras el gate de revisión visual), conf/corredor_larco/             │
# │ corredor_larco.net.xml pasa a ser FUENTE DE VERDAD CURADA y NO regenerable  │
# │ desde aquí: RE-EJECUTAR ESTE SCRIPT LO PISARÍA y borraría las correcciones  │
# │ manuales. El script queda como referencia de procedencia, no como           │
# │ regenerador idempotente.                                                    │
# └───────────────────────────────────────────────────────────────────────────┘
#
# Uso (desde simulation/):
#   bash scripts/build_corredor_larco.sh            # descarga OSM + netconvert
#   bash scripts/build_corredor_larco.sh --no-download   # reusa el .osm versionado
set -euo pipefail

# --- Ubicación -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$SIM_DIR/conf/corredor_larco"
OSM_FILE="$OUT_DIR/corredor_larco.osm"
NET_FILE="$OUT_DIR/corredor_larco.net.xml"

# --- Bounding box (recortado: Óvalo/Parque Kennedy EXCLUIDO por DHU-027) ----
# Formato: minlon,minlat,maxlon,maxlat (oeste,sur,este,norte)
BBOX="-77.0302,-12.1254,-77.0282,-12.1220"

# --- SUMO_HOME -------------------------------------------------------------
if [[ -z "${SUMO_HOME:-}" ]]; then
  if command -v netconvert >/dev/null 2>&1; then
    # deriva SUMO_HOME del binario (…/bin/netconvert → …/share/sumo)
    SUMO_HOME="$(dirname "$(dirname "$(command -v netconvert)")")/share/sumo"
  fi
fi
TYPEMAP="$SUMO_HOME/data/typemap/osmNetconvert.typ.xml"
echo "SUMO_HOME=$SUMO_HOME"
[[ -f "$TYPEMAP" ]] || { echo "ERROR: no se encontró el typemap OSM en $TYPEMAP"; exit 1; }

mkdir -p "$OUT_DIR"

# --- 1) Descarga OSM (API principal de OSM, recorte por bbox) --------------
# Se usa la OSM main API: devuelve datos recortados al bbox, limpios y compactos
# (los nodos dentro del bbox + las ways que los referencian). Requiere User-Agent.
# Alternativas equivalentes:
#   B) Overpass interpreter:  curl -fsS -A "$UA" --data-urlencode \
#        'data=[out:xml][timeout:60];(node(SUR,OESTE,NORTE,ESTE);way(...);rel(...));out body;>;out skel qt;' \
#        https://overpass-api.de/api/interpreter -o "$OSM_FILE"
#      (trae más nodos por recursión de ways que cruzan el borde; menos limpio para este bbox.)
#   C) SUMO-native: python "$SUMO_HOME/tools/osmGet.py" --bbox $BBOX --prefix corredor_larco -d "$OUT_DIR"
# NOTA: overpass-api.de/api/map devuelve HTTP 406 — no usar esa ruta.
UA="CerebroVial-thesis/1.0 (DHU-027 corredor Larco)"
if [[ "${1:-}" != "--no-download" ]]; then
  echo "Descargando OSM (OSM main API) bbox=$BBOX → $OSM_FILE"
  curl -fsS -A "$UA" "https://api.openstreetmap.org/api/0.6/map?bbox=$BBOX" -o "$OSM_FILE"
fi

# --- 2) Validación del OSM antes de netconvert (robustez) ------------------
test -s "$OSM_FILE" || { echo "ERROR: .osm vacío o inexistente ($OSM_FILE)"; exit 1; }
grep -q "<node " "$OSM_FILE" || { echo "ERROR: .osm sin <node> (¿respuesta de error de Overpass?)"; exit 1; }
if grep -qiE "<remark>|rate_limited|runtime error" "$OSM_FILE"; then
  echo "ERROR: Overpass devolvió un mensaje de error en lugar del mapa:"; grep -iE "<remark>|error" "$OSM_FILE" | head -3; exit 1
fi
echo "OSM válido: $(grep -c "<node " "$OSM_FILE") nodos, $(wc -c < "$OSM_FILE") bytes"

# --- 3) netconvert (conservador: preservar semáforos importados, NO inventar)
# --tls.guess-signals : reubica nodos OSM traffic_signals sobre las edges entrantes (preserva, no inventa)
# --tls.join false    : mantiene los TLS del corredor INDEPENDIENTES (la coordinación la hace el motor)
# --junctions.join    : limpia la avenida dividida en cruces únicos (join-dist por defecto 10 m)
# --keep-edges.by-vclass passenger : red vehicular
# NO se usa --tls.guess (inventaría semáforos donde OSM no los tiene).
#
# --remove-edges.explicit : recorte de saneamiento posterior a la revisión visual (gate
#   DHU-027). Elimina geometría fuera del corredor de 3 cruces:
#     406008845#5, 344159559#0 → las dos calzadas de Av. Benavides MÁS ALLÁ de los 2 TLS
#       extra al este (~160 m de Larco); al quedar como dead-end, netconvert descarta esos
#       2 semáforos → la red queda con EXACTAMENTE 3 TLS (Diez Canseco, Schell, Benavides).
#       Se conserva ~145-150 m de approach de Benavides al cruce con Larco (edges 406008845#1
#       y 344159559#2, que NO se remueven).
#     39441587 → 313 m de Av. Ernesto Diez Canseco colgando hacia el este (fuera del corredor).
#   Los ramales cortos Pasaje Tello (406010997) y Cristóbal Colón (406007420) se CONSERVAN.
REMOVE_EDGES="406008845#5,344159559#0,39441587"
echo "Ejecutando netconvert → $NET_FILE"
netconvert \
  --osm-files "$OSM_FILE" \
  --type-files "$TYPEMAP" \
  --output-file "$NET_FILE" \
  --proj.utm \
  --geometry.remove --remove-edges.isolated \
  --keep-edges.by-vclass passenger \
  --tls.guess-signals --tls.join false \
  --junctions.join \
  --no-turnarounds true \
  --output.street-names true --output.original-names true \
  --remove-edges.explicit "$REMOVE_EDGES"

echo "Listo: $NET_FILE"
echo "Recordatorio: tras editar en netedit, este .net.xml es la fuente de verdad (no re-ejecutar el script)."
