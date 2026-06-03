#!/usr/bin/env bash
# build_miraflores_net.sh — Reconstrucción del net de Miraflores cubriendo el DISTRITO
# completo (Camino B1). Espejo de build_corredor_larco.sh, adaptado a un bbox amplio.
#
# CONTEXTO: el net en uso conf/network/miraflores.net.xml es un recorte OSM incompleto
# (receta mínima --proj.utm, sin typemap / sin --tls.guess-signals / sin --junctions.join /
# sin --output.street-names, sobre un map.osm fuente PERDIDO). Diagnóstico: solo 3/12
# intersecciones del gazetteer PMU mapean a un semáforo modelado (avenidas Angamos /
# 28 de Julio / Paseo de la República ausentes). Este script construye un net NUEVO con
# nombre DISTINTO (miraflores_v2.net.xml) usando la receta "bien hecha" del repo, SIN tocar
# el net viejo ni su cadena (dataset / tensor / mapping / congestión).
#
# ┌───────────────────────────────────────────────────────────────────────────┐
# │ El insumo OSM se CONSERVA (miraflores_completo.osm), no es efímero — para    │
# │ no repetir el pecado del map.osm perdido. El script es reproducible: con     │
# │ --no-download reusa el .osm versionado en vez de volver a pegarle a la API.  │
# └───────────────────────────────────────────────────────────────────────────┘
#
# Uso (desde simulation/):
#   bash scripts/build_miraflores_net.sh               # descarga OSM + netconvert
#   bash scripts/build_miraflores_net.sh --no-download  # reusa el .osm versionado
set -euo pipefail

# --- Ubicación -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$SIM_DIR/conf/network"
OSM_FILE="$OUT_DIR/miraflores_completo.osm"
NET_FILE="$OUT_DIR/miraflores_v2.net.xml"

# --- Bounding box (distrito completo: cubre las 12 intersecciones PMU) ------
# Formato: minlon,minlat,maxlon,maxlat (oeste,sur,este,norte).
# Cubre Angamos (norte -12.108), 28 de Julio (sur -12.131), Paseo/Panamá (este -77.017)
# y conserva el borde oeste actual (-77.0381). Margen 110-440 m sobre el gazetteer PMU.
BBOX="-77.0381,-12.1310,-77.0170,-12.1080"

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

# --- 1) Descarga OSM (OSM main API, recorte por bbox) ----------------------
# Se usa la OSM main API /0.6/map: devuelve los datos COMPLETOS recortados al bbox
# (nodos dentro del bbox + las ways que los referencian), limpios. Requiere User-Agent.
# NOTA: overpass-api.de/api/map devuelve HTTP 406 — no usar esa ruta (ya documentado en
# build_corredor_larco.sh). Alternativa equivalente: python "$SUMO_HOME/tools/osmGet.py".
UA="CerebroVial-thesis/1.0 (net Miraflores distrito completo - B1)"
if [[ "${1:-}" != "--no-download" ]]; then
  echo "Descargando OSM (OSM main API) bbox=$BBOX → $OSM_FILE"
  curl -fsS -A "$UA" "https://api.openstreetmap.org/api/0.6/map?bbox=$BBOX" -o "$OSM_FILE"
fi

# --- 2) Validación del OSM antes de netconvert (robustez) ------------------
test -s "$OSM_FILE" || { echo "ERROR: .osm vacío o inexistente ($OSM_FILE)"; exit 1; }
grep -q "<node " "$OSM_FILE" || { echo "ERROR: .osm sin <node> (¿respuesta de error de la API?)"; exit 1; }
if grep -qiE "<remark>|rate_limited|runtime error" "$OSM_FILE"; then
  echo "ERROR: la API devolvió un mensaje de error en lugar del mapa:"; grep -iE "<remark>|error" "$OSM_FILE" | head -3; exit 1
fi
echo "OSM válido: $(grep -c "<node " "$OSM_FILE") nodos, $(wc -c < "$OSM_FILE") bytes"

# --- 3) netconvert (receta completa, igual que corredor Larco) -------------
# --type-files             : typemap OSM estándar (clases viales → el viejo net NO lo usó)
# --proj.utm               : UTM zona 18 (sin +south; netOffset desplaza al origen local)
# --geometry.remove        : colapsa geometría redundante
# --remove-edges.isolated  : descarta edges colgadas/aisladas (salud del grafo)
# --keep-edges.by-vclass passenger : red vehicular
# --tls.guess-signals      : reubica nodos OSM traffic_signals sobre las entrantes (PRESERVA, no inventa)
# --tls.join false         : TLS independientes (la coordinación la hace el motor de control)
# --junctions.join         : colapsa avenidas divididas en cruces únicos (join-dist 10 m)
# --no-turnarounds true    : sin retornos en U
# --output.street-names / --output.original-names : preserva `name` (el viejo net lo tenía vacío)
# NO se usa --tls.guess (inventaría semáforos donde OSM no los tiene).
# NO se usa --remove-edges.explicit (eso era saneamiento manual del corredor de 3 cruces).
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
  --output.street-names true --output.original-names true

echo "Listo: $NET_FILE"
echo "Validar cobertura PMU:  .venv/bin/python scripts/validate_pmu_coverage.py \\"
echo "    --net conf/network/miraflores_v2.net.xml \\"
echo "    --out ../documentation/contracts/mapeo_pmu_edges_v2.yaml --compare conf/network/miraflores.net.xml"
