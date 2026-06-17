import { useMemo, useRef } from 'react';
import { Map as MapGL, Source, Layer, type MapRef } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

import { Button } from '../../ui/Button';
import { StatusChip } from '../../ui/StatusChip';
import { useCongestionGeometry } from '../../../hooks/useCongestionGeometry';

// FASE 0 migración MapLibre — SPIKE de plumbing, SOLO DEV (chunk lazy de UiLabView).
//
// Objetivo único: validar la cañería MapLibre GL JS + react-map-gl de extremo a
// extremo — que el estilo vectorial cargue, que las 1660 aristas REALES de
// GET /congestion/geometry se dibujen sobre Miraflores, y que un flyTo navegue
// hasta la intersección crítica. NO valida color (uniforme/placeholder), NO dark,
// NO recolor por jam_level (eso es Fase 1+). Reusa el hook useCongestionGeometry
// (mismo GET con JWT + cache de sesión) en vez de re-fetchear.

// Estilo claro OpenFreeMap "liberty" — solo para validar plumbing. El dark / style
// propio es iteración posterior, fuera de este spike.
const LIBERTY_STYLE = 'https://tiles.openfreemap.org/styles/liberty';

// Vista GENERAL de Miraflores (pitch 0): el flyTo arranca acá para que sea un
// viaje observable (paneo + zoom + tilt), no un acercamiento en el lugar.
const MIRAFLORES_OVERVIEW = {
  longitude: -77.035,
  latitude: -12.115,
  zoom: 13.3,
  pitch: 0,
  bearing: 0,
} as const;

// Intersección crítica Av. Larco × Av. Benavides — coords reales del nodo
// `larco_benavides` (scripts/seed.py:59), en orden GeoJSON [lon, lat].
const LARCO_BENAVIDES: [number, number] = [-77.0301, -12.1227];

// Color uniforme PLACEHOLDER: el punto del spike es que las 1660 aristas carguen
// y se vean sobre Miraflores, NO el color. El recolor data-driven es Fase 1.
const PLACEHOLDER_LINE_COLOR = '#2563eb';

export function MapLibreSpike() {
  const mapRef = useRef<MapRef | null>(null);
  const geometry = useCongestionGeometry();

  // FeatureCollection lista para el source GeoJSON. Las features del hook ya son
  // GeoJSON estándar (LineString [lon,lat] EPSG:4326), se pasan tal cual.
  const featureCollection = useMemo(
    () =>
      geometry.data
        ? { type: 'FeatureCollection' as const, features: geometry.data.features }
        : null,
    [geometry.data],
  );

  const featureCount = geometry.data?.features.length ?? null;

  const flyToCritical = () => {
    mapRef.current?.flyTo({
      center: LARCO_BENAVIDES,
      zoom: 16,
      pitch: 55,
      bearing: -20,
      duration: 2800,
    });
  };

  return (
    <div className="space-y-4">
      <p className="text-[12px] leading-relaxed text-ink-2">
        Spike de plumbing MapLibre (Fase 0). Consume el backend REAL (core 8001 vía httpClient
        con JWT) — requiere el motor levantado (<span className="num">invoke up</span>) y rol
        operator/admin. Estilo claro <span className="num">liberty</span> de OpenFreeMap, color de
        arista uniforme placeholder. Sin recolor, sin markers, sin dark: eso es Fase 1+.
      </p>

      <div className="flex flex-wrap items-center gap-3">
        <Button variant="pri" onClick={flyToCritical}>
          ✈ Volar a Larco × Benavides
        </Button>
        {geometry.loading && <span className="text-[12px] text-ink-2">Cargando geometría…</span>}
        {featureCount !== null && (
          <StatusChip status={featureCount === 1660 ? 'ok' : 'warn'}>
            {featureCount} tramos cargados
          </StatusChip>
        )}
      </div>

      {geometry.error && (
        <div className="space-y-2 rounded-ctl border border-bad/40 bg-bad/10 p-3">
          <p className="text-[12px] leading-relaxed text-bad">{geometry.error}</p>
          <Button onClick={() => void geometry.refetch()}>Reintentar</Button>
        </div>
      )}

      <div className="relative h-[600px] w-full overflow-hidden rounded-panel border border-line">
        <MapGL
          ref={mapRef}
          mapStyle={LIBERTY_STYLE}
          initialViewState={MIRAFLORES_OVERVIEW}
          style={{ width: '100%', height: '100%' }}
        >
          {featureCollection && (
            <Source id="network-geometry" type="geojson" data={featureCollection}>
              <Layer
                id="network-geometry-lines"
                type="line"
                paint={{
                  'line-color': PLACEHOLDER_LINE_COLOR,
                  'line-width': 2,
                  'line-opacity': 0.85,
                }}
              />
            </Source>
          )}
        </MapGL>
      </div>

      <p className="text-[11px] text-ink-3">
        Criterio de cierre observacional: basemap claro carga, se ven los 1660 tramos sobre
        Miraflores con pitch, y el botón hace un viaje de cámara (paneo + zoom + tilt) hasta la
        intersección. El chip confirma el conteo real (1660, no 375).
      </p>
    </div>
  );
}
