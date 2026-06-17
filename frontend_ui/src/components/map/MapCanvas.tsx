import type { ReactNode } from 'react';
import { Map as MapGL, type MapMouseEvent } from 'react-map-gl/maplibre';
import type { StyleSpecification } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

// FASE 4 migración MapLibre — wrapper único del mapa. Envuelve <Map> de
// react-map-gl en vez del <MapContainer> de Leaflet, preservando la superficie
// de props (center sigue en [lat,lng] y se convierte a [lng,lat] adentro). El
// basemap se mantiene CartoDB dark, ahora como raster source de MapLibre (mismas
// tiles; el {s}/{r} de Leaflet se traduce a sharding por subdominio a-d, sin
// retina) — preserva el look sin meter estética nueva (el style vector propio es
// fase posterior). `isolate` sigue siendo obligatorio: contiene el stacking del
// mapa por debajo de Drawer (z 90) y Modal (z 100). Altura por flex (flex-1
// min-h-0), jamás calc(100vh-*).

const CARTO_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>';

const CARTO_DARK_STYLE: StyleSpecification = {
  version: 8,
  sources: {
    'carto-dark': {
      type: 'raster',
      tiles: ['a', 'b', 'c', 'd'].map(
        (s) => `https://${s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png`,
      ),
      tileSize: 256,
      attribution: CARTO_ATTRIBUTION,
    },
  },
  layers: [{ id: 'carto-dark', type: 'raster', source: 'carto-dark' }],
};

interface MapCanvasProps {
  center: [number, number];
  zoom: number;
  children?: ReactNode;
  /** Slot inferior-izquierdo (MapLegend). */
  legend?: ReactNode;
  /** Slot superior-derecho (MapModeBadge). */
  modeBadge?: ReactNode;
  /** Slot inferior-centro (tip flotante). */
  coachTip?: ReactNode;
  className?: string;
  /** Estilo del mapa (default: raster CartoDB dark). */
  mapStyle?: string | StyleSpecification;
  /** Capas interactivas para hover/click (p.ej. el tooltip de arista). */
  interactiveLayerIds?: string[];
  onMouseMove?: (e: MapMouseEvent) => void;
  onMouseLeave?: (e: MapMouseEvent) => void;
  /** Cursor del mapa (p.ej. 'pointer' sobre una arista interactiva). */
  cursor?: string;
}

export function MapCanvas({
  center,
  zoom,
  children,
  legend,
  modeBadge,
  coachTip,
  className = '',
  mapStyle = CARTO_DARK_STYLE,
  interactiveLayerIds,
  onMouseMove,
  onMouseLeave,
  cursor,
}: MapCanvasProps) {
  return (
    <div
      className={`relative isolate min-h-0 flex-1 overflow-hidden rounded-card border border-line ${className}`}
    >
      <MapGL
        mapStyle={mapStyle}
        initialViewState={{ longitude: center[1], latitude: center[0], zoom }}
        style={{ width: '100%', height: '100%' }}
        interactiveLayerIds={interactiveLayerIds}
        onMouseMove={onMouseMove}
        onMouseLeave={onMouseLeave}
        cursor={cursor}
      >
        {children}
      </MapGL>
      {legend != null && <div className="absolute bottom-[13px] left-[13px] z-[500]">{legend}</div>}
      {modeBadge != null && <div className="absolute right-[13px] top-3 z-[500]">{modeBadge}</div>}
      {coachTip != null && (
        <div className="absolute bottom-4 left-1/2 z-[500] -translate-x-1/2">{coachTip}</div>
      )}
    </div>
  );
}
