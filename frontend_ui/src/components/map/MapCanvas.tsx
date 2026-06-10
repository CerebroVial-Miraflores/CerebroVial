import type { ReactNode } from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import './map.css';

// FASE 1 rediseño UI — wrapper único del mapa del rediseño. Decisión cerrada:
// TileLayer CartoDB dark (las vistas v1 siguen en OSM hasta su fase). Altura
// SIEMPRE por flex del contenedor (flex-1 min-h-0) — jamás calc(100vh-*).
// `isolate` es obligatorio: los panes internos de Leaflet usan z-index 200-700
// y sin un stacking context propio pintarían sobre Drawer (z 90) y Modal (z 100).

const CARTO_DARK_URL = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
const CARTO_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>';

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
}

export function MapCanvas({ center, zoom, children, legend, modeBadge, coachTip, className = '' }: MapCanvasProps) {
  return (
    <div
      className={`relative isolate min-h-0 flex-1 overflow-hidden rounded-card border border-line ${className}`}
    >
      <MapContainer center={center} zoom={zoom} className="h-full w-full">
        <TileLayer url={CARTO_DARK_URL} attribution={CARTO_ATTRIBUTION} />
        {children}
      </MapContainer>
      {legend != null && <div className="absolute bottom-[13px] left-[13px] z-[500]">{legend}</div>}
      {modeBadge != null && <div className="absolute right-[13px] top-3 z-[500]">{modeBadge}</div>}
      {coachTip != null && (
        <div className="absolute bottom-4 left-1/2 z-[500] -translate-x-1/2">{coachTip}</div>
      )}
    </div>
  );
}
