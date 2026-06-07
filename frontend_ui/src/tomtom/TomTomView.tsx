/**
 * Vista EXPERIMENTAL de tráfico en vivo TomTom (track feature/tomtom, Fase A).
 *
 * Validación visual: SOLO se mira. Front puro, cero backend, cero persistencia.
 * Pinta la capa raster de Traffic Flow de TomTom sobre un basemap OSM propio.
 * NO mide, NO consulta flowSegmentData, NO calcula KPIs (eso es Fase B, fuera de
 * alcance). NO extrae datos de los tiles (ToS 11.6.1) — los tiles son solo para
 * pintar. La atribución TomTom es obligatoria y visible (ToS 17.3).
 *
 * MapContainer PROPIO (no reusa el de CongestionMapView). center/zoom sobre
 * Miraflores, coherentes con el resto del front.
 */
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { Navigation, AlertTriangle } from 'lucide-react';

import { TomTomFlowLayer } from './TomTomFlowLayer';
import { TomTomAttribution } from './TomTomAttribution';
import type { TomTomFlowStyle } from './types';

// Centro/zoom de Miraflores (idénticos a CongestionMapView, coherencia de encuadre).
const MIRAFLORES_CENTER: [number, number] = [-12.122, -77.028];
const DEFAULT_ZOOM = 14;

// Fase A: estilo de tile de diseño para UI oscura (decisión de producto, 1 línea).
const FLOW_STYLE: TomTomFlowStyle = 'relative0-dark';

const HAS_KEY = Boolean(import.meta.env.VITE_TOMTOM_KEY);

/** Escala relativa del flujo (relativo al free-flow). Solo lectura, leyenda visual. */
const FLOW_LEGEND: readonly { color: string; label: string }[] = [
  { color: '#2ECC71', label: 'Fluido' },
  { color: '#F4C20D', label: 'Lento' },
  { color: '#E24B4A', label: 'Congestionado' },
  { color: '#8E1B1B', label: 'Detenido / cortado' },
];

export const TomTomView = () => {
  return (
    <div className="relative w-full h-[calc(100vh-7rem)] min-h-[600px] flex flex-col">
      <div className="relative flex-1 rounded-xl overflow-hidden border border-slate-700 bg-slate-900">
        <MapContainer
          center={MIRAFLORES_CENTER}
          zoom={DEFAULT_ZOOM}
          style={{ height: '100%', width: '100%' }}
          zoomControl={false}
        >
          {/* Basemap OSM, igual al resto del front. */}
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {/* Capa de tráfico TomTom encima del basemap. Si no hay key, no monta. */}
          <TomTomFlowLayer style={FLOW_STYLE} />
        </MapContainer>

        {/* Título / rótulo de la vista (esquina superior izquierda). */}
        <div className="absolute top-4 left-4 z-[500] flex flex-col gap-1 rounded-xl bg-slate-900/90 backdrop-blur border border-slate-700 px-4 py-3 shadow-lg">
          <div className="flex items-center gap-2">
            <Navigation size={16} className="text-indigo-400" />
            <span className="text-sm font-bold text-white">Tráfico en vivo</span>
            <span className="rounded-full bg-indigo-500/20 text-indigo-300 text-[10px] font-semibold px-2 py-0.5 border border-indigo-500/30">
              EXPERIMENTAL
            </span>
          </div>
          <span className="text-xs text-slate-400">
            Capa de TomTom Traffic Flow sobre Miraflores
          </span>
        </div>

        {/* Aviso de degradación cuando falta la display key (no rompe la vista). */}
        {!HAS_KEY && (
          <div className="absolute top-4 right-4 z-[500] flex items-start gap-2 max-w-xs rounded-xl bg-amber-500/10 border border-amber-500/40 px-4 py-3 shadow-lg">
            <AlertTriangle size={16} className="text-amber-400 mt-0.5 shrink-0" />
            <span className="text-xs text-amber-200">
              <b>VITE_TOMTOM_KEY</b> no configurada. Se muestra solo el mapa base;
              la capa de tráfico de TomTom no se carga.
            </span>
          </div>
        )}

        {/* Leyenda de la escala de flujo (solo lectura). */}
        <div className="absolute bottom-4 left-4 z-[500] rounded-xl bg-slate-900/90 backdrop-blur border border-slate-700 px-4 py-3 shadow-lg">
          <span className="block text-[11px] font-semibold text-slate-300 mb-2">
            Velocidad relativa
          </span>
          <ul className="flex flex-col gap-1.5">
            {FLOW_LEGEND.map((item) => (
              <li key={item.label} className="flex items-center gap-2">
                <span
                  className="inline-block w-6 h-1.5 rounded-full"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-xs text-slate-400">{item.label}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Atribución TomTom obligatoria y visible (ToS 17.3) — no removible. */}
        <TomTomAttribution />
      </div>
    </div>
  );
};
