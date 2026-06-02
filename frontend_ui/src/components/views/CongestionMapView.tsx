/**
 * Mapa de congestión de red (HU-22, Fase 2 — render estático).
 *
 * Monta un mapa Leaflet propio (NO reusa el de DashboardView, que arrastra estado
 * de KPIs/cámaras/SSE de visión), carga geometría + estado UNA sola vez al montar,
 * los cruza con `mergeCongestion` y pinta los 375 tramos por una capa <GeoJSON>
 * coloreada por nivel (CA-22.1/22.3). Consulta puntual por tramo vía hover/click
 * (CA-22.5) y leyenda 0-5.
 *
 * Decisión firme: <GeoJSON> con `style` callback, NO <Polyline>. El endpoint
 * devuelve coordenadas [lon, lat] (GeoJSON estándar) que <GeoJSON> interpreta
 * nativo; <Polyline> obligaría a invertir las 375 geometrías a [lat,lng] a mano.
 * El `style` callback recolorea por nivel sin re-montar la capa (lo aprovecha Fase 3).
 *
 * Fuera de alcance (Fase 2): NO hay feed vivo (SSE/wake/stale es Fase 3) y el
 * componente todavía NO se cablea como tab en la navegación (Sidebar/App, Fase 4).
 */
import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import type { Layer } from 'leaflet';
import type { Feature, GeoJsonObject } from 'geojson';
import { AlertTriangle } from 'lucide-react';
import { LoadingOverlay } from '../ui/LoadingStates';
import { congestionService } from '../../services/congestionService';
import { mergeCongestion, congestionStyle } from '../../utils/congestion';
import type { MergedCongestionFeature } from '../../types/congestion';

/** Niveles 0-5 de la escala aprobada (CA-22.3) para iterar en la leyenda. */
const LEVELS = [0, 1, 2, 3, 4, 5] as const;
const LEVEL_LABEL: Record<number, string> = {
  0: 'Flujo libre',
  1: 'Leve',
  2: 'Moderado',
  3: 'Considerable',
  4: 'Alto',
  5: 'Vía cerrada',
};

/**
 * Leyenda 0-5: muestra la codificación de `congestionStyle` (color + grosor real),
 * más la entrada neutra "sin dato". Es la referencia visual de CA-22.1/22.3.
 */
const CongestionLegend = () => (
  <div className="absolute bottom-4 left-4 z-[400] bg-slate-900/90 backdrop-blur border border-slate-700 rounded-xl px-4 py-3 shadow-lg">
    <h4 className="text-xs font-bold text-white mb-2 uppercase tracking-wide">
      Nivel de congestión
    </h4>
    <ul className="flex flex-col gap-1.5">
      {LEVELS.map((level) => {
        const { color, weight } = congestionStyle(level);
        return (
          <li key={level} className="flex items-center gap-2">
            <span
              className="inline-block w-6 rounded-full"
              style={{ backgroundColor: color, height: weight }}
            />
            <span className="text-[11px] text-slate-300">
              {level} · {LEVEL_LABEL[level]}
            </span>
          </li>
        );
      })}
      <li className="flex items-center gap-2">
        <span
          className="inline-block w-6 rounded-full"
          style={{ backgroundColor: congestionStyle(null).color, height: congestionStyle(null).weight }}
        />
        <span className="text-[11px] text-slate-400 italic">sin dato</span>
      </li>
    </ul>
  </div>
);

export const CongestionMapView = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [features, setFeatures] = useState<MergedCongestionFeature[]>([]);

  useEffect(() => {
    const load = async () => {
      try {
        const [geometry, state] = await Promise.all([
          congestionService.getGeometry(), // estática, 1× al montar
          congestionService.getState(), // último snapshot, 1× al montar
        ]);
        setFeatures(mergeCongestion(geometry, state));
      } catch (err) {
        console.error('Error cargando mapa de congestión:', err);
        setError('No se pudo cargar el mapa de congestión de red.');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  // <GeoJSON data> consume un GeoJsonObject: envolvemos las features cruzadas.
  const data = { type: 'FeatureCollection', features } as unknown as GeoJsonObject;

  // style callback: pasa {color, weight} de congestionStyle tal cual (son opciones
  // de path válidas de Leaflet). Recolorea por nivel sin re-montar (Fase 3).
  const styleFeature = (feature?: Feature) =>
    congestionStyle(
      (feature?.properties as { congestion_level?: number | null })?.congestion_level,
    );

  // CA-22.5 — consulta puntual: hover (tooltip) y click (popup) muestran edge_id + nivel.
  const onEachFeature = (feature: Feature, layer: Layer) => {
    const props = feature.properties as {
      edge_id: string;
      congestion_level: number | null;
    };
    const nivel = props.congestion_level ?? 'sin dato';
    layer.bindTooltip(`Tramo ${props.edge_id} — nivel ${nivel}`);
    layer.bindPopup(`<b>Tramo</b> ${props.edge_id}<br/><b>Nivel</b> ${nivel}`);
  };

  return (
    <div className="relative w-full h-full min-h-[600px] flex flex-col">
      <div className="relative flex-1 min-h-[600px] rounded-xl overflow-hidden border border-slate-700 bg-slate-900">
        {loading && <LoadingOverlay message="Cargando mapa de congestión..." />}
        {error && (
          <div className="absolute inset-0 z-[500] flex flex-col items-center justify-center bg-slate-900/90 text-white p-6 text-center">
            <AlertTriangle size={48} className="text-rose-500 mb-4" />
            <h3 className="text-xl font-bold mb-2">Error de Conexión</h3>
            <p className="text-slate-400 mb-6 max-w-md">{error}</p>
            <button
              onClick={() => window.location.reload()}
              className="bg-indigo-600 px-6 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
            >
              Reintentar
            </button>
          </div>
        )}
        {!loading && !error && (
          <MapContainer
            center={[-12.122, -77.028]}
            zoom={14}
            style={{ height: '100%', width: '100%' }}
            zoomControl={false}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            <GeoJSON data={data} style={styleFeature} onEachFeature={onEachFeature} />
            <CongestionLegend />
          </MapContainer>
        )}
      </div>
    </div>
  );
};
