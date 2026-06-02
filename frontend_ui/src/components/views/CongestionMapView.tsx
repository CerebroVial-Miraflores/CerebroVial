/**
 * Mapa de congestión de red (HU-22) — render + feed vivo.
 *
 * Monta un mapa Leaflet propio (NO reusa el de DashboardView, que arrastra estado
 * de KPIs/cámaras/SSE de visión), carga geometría + estado al montar, los cruza con
 * `mergeCongestion` y pinta los 375 tramos por una capa <GeoJSON> coloreada por
 * nivel (CA-22.1/22.3). Consulta puntual por tramo vía hover/click (CA-22.5) y
 * leyenda 0-5.
 *
 * Decisión firme: <GeoJSON> con `style` callback, NO <Polyline>. El endpoint
 * devuelve coordenadas [lon, lat] (GeoJSON estándar) que <GeoJSON> interpreta
 * nativo; <Polyline> obligaría a invertir las 375 geometrías a [lat,lng] a mano.
 *
 * FEED VIVO (Fase 3, CA-22.2): se abre un stream SSE de congestión; cada wake
 * re-lee SOLO el estado (la geometría es estática, NO se re-pide) y recolorea.
 *
 * Vía de recolorización — REMONTE POR `key` (no `setStyle` imperativo): react-leaflet 5
 * NO refresca la prop `data` de <GeoJSON> en cambios de estado (la capa L.GeoJSON
 * cachea los features iniciales). Cambiar el `useState` de features NO recolorea.
 * Por eso la <GeoJSON> lleva una `key` que cambia en cada update; React remonta la
 * capa con los datos nuevos. El remonte re-ejecuta `onEachFeature`, así que recolor,
 * tooltip/popup (CA-22.5) y atenuado stale se refrescan juntos, sin la trampa de
 * propiedades stale de `layer.setStyle`. A ~60 s de cadencia el costo de remontar
 * 375 LineStrings es despreciable.
 *
 * STALE (CA-22.4): si no llega un wake en 90 s, se atenúan los tramos (opacidad 50%,
 * manteniendo color+grosor) y aparece un banner "datos de hace X". Un wake posterior
 * lo limpia.
 *
 * Fuera de alcance: el componente todavía NO se cablea como tab en la navegación
 * (Sidebar/App, Fase 4).
 */
import { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import type { Layer, PathOptions } from 'leaflet';
import type { Feature, GeoJsonObject } from 'geojson';
import { AlertTriangle, Clock } from 'lucide-react';
import { LoadingOverlay } from '../ui/LoadingStates';
import { congestionService } from '../../services/congestionService';
import { openCongestionStream } from '../../services/congestionSseClient';
import { mergeCongestion, congestionStyle, elapsedSeconds } from '../../utils/congestion';
import type {
  MergedCongestionFeature,
  GeometryFeatureCollection,
  CongestionStateResponse,
} from '../../types/congestion';

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

/** Sin wake en este lapso → estado desactualizado (CA-22.4). 1.5× la cadencia de 60 s. */
const STALE_TIMEOUT_MS = 90_000;
/** Opacidad de stroke de los tramos en estado stale: atenúa sin perder color+grosor. */
const STALE_OPACITY = 0.5;

/** Último (más reciente) snapshot_timestamp del estado, o null si no hay aristas. */
function latestSnapshot(state: CongestionStateResponse): string | null {
  return state.edges.reduce<string | null>(
    (max, e) => (max === null || e.snapshot_timestamp > max ? e.snapshot_timestamp : max),
    null,
  );
}

/** Formatea segundos transcurridos como "X s" o "M min S s" para el banner stale. */
function formatAge(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s`;
  const min = Math.floor(s / 60);
  const rem = s % 60;
  return rem === 0 ? `${min} min` : `${min} min ${rem} s`;
}

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
  // Feed vivo (Fase 3): stale + datos para el banner + key de remonte.
  const [stale, setStale] = useState(false);
  const [latestTs, setLatestTs] = useState<string | null>(null);
  const [now, setNow] = useState<Date>(() => new Date());
  // Sube en cada update; junto a `stale` forma la key que fuerza el remonte de
  // <GeoJSON> (vía de recolorización elegida).
  const [renderSeq, setRenderSeq] = useState(0);
  // Geometría estática cacheada: se cruza con cada estado nuevo SIN re-pedirla.
  const geometryRef = useRef<GeometryFeatureCollection | null>(null);

  useEffect(() => {
    let cancelled = false;
    let staleTimer: number | null = null;

    // (Re)arma el timer de stale: si no llega otro wake en STALE_TIMEOUT_MS,
    // marca el estado como desactualizado (CA-22.4).
    const armStale = () => {
      if (staleTimer !== null) window.clearTimeout(staleTimer);
      staleTimer = window.setTimeout(() => {
        if (cancelled) return;
        setNow(new Date());
        setStale(true);
      }, STALE_TIMEOUT_MS);
    };

    // Aplica un estado fresco: cruza con la geometría cacheada, recolorea
    // (bump de renderSeq → remonte), sale de stale y re-arma el timer.
    const applyState = (state: CongestionStateResponse) => {
      const geometry = geometryRef.current;
      if (!geometry) return;
      setFeatures(mergeCongestion(geometry, state));
      setLatestTs(latestSnapshot(state));
      setStale(false);
      setRenderSeq((s) => s + 1);
      armStale();
    };

    // Carga inicial: geometría (estática) + estado, 1× cada uno al montar.
    const init = async () => {
      try {
        const [geometry, state] = await Promise.all([
          congestionService.getGeometry(),
          congestionService.getState(),
        ]);
        if (cancelled) return;
        geometryRef.current = geometry;
        applyState(state);
      } catch (err) {
        if (cancelled) return;
        console.error('Error cargando mapa de congestión:', err);
        setError('No se pudo cargar el mapa de congestión de red.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    // Wake del feed: re-lee SOLO el estado (la geometría NO se re-pide) y recolorea.
    const onWake = async () => {
      try {
        const state = await congestionService.getState();
        if (cancelled) return;
        applyState(state);
      } catch (err) {
        // Falla de re-lectura: conservamos el último estado. El staleTimer ya
        // armado atenuará si no llegan más wakes.
        if (cancelled) return;
        console.error('Error re-leyendo estado de congestión:', err);
      }
    };

    void init();
    const controller = openCongestionStream({ onWake: () => void onWake() });

    return () => {
      cancelled = true;
      if (staleTimer !== null) window.clearTimeout(staleTimer);
      controller.abort();
    };
  }, []);

  // Mientras stale: tick de 1 s para refrescar el contador "hace X" del banner.
  useEffect(() => {
    if (!stale) return;
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, [stale]);

  // <GeoJSON data> consume un GeoJsonObject: envolvemos las features cruzadas.
  const data = { type: 'FeatureCollection', features } as unknown as GeoJsonObject;

  // style callback: {color, weight} de congestionStyle (opciones de path válidas
  // de Leaflet). En stale, atenúa la opacidad de stroke manteniendo color+grosor.
  const styleFeature = (feature?: Feature): PathOptions => {
    const base = congestionStyle(
      (feature?.properties as { congestion_level?: number | null })?.congestion_level,
    );
    return stale ? { ...base, opacity: STALE_OPACITY } : base;
  };

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
            {/* key cambia por update (renderSeq) y al togglear stale → remonta la
                capa y recolorea (react-leaflet 5 no refresca `data` in-place). */}
            <GeoJSON
              key={`${renderSeq}-${stale ? 'stale' : 'live'}`}
              data={data}
              style={styleFeature}
              onEachFeature={onEachFeature}
            />
            <CongestionLegend />
          </MapContainer>
        )}
        {/* Banner stale (CA-22.4): datos desactualizados → "datos de hace X". */}
        {!loading && !error && stale && latestTs && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-[500] flex items-center gap-2 bg-amber-500/95 text-slate-900 font-semibold px-4 py-2 rounded-full shadow-lg">
            <Clock size={16} />
            <span className="text-sm">
              Datos de hace {formatAge(elapsedSeconds(latestTs, now))}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
