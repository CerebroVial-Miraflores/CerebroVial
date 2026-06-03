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
import { useCallback, useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import type { Layer, PathOptions } from 'leaflet';
import type { Feature, GeoJsonObject } from 'geojson';
import { AlertTriangle, Clock, History, Radio } from 'lucide-react';
import { LoadingOverlay } from '../ui/LoadingStates';
import { congestionService } from '../../services/congestionService';
import { openCongestionStream } from '../../services/congestionSseClient';
import {
  mergeCongestion,
  mergeCongestionAtIndex,
  congestionStyle,
  elapsedSeconds,
} from '../../utils/congestion';
import type {
  MergedCongestionFeature,
  GeometryFeatureCollection,
  CongestionStateResponse,
  CongestionSeriesResponse,
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

/** Fecha de hoy en `YYYY-MM-DD` con componentes LOCALES (no UTC): es el default
 *  del date picker, debe coincidir con el "hoy" del usuario, no con UTC. */
function todayIsoLocal(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

/** Marcas de hora orientativas sobre la pista del slider (cada 3 h, 0..24). */
const HOUR_MARKS = [0, 3, 6, 9, 12, 15, 18, 21, 24] as const;

/**
 * Instante `t0 + i*stepS` formateado `HH:MM` (00:00–23:59). `t0` es naive-UTC
 * (mismo criterio que `normalizeToUtc` en utils): se parsea como UTC y se lee con
 * `getUTC*` para no desfasar por zona horaria local.
 */
function formatHourLabel(t0: string | null, stepS: number, i: number): string {
  if (!t0) return '--:--';
  const base = Date.parse(/[zZ]$|[+-]\d{2}:?\d{2}$/.test(t0) ? t0 : `${t0}Z`);
  const d = new Date(base + i * stepS * 1000);
  const hh = String(d.getUTCHours()).padStart(2, '0');
  const mm = String(d.getUTCMinutes()).padStart(2, '0');
  return `${hh}:${mm}`;
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
  // Modo de la vista (HU-23). El SSE/stale (Effect B) y la carga de serie
  // (Effect C) dependen de `mode`; la carga inicial de geometría (Effect A) NO.
  const [mode, setMode] = useState<'live' | 'historic'>('live');
  // Modo histórico (HU-23): día seleccionado, serie del día y posición del slider.
  const [day, setDay] = useState<string>(() => todayIsoLocal());
  const [series, setSeries] = useState<CongestionSeriesResponse | null>(null);
  const [sliderIndex, setSliderIndex] = useState(0);
  const [seriesLoading, setSeriesLoading] = useState(false);
  const [seriesError, setSeriesError] = useState<string | null>(null);
  // Geometría estática cacheada: se cruza con cada estado nuevo SIN re-pedirla.
  const geometryRef = useRef<GeometryFeatureCollection | null>(null);
  // Handle del timer de stale, elevado a ref para compartirlo entre la carga
  // inicial (Effect A, primer armado) y el feed vivo (Effect B, re-armado en
  // cada wake + limpieza en cleanup).
  const staleTimerRef = useRef<number | null>(null);
  // Distingue el mount inicial en 'live' (Effect A ya cargó el estado → NO re-leer)
  // del retorno a 'live' desde histórico (SÍ re-leer estado). Ver Effect B.
  const firstLiveRef = useRef(true);

  // (Re)arma el timer de stale: si no llega otro wake en STALE_TIMEOUT_MS, marca
  // el estado como desactualizado (CA-22.4). Estable: lo comparten ambos effects.
  const armStale = useCallback(() => {
    if (staleTimerRef.current !== null) window.clearTimeout(staleTimerRef.current);
    staleTimerRef.current = window.setTimeout(() => {
      setNow(new Date());
      setStale(true);
    }, STALE_TIMEOUT_MS);
  }, []);

  // Aplica un estado fresco: cruza con la geometría cacheada, recolorea (bump de
  // renderSeq → remonte), sale de stale y re-arma el timer. Estable: lo invocan
  // tanto la carga inicial (Effect A) como cada wake del feed (Effect B).
  const applyState = useCallback(
    (state: CongestionStateResponse) => {
      const geometry = geometryRef.current;
      if (!geometry) return;
      setFeatures(mergeCongestion(geometry, state));
      setLatestTs(latestSnapshot(state));
      setStale(false);
      setRenderSeq((s) => s + 1);
      armStale();
    },
    [armStale],
  );

  // Effect A — carga inicial: geometría (estática) + estado, 1× cada uno al
  // montar. NO depende de `mode`: la geometría se pide una sola vez pase lo que
  // pase con el modo (`applyState` es estable, así que este effect corre 1×).
  useEffect(() => {
    let cancelled = false;

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

    void init();

    return () => {
      cancelled = true;
    };
  }, [applyState]);

  // Effect B — feed vivo (SSE + stale), dependiente de `mode`. En 'live' abre el
  // stream y el wake re-arma el stale (vía applyState); en 'historic' NO abre
  // stream NI arma stale. El cleanup limpia el timer de stale (compartido) y
  // abortea el stream — el clearTimeout evita que el callback dispare tras montar.
  useEffect(() => {
    if (mode !== 'live') return;

    let cancelled = false;

    // Wake del feed: re-lee SOLO el estado (la geometría NO se re-pide) y recolorea.
    const onWake = async () => {
      try {
        const state = await congestionService.getState();
        if (cancelled) return;
        applyState(state);
      } catch (err) {
        // Falla de re-lectura: conservamos el último estado. El timer de stale ya
        // armado atenuará si no llegan más wakes.
        if (cancelled) return;
        console.error('Error re-leyendo estado de congestión:', err);
      }
    };

    // Al RE-entrar en 'live' desde histórico, re-leer el estado (puede haber
    // avanzado mientras el feed estaba silenciado). El mount inicial NO re-lee:
    // Effect A ya cargó el estado (preserva `getState` 1× al montar).
    if (firstLiveRef.current) {
      firstLiveRef.current = false;
    } else {
      void onWake();
    }

    const controller = openCongestionStream({ onWake: () => void onWake() });

    return () => {
      cancelled = true;
      if (staleTimerRef.current !== null) {
        window.clearTimeout(staleTimerRef.current);
        staleTimerRef.current = null;
      }
      controller.abort();
    };
  }, [mode, applyState]);

  // Effect C — carga de la serie del día (modo histórico). Al entrar en histórico
  // o cambiar de día, pide getSeries(day); maneja loading/error. `count === 0`
  // (día sin datos) NO es error: se refleja como serie vacía (CA-23.7).
  useEffect(() => {
    if (mode !== 'historic') return;

    let cancelled = false;
    setSeriesLoading(true);
    setSeriesError(null);

    congestionService
      .getSeries(day)
      .then((s) => {
        if (cancelled) return;
        setSeries(s);
        setSliderIndex(0);
      })
      .catch((err) => {
        if (cancelled) return;
        console.error('Error cargando la serie histórica de congestión:', err);
        setSeriesError('No se pudo cargar la serie histórica de este día.');
        setSeries(null);
      })
      .finally(() => {
        if (!cancelled) setSeriesLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [mode, day]);

  // Effect D — repintado histórico: cruza la geometría cacheada con el nivel del
  // índice del slider y recolorea (la `key` del <GeoJSON> lleva `sliderIndex`, así
  // que cada movimiento remonta la capa). Con serie vacía (count 0) el merge
  // devuelve todas las features con nivel null → mapa neutro, coherente con
  // "día sin datos" y sin contaminar las features del modo vivo.
  useEffect(() => {
    if (mode !== 'historic') return;
    const geometry = geometryRef.current;
    if (!geometry || !series) return;
    setFeatures(mergeCongestionAtIndex(geometry, series, sliderIndex));
  }, [mode, series, sliderIndex]);

  // Mientras stale: tick de 1 s para refrescar el contador "hace X" del banner.
  useEffect(() => {
    if (!stale) return;
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, [stale]);

  // <GeoJSON data> consume un GeoJsonObject: envolvemos las features cruzadas.
  const data = { type: 'FeatureCollection', features } as unknown as GeoJsonObject;

  // style callback: {color, weight} de congestionStyle (opciones de path válidas
  // de Leaflet). En stale atenúa la opacidad manteniendo color+grosor — SOLO en
  // modo 'live'; en histórico no hay feed que envejezca, así que no se atenúa.
  const styleFeature = (feature?: Feature): PathOptions => {
    const base = congestionStyle(
      (feature?.properties as { congestion_level?: number | null })?.congestion_level,
    );
    return mode === 'live' && stale ? { ...base, opacity: STALE_OPACITY } : base;
  };

  // Key del remonte: en histórico depende de `sliderIndex` (cada movimiento del
  // slider remonta la capa y recolorea); en vivo, de `renderSeq`/`stale` (Fase 3).
  const geoKey =
    mode === 'historic'
      ? `historic-${sliderIndex}`
      : `${renderSeq}-${stale ? 'stale' : 'live'}`;

  // Modo histórico: nº de muestras (longitud de la serie) y estado "día sin datos".
  const seriesLength = series?.edges[0]?.levels.length ?? 0;
  const noData = series !== null && (series.count === 0 || seriesLength === 0);

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
    <div className="relative w-full h-[calc(100vh-7rem)] min-h-[600px] flex flex-col">
      <div className="relative flex-1 rounded-xl overflow-hidden border border-slate-700 bg-slate-900">
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
            {/* key cambia por update (renderSeq/stale en vivo, sliderIndex en
                histórico) → remonta la capa y recolorea (react-leaflet 5 no
                refresca `data` in-place). */}
            <GeoJSON
              key={geoKey}
              data={data}
              style={styleFeature}
              onEachFeature={onEachFeature}
            />
            <CongestionLegend />
          </MapContainer>
        )}
        {/* Controles superiores: toggle vivo/histórico (CA-23.6) + rótulo y date
            picker del modo histórico (CA-23.4 / Paso 3). */}
        {!loading && !error && (
          <div className="absolute top-4 right-4 z-[500] flex flex-col items-end gap-2">
            <div className="flex rounded-full overflow-hidden border border-slate-700 bg-slate-900/90 backdrop-blur shadow-lg">
              <button
                type="button"
                onClick={() => setMode('live')}
                aria-pressed={mode === 'live'}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-sm font-semibold transition-colors ${
                  mode === 'live'
                    ? 'bg-indigo-600 text-white'
                    : 'text-slate-300 hover:bg-slate-800'
                }`}
              >
                <Radio size={14} /> Vivo
              </button>
              <button
                type="button"
                onClick={() => setMode('historic')}
                aria-pressed={mode === 'historic'}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-sm font-semibold transition-colors ${
                  mode === 'historic'
                    ? 'bg-indigo-600 text-white'
                    : 'text-slate-300 hover:bg-slate-800'
                }`}
              >
                <History size={14} /> Histórico
              </button>
            </div>
            {mode === 'historic' && (
              <div className="flex flex-col items-end gap-2 bg-slate-900/90 backdrop-blur border border-slate-700 rounded-xl px-4 py-3 shadow-lg">
                <div className="flex items-center gap-2">
                  <History size={16} className="text-indigo-400" />
                  <span
                    className="text-sm font-bold text-white"
                    title="(día de muestra, datos simulados)"
                  >
                    Visualización histórica
                  </span>
                </div>
                <label className="flex items-center gap-2 text-xs text-slate-300">
                  Día
                  <input
                    type="date"
                    value={day}
                    onChange={(e) => setDay(e.target.value)}
                    aria-label="Día a visualizar"
                    className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-white text-xs"
                  />
                </label>
                {seriesError && (
                  <span className="text-xs text-rose-400">{seriesError}</span>
                )}
              </div>
            )}
          </div>
        )}
        {/* Slider temporal + label de hora + marcas (Paso 4 / CA-23.1-3) y estado
            "día sin datos" (CA-23.7). Solo en histórico, sobre el mapa. */}
        {!loading && !error && mode === 'historic' && !seriesError && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-[500] w-[min(90%,640px)] bg-slate-900/90 backdrop-blur border border-slate-700 rounded-xl px-5 py-4 shadow-lg">
            {seriesLoading ? (
              <p className="text-sm text-slate-300 text-center">
                Cargando serie del día…
              </p>
            ) : series ? (
              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs uppercase tracking-wide text-slate-400">
                    Hora
                  </span>
                  <span className="text-lg font-bold text-white tabular-nums">
                    {noData
                      ? '--:--'
                      : formatHourLabel(series.t0, series.step_s ?? 60, sliderIndex)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={Math.max(0, seriesLength - 1)}
                  value={sliderIndex}
                  onChange={(e) => setSliderIndex(Number(e.target.value))}
                  disabled={noData}
                  aria-label="Hora del día"
                  className="w-full accent-indigo-500 disabled:opacity-40"
                />
                {/* Marcas orientativas cada 3 h (0..24). La marca 24 es el borde del
                    eje (23:59 es el último dato), no un instante con dato. */}
                <div className="flex justify-between text-[10px] text-slate-500 tabular-nums">
                  {HOUR_MARKS.map((h) => (
                    <span key={h}>{String(h).padStart(2, '0')}h</span>
                  ))}
                </div>
                {noData && (
                  <p className="text-sm text-slate-400 text-center pt-1">
                    No hay datos para este día
                  </p>
                )}
              </div>
            ) : null}
          </div>
        )}
        {/* Banner stale (CA-22.4): datos desactualizados → "datos de hace X".
            Solo aplica en modo 'live' (en histórico no hay feed que envejezca). */}
        {!loading && !error && mode === 'live' && stale && latestTs && (
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
