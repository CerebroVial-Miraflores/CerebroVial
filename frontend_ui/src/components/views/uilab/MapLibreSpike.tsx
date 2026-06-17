import { useMemo, useRef } from 'react';
import { Map as MapGL, Source, Layer, type MapRef } from 'react-map-gl/maplibre';
import type { ExpressionSpecification } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

import { Button } from '../../ui/Button';
import { StatusChip, type Status } from '../../ui/StatusChip';
import { useCongestionGeometry } from '../../../hooks/useCongestionGeometry';
import { useCongestionState } from '../../../hooks/useCongestionState';
import { jamLevelStyle, JAM_LEVEL_LEGEND, type JamLevel } from '../../map/edgeStyle';
import { mergeCongestion } from '../../../utils/congestion';
import type { StreamConnectionState } from '../../../hooks/types';

// FASE 1 migración MapLibre — color data-driven + estado vivo, SOLO DEV (chunk
// lazy de UiLabView). NO toca CommandView/CommandMap.
//
// Sobre el plumbing de Fase 0: las 1660 aristas ahora pintan por su nivel de
// congestión 0-5 (escala de edgeStyle.ts) y RECOLOREAN en vivo. Reusa:
//   - useCongestionState: GET /congestion/state + SSE /state/stream (wake → refetch).
//   - mergeCongestion: join geometry × state por edge_id (lógica pura existente).
//   - jamLevelStyle / JAM_LEVEL_LEGEND: grosor/opacidad y leyenda de la escala real.
// La mejora vs Leaflet: cambiar el `data` del <Source> dispara source.setData()
// internamente en react-map-gl → recolor IN-PLACE, sin remontar la capa (Leaflet
// requería remount-by-key porque setStyle no actualiza la className del path).

const LIBERTY_STYLE = 'https://tiles.openfreemap.org/styles/liberty';

const MIRAFLORES_OVERVIEW = {
  longitude: -77.035,
  latitude: -12.115,
  zoom: 13.3,
  pitch: 0,
  bearing: 0,
} as const;

// Intersección crítica Av. Larco × Av. Benavides (nodo `larco_benavides`,
// scripts/seed.py:59), en orden GeoJSON [lon, lat].
const LARCO_BENAVIDES: [number, number] = [-77.0301, -12.1227];

// Color por nivel: MapLibre no resuelve var() en un paint, así que estos
// literales SON el puente necesario hacia los tokens. Byte-alineados con
// tokens.css y el mapeo de map.css (.edge-jam-N): 0-1 ok-road / 2-3 warn /
// 4 bad / 5 sev / sin-dato ink-3. Si cambia el token, cambia acá (no es una
// escala nueva: es la misma de edgeStyle.ts portada a literal).
const COLOR_OK_ROAD = '#0fae79'; // --color-ok-road
const COLOR_WARN = '#f59e0b'; // --color-warn
const COLOR_BAD = '#ef4444'; // --color-bad
const COLOR_SEV = '#a855f7'; // --color-sev
const COLOR_NEUTRAL = '#5b6275'; // --color-ink-3 (arista sin estado / huérfana)

const LEVEL_COLOR: Record<JamLevel, string> = {
  0: COLOR_OK_ROAD,
  1: COLOR_OK_ROAD,
  2: COLOR_WARN,
  3: COLOR_WARN,
  4: COLOR_BAD,
  5: COLOR_SEV,
};

const LEVELS: JamLevel[] = [0, 1, 2, 3, 4, 5];

// Expresiones data-driven construidas DESDE edgeStyle.ts (grosor/opacidad) y
// LEVEL_COLOR (color): un `match` sobre congestion_level. Entrada null (arista
// sin estado o pre-merge) → cae al fallback neutro. jamLevelStyle(null) da el
// NEUTRAL_STYLE (weight 3, opacity 0.35) sin duplicar la tabla.
const LINE_COLOR: ExpressionSpecification = [
  'match',
  ['get', 'congestion_level'],
  ...LEVELS.flatMap((l) => [l, LEVEL_COLOR[l]]),
  COLOR_NEUTRAL,
] as ExpressionSpecification;

// Grosor uniforme fino (2px), como el placeholder azul de Fase 0. Decisión de
// producto: se prioriza el grosor parejo sobre la redundancia no-cromática por
// nivel (weight 3→9 de edgeStyle.ts / CA-22.3) — el nivel se lee solo por color.
const LINE_WIDTH = 2;

const LINE_OPACITY: ExpressionSpecification = [
  'match',
  ['get', 'congestion_level'],
  ...LEVELS.flatMap((l) => [l, jamLevelStyle(l).opacity]),
  jamLevelStyle(null).opacity,
] as ExpressionSpecification;

const CONNECTION_STATUS: Record<StreamConnectionState, Status> = {
  connecting: 'warn',
  open: 'ok',
  retrying: 'warn',
  closed: 'bad',
};

function timeLabel(epochMs: number | null): string {
  return epochMs === null ? '—' : new Date(epochMs).toLocaleTimeString();
}

export function MapLibreSpike() {
  const mapRef = useRef<MapRef | null>(null);
  const geometry = useCongestionGeometry();
  const state = useCongestionState({ staleAfterMs: 90_000 });

  // GeoJSON ya coloreable: geometry × state por edge_id. Antes de que llegue el
  // primer /state, las features van sin congestion_level → el match las pinta
  // neutras. Cada wake SSE avanza state.data → nueva referencia acá → react-map-gl
  // hace setData (recolor in-place, sin remontar el <Layer>).
  const featureCollection = useMemo(() => {
    if (!geometry.data) return null;
    const features = state.data
      ? mergeCongestion(geometry.data, state.data)
      : geometry.data.features;
    return { type: 'FeatureCollection' as const, features };
  }, [geometry.data, state.data]);

  const featureCount = geometry.data?.features.length ?? null;
  const statedEdges = state.data?.edges.length ?? null;

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
        Fase 1: las 1660 aristas pintan por nivel de congestión 0-5 (misma escala que
        edgeStyle.ts) y recolorean en vivo al llegar un wake SSE — sin remontar la capa. Consume el
        backend REAL (core 8001, JWT, rol operator/admin). Para forzar wakes observables:{' '}
        <span className="num">
          .venv/bin/python scripts/replay_congestion.py --mode vivo --day seed051 --speedup 60
        </span>
        .
      </p>

      <div className="flex flex-wrap items-center gap-3">
        <Button variant="pri" onClick={flyToCritical}>
          ✈ Volar a Larco × Benavides
        </Button>
        {geometry.loading && <span className="text-[12px] text-ink-2">Cargando geometría…</span>}
        {featureCount !== null && (
          <StatusChip status={featureCount === 1660 ? 'ok' : 'warn'}>
            {featureCount} tramos
          </StatusChip>
        )}
        <StatusChip status={CONNECTION_STATUS[state.connection]}>
          SSE {state.connection.toUpperCase()}
        </StatusChip>
        {statedEdges !== null && (
          <StatusChip status={state.isStale ? 'warn' : 'ok'}>
            {statedEdges} con estado · {state.isStale ? 'STALE' : 'FRESCO'}
          </StatusChip>
        )}
        <span className="text-[11px] text-ink-3">
          últ. estado: <span className="num">{timeLabel(state.lastUpdated)}</span>
        </span>
      </div>

      {/* Leyenda de la escala real (JAM_LEVEL_LEGEND de edgeStyle.ts). */}
      <div className="flex flex-wrap items-center gap-4">
        {JAM_LEVEL_LEGEND.map((item) => (
          <span key={item.label} className="flex items-center gap-1.5 text-[11px] text-ink-2">
            <span className={`h-2.5 w-4 rounded-full ${item.swatchClass}`} />
            {item.label}
          </span>
        ))}
        <span className="flex items-center gap-1.5 text-[11px] text-ink-3">
          <span className="h-2.5 w-4 rounded-full bg-ink-3/60" />
          Sin dato
        </span>
      </div>

      {(geometry.error || state.error) && (
        <div className="space-y-2 rounded-ctl border border-bad/40 bg-bad/10 p-3">
          {geometry.error && (
            <p className="text-[12px] leading-relaxed text-bad">geometry: {geometry.error}</p>
          )}
          {state.error && (
            <p className="text-[12px] leading-relaxed text-bad">state: {state.error}</p>
          )}
          <div className="flex gap-2">
            {geometry.error && (
              <Button onClick={() => void geometry.refetch()}>Reintentar geometría</Button>
            )}
            {state.error && <Button onClick={() => void state.refetch()}>Reintentar estado</Button>}
          </div>
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
                layout={{ 'line-cap': 'round', 'line-join': 'round' }}
                paint={{
                  'line-color': LINE_COLOR,
                  'line-width': LINE_WIDTH,
                  'line-opacity': LINE_OPACITY,
                }}
              />
            </Source>
          )}
        </MapGL>
      </div>

      <p className="text-[11px] text-ink-3">
        Criterio de cierre: las aristas pintan por nivel con la escala verde→ámbar→rojo→violeta
        (no el placeholder azul de Fase 0), y recolorean solas al llegar un wake del replay, sin
        parpadeo de remonte.
      </p>
    </div>
  );
}
