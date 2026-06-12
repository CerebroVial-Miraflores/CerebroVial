// FASE 3 rediseño UI — tarjeta del mapa operativo (spec: .map-card del
// prototipo): head con capas/modos/controles temporales + MapCanvas con la
// capa GeoJSON remontada por key (caveat react-leaflet en edgeStyle.ts) y los
// NodeMarkers reales.
//
// Presentacional: todo el estado llega por props (la URL es la fuente de
// verdad en CommandView). El slider de timestep es un <input type="range">
// propio con tokens — NO se reusa control/Slider (v1 slate, zona "reusar sin
// reescribir" del playground) — decisión D10.
import { useEffect, useState } from 'react';
import { GeoJSON } from 'react-leaflet';
import type { Layer } from 'leaflet';
import type { Feature, FeatureCollection } from 'geojson';

import { MapCanvas } from '../../map/MapCanvas';
import { MapLegend } from '../../map/MapLegend';
import { MapModeBadge } from '../../map/MapModeBadge';
import { NodeMarker } from '../../map/NodeMarker';
import { LayerChips } from '../../map/LayerChips';
import { jamLevelPathOptions } from '../../map/edgeStyle';
import { SegmentedControl } from '../../ui/SegmentedControl';
import { StatusChip } from '../../ui/StatusChip';
import { HChip } from '../../ui/Chip';
import { Button } from '../../ui/Button';
import type { MergedCongestionFeature } from '../../../types/congestion';
import {
  formatHourLabel,
  legendForMode,
  type CommandMarker,
  type CommandMode,
  type PredictionHorizon,
} from './derive';

const MAP_CENTER: [number, number] = [-12.122, -77.028];
const MAP_ZOOM = 14;

const MODE_OPTIONS = [
  { value: 'historico', label: 'Histórico' },
  { value: 'ahora', label: '● Ahora' },
  { value: 'prediccion', label: 'Predicción' },
] as const;

export interface CommandMapProps {
  mode: CommandMode;
  onChangeMode: (m: CommandMode) => void;
  layers: { traffic: boolean; nodes: boolean };
  onToggleLayer: (id: 'traffic' | 'nodes') => void;
  horizon: PredictionHorizon;
  onChangeHorizon: (h: PredictionHorizon) => void;
  dia: string;
  onChangeDia: (d: string) => void;
  /** Índice de timestep ya clampeado + largo de la serie del día. */
  t: number;
  tMax: number;
  serieT0: string | null;
  serieStepS: number;
  onChangeT: (i: number) => void;
  features: MergedCongestionFeature[] | null;
  geoKey: string;
  markers: CommandMarker[];
  onNodeClick: (cameraId: string) => void;
  live: { lastUpdated: number | null; isStale: boolean };
  /** Estado del recurso del modo activo (loading/error/empty honestos). */
  modePanel: { loading: boolean; error: string | null; onRetry: () => void; empty: string | null };
  /** Ring warn ~900 ms (acción "Ver predicción" del panel de alertas). */
  flash: boolean;
  coachTip?: string | null;
}

/** "hace Ns" del head (tick local de 5 s, como el `upd` del prototipo). */
function useElapsedLabel(lastUpdated: number | null): string | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 5000);
    return () => window.clearInterval(id);
  }, []);
  if (lastUpdated === null) return null;
  return `· hace ${Math.max(0, Math.round((now - lastUpdated) / 1000))} s`;
}

function onEachFeature(feature: Feature, layer: Layer) {
  const props = (feature as MergedCongestionFeature).properties;
  const nivel = props.congestion_level === null ? 'sin dato' : `nivel ${props.congestion_level}`;
  layer.bindTooltip(`${props.edge_id} · ${nivel}`, { sticky: true });
}

export function CommandMap({
  mode,
  onChangeMode,
  layers,
  onToggleLayer,
  horizon,
  onChangeHorizon,
  dia,
  onChangeDia,
  t,
  tMax,
  serieT0,
  serieStepS,
  onChangeT,
  features,
  geoKey,
  markers,
  onNodeClick,
  live,
  modePanel,
  flash,
  coachTip,
}: CommandMapProps) {
  const elapsed = useElapsedLabel(live.lastUpdated);
  const tLabel = formatHourLabel(serieT0, serieStepS, t);
  const legend = legendForMode(mode, tLabel);

  return (
    <section
      className={`flex min-w-0 flex-1 flex-col overflow-hidden rounded-panel border bg-linear-to-b from-white/4 to-white/1 transition-shadow duration-400 ${
        flash ? 'border-warn/55 shadow-[0_0_0_2px] shadow-warn/40' : 'border-line'
      }`}
    >
      {/* map-head */}
      <div className="flex flex-wrap items-center gap-2.5 border-b border-line px-3.5 py-[11px]">
        <span className="flex items-center gap-[9px] text-[13.5px] font-bold">
          {mode === 'ahora' && (
            <span aria-hidden="true" className="h-[7px] w-[7px] animate-pulse rounded-full bg-ok" />
          )}
          Mapa operativo · Miraflores
          {mode === 'ahora' && elapsed !== null && (
            <span className="num text-[10.5px] font-normal text-ink-2">{elapsed}</span>
          )}
        </span>
        {mode === 'ahora' && live.isStale && (
          <StatusChip status="warn">DATOS VIEJOS</StatusChip>
        )}

        <LayerChips
          layers={[
            { id: 'traffic', label: 'Tráfico', on: layers.traffic },
            { id: 'nodes', label: 'Semáforos', on: layers.nodes },
          ]}
          onToggle={(id) => onToggleLayer(id as 'traffic' | 'nodes')}
        />

        <div className="flex-1" />

        {mode === 'historico' && (
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="date"
              aria-label="Día del histórico"
              value={dia}
              onChange={(event) => onChangeDia(event.target.value)}
              className="num rounded-ctl border border-line bg-panel px-2.5 py-[5px] text-[11.5px] text-ink [color-scheme:dark]"
            />
            <input
              type="range"
              aria-label="Paso temporal"
              min={0}
              max={tMax}
              value={t}
              disabled={tMax <= 0}
              onChange={(event) => onChangeT(Number(event.target.value))}
              className="w-36 accent-(--color-info)"
            />
            <span className="num min-w-[44px] text-[11.5px] font-bold text-info">{tLabel}</span>
          </div>
        )}

        {mode === 'prediccion' && (
          <div className="flex items-center gap-[5px]">
            <HChip on={horizon === 15} onToggle={() => onChangeHorizon(15)}>
              +15 min
            </HChip>
            <HChip on={horizon === 30} onToggle={() => onChangeHorizon(30)}>
              +30
            </HChip>
            {/* +60 del prototipo NO existe: el modelo predice horizon=30 pasos (D4). */}
          </div>
        )}

        <SegmentedControl
          options={MODE_OPTIONS}
          value={mode}
          onChange={onChangeMode}
          ariaLabel="Modo temporal del mapa"
        />
      </div>

      {/* Estado del recurso del modo (paridad: empty/error honestos, sin mock) */}
      {(modePanel.error !== null || modePanel.empty !== null || modePanel.loading) && (
        <div className="flex flex-wrap items-center gap-3 border-b border-line px-3.5 py-2 text-[11.5px] text-ink-2">
          {modePanel.loading && <span>Cargando datos del modo…</span>}
          {!modePanel.loading && modePanel.error !== null && (
            <>
              <span className="text-bad">{modePanel.error}</span>
              <Button onClick={modePanel.onRetry}>Reintentar</Button>
            </>
          )}
          {!modePanel.loading && modePanel.error === null && modePanel.empty !== null && (
            <span>{modePanel.empty}</span>
          )}
        </div>
      )}

      <MapCanvas
        center={MAP_CENTER}
        zoom={MAP_ZOOM}
        className="min-h-[420px] md:min-h-[520px]"
        legend={<MapLegend title={legend.title} items={legend.items} />}
        modeBadge={
          <MapModeBadge
            mode={mode === 'ahora' ? 'live' : mode === 'prediccion' ? 'prediction' : 'historic'}
            offsetMin={horizon}
            timestampLabel={`${dia} ${tLabel}`}
          />
        }
        coachTip={
          coachTip ? (
            <span className="whitespace-nowrap rounded-full border border-brand/50 bg-brand/16 px-4 py-2 text-[11.5px] text-brand-2 backdrop-blur-md">
              {coachTip}
            </span>
          ) : undefined
        }
      >
        {layers.traffic && features !== null && (
          <GeoJSON
            key={geoKey}
            data={{ type: 'FeatureCollection', features } as unknown as FeatureCollection}
            style={(feature) =>
              jamLevelPathOptions(
                (feature as MergedCongestionFeature | undefined)?.properties.congestion_level,
              )
            }
            onEachFeature={onEachFeature}
          />
        )}
        {layers.nodes &&
          markers.map((marker) => (
            <NodeMarker
              key={marker.id}
              position={marker.position}
              status={marker.status}
              critical={marker.critical}
              onClick={() => onNodeClick(marker.id)}
            />
          ))}
      </MapCanvas>
    </section>
  );
}
