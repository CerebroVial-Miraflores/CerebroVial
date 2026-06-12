// FASE 3 rediseño UI — Centro de Comando ("/"), la vista que fusiona el
// dashboard, el mapa de congestión y las alertas de v1 (spec: vista Operación
// del prototipo).
//
// Responsabilidades: estado URL (useCommandUrlState — la URL es la fuente de
// verdad de modo/dia/t/nodo/panel), orquestación de hooks de datos, buffers de
// sparkline y composición KpiStrip + CommandMap + AlertsPanel.
//
// B3: click en nodo y "Ver intersección" abren el drawer de intersección por
// ?nodo= (la costura 3A vía /camara/:id quedó reemplazada; /camara/:id queda
// como destino de "Detalle de cámara" del drawer → CameraDetailV2 en F4). Click
// en KPI → modal.
import { useEffect, useMemo, useRef, useState } from 'react';

import { useCongestionGeometry } from '../../../hooks/useCongestionGeometry';
import { useCongestionPrediction } from '../../../hooks/useCongestionPrediction';
import { useCongestionSeries } from '../../../hooks/useCongestionSeries';
import { useCongestionState } from '../../../hooks/useCongestionState';
import { useIntersections } from '../../../hooks/useIntersections';
import { useVisionAggregates } from '../../../hooks/useVisionAggregates';
import { AlertsPanel } from './AlertsPanel';
import { CommandMap } from './CommandMap';
import { IntersectionDrawer } from './IntersectionDrawer';
import { KpiModal } from './KpiModal';
import { KpiStrip, type KpiKind } from './KpiStrip';
import { useAdaptiveNodes } from './useAdaptiveNodes';
import { useCommandUrlState } from './useCommandUrlState';
import {
  clampIndex,
  edgeLevelForNode,
  featuresForMode,
  geoJsonKeyFor,
  markersFrom,
  networkCongestionIndex,
  predictionLabelAt,
  pushSample,
  seriesToNetworkIndex,
  todayIsoLocal,
  type PredictionHorizon,
} from './derive';

/**
 * Buffer del sparkline acumulado en render (render-guard, patrón
 * useVisionStream): appendea solo cuando el valor cambia, sin effects.
 *
 * `seed` (B0): historia inicial opcional — el índice de red se siembra desde
 * la serie del día actual (useCongestionSeries) cuando existe; se aplica solo
 * si aporta más puntos que lo acumulado en vivo. Las sparks de visión (vel/
 * flu) NO tienen fuente histórica directa: se llenan en sesión, muestra a
 * muestra — comportamiento esperado, el placeholder del strip cubre el inicio.
 */
function useSparkBuffer(
  value: number | null,
  seed: readonly number[] | null = null,
): readonly number[] {
  const [hist, setHist] = useState<readonly number[]>([]);
  const [prevSeed, setPrevSeed] = useState<readonly number[] | null>(null);
  if (seed !== prevSeed) {
    setPrevSeed(seed);
    if (seed !== null && seed.length > hist.length) {
      setHist(seed);
    }
  }
  const [prev, setPrev] = useState<number | null>(null);
  if (value !== prev) {
    setPrev(value);
    setHist((p) => pushSample(p, value));
  }
  return hist;
}

const COACH_TIP_SHOW_MS = 1600;
const COACH_TIP_HIDE_MS = 9500;
const FLASH_MS = 900;

export function CommandView() {
  const url = useCommandUrlState();

  // ── Datos ────────────────────────────────────────────────────────────────
  // Markers: el `status` real (agregado Waze) se refresca por polling de 60 s
  // (v1 re-leía en cada wake del SSE de congestión; el polling es equivalente
  // y no acopla este hook al ciclo de vida del stream).
  const intersections = useIntersections({ intervalMs: 60_000 });
  const congestionState = useCongestionState();
  const geometry = useCongestionGeometry();
  const series = useCongestionSeries(url.mode === 'historico' ? url.dia : '');
  const prediction = useCongestionPrediction();
  const adaptive = useAdaptiveNodes();
  // B0: serie del día ACTUAL (one-shot) para sembrar la spark del índice con
  // historia real cuando la ventana del día ya tiene datos.
  const [today] = useState(() => todayIsoLocal());
  const todaySeries = useCongestionSeries(today);

  const cameraIds = useMemo(
    () => (intersections.data ?? []).map((i) => i.id),
    [intersections.data],
  );
  const vision = useVisionAggregates(cameraIds);

  // ── Estado local (no-URL) ────────────────────────────────────────────────
  const [horizon, setHorizon] = useState<PredictionHorizon>(15);
  const [layers, setLayers] = useState({ traffic: true, nodes: true });
  const [flash, setFlash] = useState(false);
  const [coachVisible, setCoachVisible] = useState(false);
  const [kpiOpen, setKpiOpen] = useState<KpiKind | null>(null);
  const flashTimerRef = useRef<number | null>(null);

  useEffect(() => {
    const show = window.setTimeout(() => setCoachVisible(true), COACH_TIP_SHOW_MS);
    const hide = window.setTimeout(() => setCoachVisible(false), COACH_TIP_HIDE_MS);
    return () => {
      window.clearTimeout(show);
      window.clearTimeout(hide);
      if (flashTimerRef.current !== null) window.clearTimeout(flashTimerRef.current);
    };
  }, []);

  // ── Derivaciones ─────────────────────────────────────────────────────────
  const idxValue = networkCongestionIndex(congestionState.data);
  const velValue = vision.aggregate.meanSpeedKmh;
  const fluValue = vision.aggregate.totalFlowVph;
  const idxSeed = useMemo(() => {
    const networkIndex = seriesToNetworkIndex(todaySeries.data);
    return networkIndex ? networkIndex.points.slice(-24) : null;
  }, [todaySeries.data]);
  const idxSpark = useSparkBuffer(idxValue, idxSeed);
  const velSpark = useSparkBuffer(velValue);
  const fluSpark = useSparkBuffer(fluValue);

  const serieLen = series.data?.edges[0]?.levels.length ?? 0;
  const tClamped = clampIndex(url.tRaw, serieLen);

  const markers = useMemo(() => markersFrom(intersections.data), [intersections.data]);

  const features = useMemo(
    () =>
      featuresForMode({
        mode: url.mode,
        geometry: geometry.data,
        state: congestionState.data,
        series: series.data,
        tClamped,
        prediction: prediction.data,
        horizon,
      }),
    [url.mode, geometry.data, congestionState.data, series.data, tClamped, prediction.data, horizon],
  );

  // Drawer (B3): nivel real del tramo del nodo abierto (convención cam_<node_id>).
  const drawerNodeId = url.nodo !== null ? url.nodo.replace(/^cam_/, '') : '';
  const drawerEdgeLevel = useMemo(
    () => edgeLevelForNode(geometry.data, congestionState.data, drawerNodeId),
    [geometry.data, congestionState.data, drawerNodeId],
  );

  // KPI modal idx (B3): serie real del día en contexto (?dia en histórico,
  // hoy en los demás modos — reusa el fetch del seed de la spark).
  const idxModalSource = url.mode === 'historico' ? series : todaySeries;
  const idxModalDay = url.mode === 'historico' ? url.dia : today;
  const idxModalSeries = useMemo(
    () => seriesToNetworkIndex(idxModalSource.data),
    [idxModalSource.data],
  );

  const geoKey = geoJsonKeyFor(url.mode, {
    lastUpdated: congestionState.lastUpdated,
    dia: url.dia,
    tClamped,
    horizon,
    baseTimestep: prediction.data?.base_timestep ?? null,
  });

  // Estado honesto del recurso del modo activo (política de paridad).
  const modePanel = useMemo(() => {
    if (url.mode === 'historico') {
      return {
        loading: series.loading || geometry.loading,
        error: series.error ?? geometry.error,
        onRetry: () => {
          void series.refetch();
          void geometry.refetch();
        },
        empty:
          series.data !== null && series.data.t0 === null
            ? `El día ${url.dia} no tiene datos de congestión.`
            : null,
      };
    }
    if (url.mode === 'prediccion') {
      return {
        loading: prediction.loading || geometry.loading,
        error:
          prediction.error !== null
            ? 'Servicio de predicción no disponible.'
            : geometry.error,
        onRetry: () => {
          void prediction.refetch();
          void geometry.refetch();
        },
        empty:
          prediction.data !== null && prediction.data.edges.length === 0
            ? 'La predicción no trae aristas para la ventana actual.'
            : null,
      };
    }
    return {
      loading: congestionState.loading || geometry.loading,
      error: congestionState.error ?? geometry.error,
      onRetry: () => {
        void congestionState.refetch();
        void geometry.refetch();
      },
      empty:
        congestionState.data !== null && congestionState.data.edges.length === 0
          ? 'Sin datos de congestión en la ventana actual — la red pinta neutra.'
          : null,
    };
  }, [url.mode, url.dia, series, prediction, congestionState, geometry]);

  // ── Interacciones ────────────────────────────────────────────────────────
  const showPrediction = () => {
    url.setMode('prediccion');
    setFlash(true);
    if (flashTimerRef.current !== null) window.clearTimeout(flashTimerRef.current);
    flashTimerRef.current = window.setTimeout(() => setFlash(false), FLASH_MS);
  };

  // B3: click en nodo / "Ver intersección" → drawer por ?nodo= (URL, con back).
  const openNode = (cameraId: string) => {
    url.openNode(cameraId);
  };

  // ?panel=alertas (D3.1a, /alertas → /?panel=alertas): one-shot scroll+focus
  // al panel y limpieza del param (replace) — el effect no re-corre.
  const panelRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (url.panel !== 'alertas') return;
    panelRef.current?.scrollIntoView?.({ block: 'start' });
    panelRef.current?.focus?.();
    url.clearPanel();
  }, [url, url.panel]);

  return (
    <div className="flex min-h-full flex-col">
      <KpiStrip
        idx={{ value: idxValue, spark: idxSpark }}
        vel={{ value: velValue, spark: velSpark }}
        flu={{ value: fluValue, spark: fluSpark }}
        sem={{ active: adaptive.activeCount }}
        pred={{
          label: predictionLabelAt(prediction.data, 15),
          unavailable: prediction.error !== null,
          onRetry: () => void prediction.refetch(),
        }}
        onShowPrediction={showPrediction}
        onOpenKpi={setKpiOpen}
      />

      <div className="flex flex-1 flex-col gap-[13px] min-[1240px]:flex-row min-[1240px]:items-stretch">
        <CommandMap
          mode={url.mode}
          onChangeMode={url.setMode}
          layers={layers}
          onToggleLayer={(id) => setLayers((prev) => ({ ...prev, [id]: !prev[id] }))}
          horizon={horizon}
          onChangeHorizon={setHorizon}
          dia={url.dia}
          onChangeDia={url.setDia}
          t={tClamped}
          tMax={Math.max(0, serieLen - 1)}
          serieT0={series.data?.t0 ?? null}
          serieStepS={series.data?.step_s ?? 60}
          onChangeT={url.setT}
          features={features}
          geoKey={geoKey}
          markers={markers}
          onNodeClick={openNode}
          live={{ lastUpdated: congestionState.lastUpdated, isStale: congestionState.isStale }}
          modePanel={modePanel}
          flash={flash}
          coachTip={coachVisible ? '✨ Probá: clic en un nodo del mapa' : null}
        />

        <div
          ref={panelRef}
          tabIndex={-1}
          className="w-full shrink-0 outline-none min-[1240px]:w-[356px]"
        >
          <AlertsPanel onOpenNode={openNode} onShowPrediction={showPrediction} />
        </div>
      </div>

      {/* Overlays (B3): drawer por ?nodo= y modal por estado local. Si
          conviven, Esc cierra el de arriba (overlay-stack de B1). */}
      <IntersectionDrawer
        cameraId={url.nodo}
        intersections={intersections.data}
        intersectionsLoading={intersections.loading}
        vision={url.nodo !== null ? vision.byCamera[url.nodo] : undefined}
        edgeLevel={drawerEdgeLevel}
        onClose={url.closeNode}
      />
      <KpiModal
        kind={kpiOpen}
        onClose={() => setKpiOpen(null)}
        idx={{
          series: idxModalSeries,
          loading: idxModalSource.loading,
          error: idxModalSource.error,
          onRetry: () => void idxModalSource.refetch(),
          day: idxModalDay,
          predictionUnavailable: prediction.error !== null,
        }}
        adaptive={adaptive}
      />
    </div>
  );
}
