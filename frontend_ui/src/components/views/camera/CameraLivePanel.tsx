import { useCallback, useEffect, useState } from 'react';
import { Activity, AlertTriangle, Car, Gauge, Sparkles, Waves } from 'lucide-react';

import { useVisionStream } from '../../../hooks/useVisionStream';
import { predictionService, type PredictionResult } from '../../../services/predictionService';
import {
  edgeInferenceService,
  EdgeCapacityError,
} from '../../../services/edgeInferenceService';
import { congestionLabel, densityPercent } from '../../../utils/trafficLabels';
import { HlsPlayer } from '../../HlsPlayer';
import { SegmentedControl } from '../../ui/SegmentedControl';
import type { Status } from '../../ui/StatusChip';
import { StatusChip } from '../../ui/StatusChip';
import { AnnotatedCameraStream, type StreamStatus } from './AnnotatedCameraStream';

// FASE 4 Mitad A — panel EN VIVO del detalle de cámara. Ciclo de vida DELIBERADO
// de la inferencia (toggle Directo/Detección), sobre el design system:
//  - Al montar SOLO se consulta el estado: GET /cameras/inference-status define
//    si esta cámara ya viene en ALTA (toggle reflejado, idempotente, StrictMode-safe).
//  - El alta/baja del YOLO la decide el usuario con el toggle, NUNCA un effect:
//    check (ALTA) → POST /cameras/{id}; uncheck (BAJA) → DELETE /cameras/{id}.
//    INVARIANTE: el DELETE va SOLO en el handler de uncheck, JAMÁS en cleanup/
//    unmount (un DELETE en cleanup bajo <StrictMode> aterriza tras el POST del
//    remonte y baja la cámara recién dada de alta — lección de v1).
//  - BAJA → HlsPlayer (HLS directo de Claro, sin cajas, baja latencia), panel
//    atenuado. ALTA → AnnotatedCameraStream: <canvas> con el MJPEG processed del
//    edge (GET /video/{id}?type=processed) vía fetch + ReadableStream, cajas ya
//    dibujadas server-side (cero alineación cliente), reconexión propia.
//  - SSE de métricas vía useVisionStream (gate isActive → se apaga solo en BAJA).
//  - Insights de IA: DESHABILITADO hasta HU-03 (ver PREDICTION_ENABLED abajo).
// Política de datos: vehículos = REAL (conteo YOLO); flujo/ocupación =
// REAL-CON-CAVEAT (presencia extrapolada); velocidad = REAL-CON-CAVEAT (sin
// calibrar, DEUDA-SPEED-CALIB). Peatones/incidentes de v1 NO migran (eran 0
// hardcodeado, dato inventado). Controles de dispositivo de v1 NO migran (eran
// botones sin handler ni backend).

// Predicción deshabilitada hasta HU-03: el cliente todavía manda el shape RF viejo
// y el backend GRU (TTH-09) exige otro contrato → 422 consistente (ruptura conocida
// Delta-01). No se dispara el POST condenado hasta migrar el cliente al shape GRU.
const PREDICTION_ENABLED = false;

const TOGGLE_OPTIONS = [
  { value: 'baja' as const, label: 'Directo' },
  { value: 'alta' as const, label: 'Detección' },
];

interface CameraLivePanelProps {
  cameraId: string;
  streamUrl: string | null;
}

function statusForLabel(label: string): Status {
  if (label === 'Alto') return 'bad';
  if (label === 'Moderado') return 'warn';
  return 'ok';
}

export function CameraLivePanel({ cameraId, streamUrl }: CameraLivePanelProps) {
  // Toggle ALTA/BAJA de inferencia. Init: BAJA; el estado real lo resuelve el GET
  // de abajo. `statusResolving` evita el parpadeo del toggle antes de esa respuesta.
  const [isActive, setIsActive] = useState(false);
  const [statusResolving, setStatusResolving] = useState(() => !!streamUrl);
  // `isLoading` ("Activando detección…") cubre la espera del primer frame anotado:
  // tras dar el alta (toggle ALTA) o si el GET inicial ya encontró la cámara en ALTA.
  const [isLoading, setIsLoading] = useState(false);
  // `notice` blando: aviso transitorio (capacidad del edge / alta fallida) que NO
  // tapa el video — en BAJA el HlsPlayer directo sigue visible (no depende del edge).
  const [notice, setNotice] = useState<string | null>(null);
  // `error` duro reemplaza el video; única causa: sin stream configurado (derivado).
  const error = streamUrl ? null : 'Esta cámara no tiene un stream configurado.';

  // Al montar: SOLO consultar estado (GET idempotente, StrictMode-safe). NO postea,
  // NO borra. Refleja en el toggle si la cámara ya viene infiriendo en el edge.
  useEffect(() => {
    if (!streamUrl) return;
    let cancelled = false;

    edgeInferenceService
      .getInferenceStatus()
      .then((status) => {
        if (cancelled) return;
        const active = status.inferring.includes(cameraId);
        setIsActive(active);
        setIsLoading(active); // si ya infiere, esperamos el primer frame anotado
        setStatusResolving(false);
      })
      .catch((err) => {
        console.error('Consulta de inference-status falló', err);
        if (cancelled) return;
        // No bloquea el panel: se asume BAJA y el toggle queda operable.
        setStatusResolving(false);
      });

    return () => {
      cancelled = true;
    };
  }, [cameraId, streamUrl]);

  // Handler del toggle — ÚNICO lugar donde se hace POST/DELETE. Nunca en effect/cleanup.
  const handleToggle = useCallback(
    (next: 'baja' | 'alta') => {
      if (!streamUrl || next === (isActive ? 'alta' : 'baja')) return;

      if (next === 'alta') {
        setNotice(null);
        setIsLoading(true);
        edgeInferenceService
          .startInference(cameraId, { source: streamUrl, source_type: 'hls', zones: {} })
          .then(() => setIsActive(true))
          .catch((err) => {
            setIsActive(false);
            setIsLoading(false);
            if (err instanceof EdgeCapacityError) {
              setNotice('El edge está a capacidad: no se pueden activar más cámaras a la vez.');
            } else {
              console.error('Alta de detección en el edge falló', err);
              setNotice('No se pudo iniciar la detección. Reintentá en unos segundos.');
            }
          });
      } else {
        // BAJA → DELETE. Si falla, igual quedamos en BAJA (el watchdog del edge cierra).
        setIsActive(false);
        setIsLoading(false);
        setNotice(null);
        edgeInferenceService.stopInference(cameraId).catch((err) => {
          console.error('Baja de detección en el edge falló', err);
        });
      }
    },
    [cameraId, streamUrl, isActive],
  );

  // SSE de métricas: solo tras el alta OK (isActive). El hook auto-resetea sus
  // métricas al cambiar de cámara/enabled.
  const vision = useVisionStream(cameraId, { enabled: isActive });
  const metrics = vision.data?.metrics ?? null;

  // isLoading se limpia con el primer frame anotado del server ('streaming'). El
  // componente maneja su propia reconexión y muestra el badge "Reconectando…" en los
  // cortes transitorios — no se setea un error duro desde el stream (reconecta solo).
  const handleStatus = useCallback((status: StreamStatus) => {
    if (status === 'streaming') setIsLoading(false);
  }, []);

  const vehicles = metrics ? Math.round(metrics.unique_vehicles) : 0;
  const flowPerHour = metrics ? Math.round(metrics.flow_vehicles_per_hour) : 0;
  const speed = metrics?.mean_speed_kmh ?? null;
  const occupancy = metrics?.mean_occupancy ?? 0;
  const congestion = congestionLabel(occupancy);
  const density = densityPercent(occupancy);

  // Insights de IA: el polling (10 s, shape RF) queda DESHABILITADO por
  // PREDICTION_ENABLED=false hasta HU-03 — disparaba 422 consistente contra el
  // backend GRU (Delta-01). El cableado se preserva para revivirlo al migrar el
  // cliente al contrato GRU. Mientras tanto la card muestra "Predicción no disponible".
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  useEffect(() => {
    if (!PREDICTION_ENABLED) return; // ver PREDICTION_ENABLED arriba (deuda HU-03 / Delta-01)
    if (vehicles <= 0) return;
    let cancelled = false;
    const fetchPrediction = async () => {
      try {
        const result = await predictionService.predictTraffic({
          camera_id: cameraId,
          total_vehicles: vehicles,
          occupancy_rate: occupancy,
          flow_rate_per_min: flowPerHour / 60,
          avg_speed: speed ?? 0,
          avg_density: occupancy * 100,
        });
        if (!cancelled) setPrediction(result);
      } catch (err) {
        console.error('Prediction fetch failed', err);
      }
    };
    void fetchPrediction();
    const interval = window.setInterval(() => void fetchPrediction(), 10000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [cameraId, vehicles, flowPerHour, speed, occupancy]);

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      {/* Player + insights */}
      <div className="flex flex-col gap-4 lg:col-span-2">
        <div className="relative grid min-h-[300px] place-items-center overflow-hidden rounded-panel border border-line bg-black lg:min-h-[400px]">
          {error ? (
            <div className="flex flex-col items-center gap-3 px-6 text-center">
              <AlertTriangle className="h-9 w-9 text-warn" aria-hidden="true" />
              <p className="max-w-xs text-[12.5px] text-ink-2">{error}</p>
            </div>
          ) : streamUrl && !statusResolving ? (
            /* Switch del video según el toggle. BAJA → HlsPlayer (HLS directo de Claro,
               sin cajas, baja latencia): es un feed válido → SIEMPRE a opacity full.
               ALTA → AnnotatedCameraStream: frame anotado server-side (cajas ya
               dibujadas por el edge; enabled=isActive, espejo del gate del SSE).
               El atenuado de "sin inferencia" va sobre las MÉTRICAS, no sobre el video. */
            <div className="absolute inset-0">
              {isActive ? (
                <AnnotatedCameraStream
                  cameraId={cameraId}
                  enabled={isActive}
                  onStatusChange={handleStatus}
                />
              ) : (
                <HlsPlayer src={streamUrl} objectFit="contain" />
              )}
            </div>
          ) : null}

          {/* Spinner: resolviendo el estado inicial (GET) o activando (espera 1er frame). */}
          {!error && (statusResolving || (isActive && isLoading)) && (
            <div className="absolute inset-0 grid place-items-center gap-3 bg-black/60">
              <div className="flex flex-col items-center gap-3">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-brand/30 border-t-brand" />
                <span className="text-[10.5px] uppercase tracking-[0.12em] text-ink-2">
                  {statusResolving ? 'Conectando…' : 'Activando detección…'}
                </span>
              </div>
            </div>
          )}

          {/* Toggle Directo/Detección — control del player (in-panel, pegado al video),
              distinto del control de modo de vista del header. */}
          {streamUrl && !error && !statusResolving && (
            <div className="absolute right-3 top-3">
              <SegmentedControl
                ariaLabel="Modo del video"
                options={TOGGLE_OPTIONS}
                value={isActive ? 'alta' : 'baja'}
                onChange={handleToggle}
              />
            </div>
          )}

          {isActive && !error && (
            <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full border border-bad/30 bg-bad/10 px-3 py-1 backdrop-blur-md">
              <span className="h-2 w-2 animate-pulse rounded-full bg-bad" />
              <span className="text-[10px] font-extrabold uppercase tracking-[0.12em] text-bad">
                En vivo
              </span>
            </div>
          )}

          {/* Aviso blando (capacidad del edge / alta fallida): no tapa el video. */}
          {notice && !error && (
            <div className="absolute inset-x-3 bottom-3 flex items-center gap-2 rounded-ctl border border-warn/30 bg-black/70 px-3 py-2 backdrop-blur-md">
              <AlertTriangle className="h-4 w-4 shrink-0 text-warn" aria-hidden="true" />
              <span className="text-[11px] text-ink-2">{notice}</span>
            </div>
          )}
        </div>

        {/* Insights de IA — deshabilitado hasta HU-03 (ver PREDICTION_ENABLED) */}
        <section className="overflow-hidden rounded-card border border-brand/40 bg-linear-to-br from-brand/14 to-accent/5 p-3.5">
          <div className="mb-2 flex items-center gap-2 text-[10px] font-extrabold uppercase tracking-[0.12em] text-brand-2">
            <Sparkles size={12} aria-hidden="true" /> Insights de IA · CerebroVial
          </div>
          {prediction ? (
            <div className="flex flex-col gap-2.5">
              <p className="text-sm leading-[1.45] text-ink">{prediction.message}</p>
              <div className="grid grid-cols-3 gap-2 text-center">
                {[
                  { t: 'Ahora', v: prediction.data.current_congestion_level },
                  { t: '+15 min', v: prediction.data.predicted_congestion_15min },
                  { t: '+30 min', v: prediction.data.predicted_congestion_30min },
                ].map((c) => (
                  <div key={c.t} className="rounded-ctl border border-line bg-white/2 px-2 py-2">
                    <div className="text-[10px] text-ink-2">{c.t}</div>
                    <div className="num mt-0.5 text-[12.5px] font-bold text-ink">{c.v}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-[12.5px] italic text-ink-2">Predicción no disponible</p>
          )}
        </section>
      </div>

      {/* Métricas en vivo — derivadas de inferencia: atenuadas en BAJA (Directo) para
          señalar que NO hay detección activa. El video directo, en cambio, va full. */}
      <div
        className={`flex flex-col gap-4 transition-opacity duration-300 ${
          isActive ? 'opacity-100' : 'opacity-50'
        }`}
      >
        <section className="rounded-panel border border-line bg-linear-to-b from-white/4 to-white/1 p-3.5">
          <h3 className="mb-3 text-[12.5px] font-bold">Métricas en vivo</h3>
          <div className="grid grid-cols-2 gap-[9px]">
            <Metric
              icon={<Car size={12} aria-hidden="true" />}
              label="Vehículos detectados"
              value={String(vehicles)}
            />
            <Metric
              icon={<Activity size={12} aria-hidden="true" />}
              label="Vel. promedio"
              value={speed != null ? `~${Math.round(speed)}` : '—'}
              unit="km/h"
              caveat="aprox"
              title="Visión · velocidad experimental, sin calibrar (DEUDA-SPEED-CALIB)."
            />
            <Metric
              icon={<Waves size={12} aria-hidden="true" />}
              label="Flujo"
              value={String(flowPerHour)}
              unit="veh/h"
              caveat="estim"
              title="Visión · presencia extrapolada, no line-crossing."
            />
            <Metric
              icon={<Gauge size={12} aria-hidden="true" />}
              label="Densidad"
              value={density}
              caveat="estim"
              title="Visión · ocupación de zona por bboxes (presencia extrapolada)."
            />
          </div>
        </section>

        <section className="rounded-panel border border-line bg-linear-to-b from-white/4 to-white/1 p-3.5">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-[12.5px] font-bold">Estado del tráfico</h3>
            <StatusChip status={statusForLabel(congestion)}>{congestion.toUpperCase()}</StatusChip>
          </div>
          <div className="flex flex-col gap-3.5">
            <Bar label="Congestión" valueText={congestion} status={statusForLabel(congestion)} occupancy={occupancy} />
            <Bar label="Densidad vehicular" valueText={density} status="info" occupancy={occupancy} />
          </div>
          <p className="mt-3 text-[10.5px] leading-[1.4] text-ink-3">
            Métricas derivadas de visión computacional · presencia extrapolada, no aforo calibrado.
          </p>
        </section>
      </div>
    </div>
  );
}

function Metric({
  icon,
  label,
  value,
  unit,
  caveat,
  title,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  unit?: string;
  caveat?: string;
  title?: string;
}) {
  return (
    <div title={title} className="rounded-ctl border border-line bg-white/2 px-3 py-2.5">
      <div className="flex items-center gap-1 text-[10px] text-ink-2">
        {icon} {label}
        {caveat && (
          <span className="ml-auto text-[9px] font-bold uppercase tracking-[0.08em] text-warn">
            {caveat}
          </span>
        )}
      </div>
      <div className="num mt-1 flex items-baseline gap-1 text-lg font-bold text-ink">
        {value}
        {unit && <em className="text-[10.5px] font-semibold not-italic text-ink-2">{unit}</em>}
      </div>
    </div>
  );
}

function Bar({
  label,
  valueText,
  status,
  occupancy,
}: {
  label: string;
  valueText: string;
  status: Status | 'info';
  occupancy: number;
}) {
  const pct = Math.max(0, Math.min(100, Math.round(occupancy * 100)));
  const fill =
    status === 'bad' ? 'bg-bad' : status === 'warn' ? 'bg-warn' : status === 'info' ? 'bg-info' : 'bg-ok';
  return (
    <div>
      <div className="mb-1 flex justify-between text-[11px] text-ink-2">
        <span>{label}</span>
        <span className="num font-bold text-ink">{valueText}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-panel-2">
        <div
          className={`h-full transition-all duration-500 ease-fluid ${fill}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
