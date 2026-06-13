import { useCallback, useEffect, useState } from 'react';
import { Activity, AlertTriangle, Car, Gauge, Sparkles, Waves } from 'lucide-react';

import { useVisionStream } from '../../../hooks/useVisionStream';
import { useDetections } from '../../../hooks/useDetections';
import { predictionService, type PredictionResult } from '../../../services/predictionService';
import { congestionLabel, densityPercent } from '../../../utils/trafficLabels';
import type { Status } from '../../ui/StatusChip';
import { StatusChip } from '../../ui/StatusChip';
import { HlsPlayer, type PlayerStatus } from '../../HlsPlayer';
import { DetectionOverlay } from './DetectionOverlay';

// FASE 4 Mitad A — panel EN VIVO del detalle de cámara. Orquesta el flujo
// delicado de v1 (CameraDetailView), preservado 1:1 sobre el design system:
//  - alta on-demand del YOLO en el edge (POST /cameras/{id}, sin JWT — deuda
//    edge intacta; SIN DELETE en cleanup, lección StrictMode de v1);
//  - SSE de métricas vía useVisionStream (gate isActive);
//  - VIDEO HLS directo de Claro vía HlsPlayer (object-fit:contain), reemplaza al
//    MJPEG entrecortado del edge (DEUDA-MJPEG-BROWSER resuelta para esta vista);
//  - OVERLAY de cajas (DetectionOverlay) sobre el video, alimentado por el polling
//    de useDetections (GET /detections/{id}/latest). El POST de alta ahora alimenta
//    el overlay (la inferencia), no el video — por eso se mantiene intacto;
//  - Insights de IA vía predictionService.predictTraffic (backend real).
// Política de datos: vehículos = REAL (conteo YOLO); flujo/ocupación =
// REAL-CON-CAVEAT (presencia extrapolada); velocidad = REAL-CON-CAVEAT (sin
// calibrar, DEUDA-SPEED-CALIB). Peatones/incidentes de v1 NO migran (eran 0
// hardcodeado, dato inventado). Controles de dispositivo de v1 NO migran (eran
// botones sin handler ni backend).

const EDGE_API_URL: string = import.meta.env?.VITE_EDGE_API_URL || 'http://localhost:8000';

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
  // Alta on-demand del YOLO. Init perezoso (el panel se re-monta por cámara vía
  // `key`): cargando si hay stream que dar de alta; error directo si no lo hay.
  const [isActive, setIsActive] = useState(false);
  const [isLoading, setIsLoading] = useState(() => !!streamUrl);
  const [error, setError] = useState<string | null>(
    () => (streamUrl ? null : 'Esta cámara no tiene un stream configurado.'),
  );

  // POST de alta al montar. El front SOLO hace POST (idempotente en el edge);
  // la BAJA la posee el edge (single-slot + watchdog). NUNCA un DELETE en el
  // cleanup: bajo <StrictMode> aterrizaría tras el POST del remonte y removería
  // la cámara recién dada de alta ("Cargando…" eterno). Lección de v1.
  useEffect(() => {
    if (!streamUrl) return;
    let cancelled = false;

    fetch(`${EDGE_API_URL}/cameras/${cameraId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: streamUrl, source_type: 'hls', zones: {} }),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`Edge respondió ${res.status}`);
        if (!cancelled) setIsActive(true);
      })
      .catch((err) => {
        console.error('Alta de cámara en el edge falló', err);
        if (!cancelled) {
          setError('No se pudo iniciar la detección. La cámara de Claro podría no estar disponible.');
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [cameraId, streamUrl]);

  // SSE de métricas: solo tras el alta OK (isActive). El hook auto-resetea sus
  // métricas al cambiar de cámara/enabled.
  const vision = useVisionStream(cameraId, { enabled: isActive });
  const metrics = vision.data?.metrics ?? null;

  // Polling de detecciones para el overlay (gate isActive, igual que el SSE de
  // métricas). El hook ya filtra por frescura (~3s, reloj del edge).
  const detections = useDetections(cameraId, { enabled: isActive });

  // isLoading se limpia cuando el video HLS empieza a reproducir (ya no depende
  // del MJPEG). 'offline' del player → señal honesta de pérdida de señal.
  const handleStatus = useCallback((status: PlayerStatus) => {
    if (status === 'playing') {
      setIsLoading(false);
    } else if (status === 'offline') {
      setIsLoading(false);
      setError('Se perdió la señal de la cámara.');
    }
  }, []);

  const vehicles = metrics ? Math.round(metrics.unique_vehicles) : 0;
  const flowPerHour = metrics ? Math.round(metrics.flow_vehicles_per_hour) : 0;
  const speed = metrics?.mean_speed_kmh ?? null;
  const occupancy = metrics?.mean_occupancy ?? 0;
  const congestion = congestionLabel(occupancy);
  const density = densityPercent(occupancy);

  // Insights de IA: predicción del modelo servido alimentada por las métricas de
  // visión. Polling 10 s (igual que v1). El `source` del endpoint /predictions
  // queda hardcodeado en el backend — deuda del cierre del STGNN servido, NO se
  // toca acá.
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  useEffect(() => {
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
          ) : streamUrl ? (
            <>
              {/* Video HLS directo de Claro (fluido), independiente del alta. */}
              <HlsPlayer
                src={streamUrl}
                controls={false}
                objectFit="contain"
                onStatusChange={handleStatus}
              />
              {/* Overlay de cajas: solo con el pipeline activo (isActive); el hook
                  ya devuelve [] si las detecciones están stale o hay error. */}
              {isActive && <DetectionOverlay boxes={detections.boxes} frame={detections.frame} />}
            </>
          ) : null}

          {isLoading && !error && (
            <div className="absolute inset-0 grid place-items-center gap-3 bg-black/60">
              <div className="flex flex-col items-center gap-3">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-brand/30 border-t-brand" />
                <span className="text-[10.5px] uppercase tracking-[0.12em] text-ink-2">
                  Activando detección…
                </span>
              </div>
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
        </div>

        {/* Insights de IA — predicción del modelo servido (dato real) */}
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
            <p className="text-[12.5px] italic text-ink-2">Analizando patrones…</p>
          )}
        </section>
      </div>

      {/* Métricas en vivo */}
      <div className="flex flex-col gap-4">
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
