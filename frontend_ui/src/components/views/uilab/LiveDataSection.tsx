import { useState } from 'react';

import { Button } from '../../ui/Button';
import { Chip } from '../../ui/Chip';
import { StatusChip } from '../../ui/StatusChip';
import type { Status } from '../../ui/StatusChip';
import { useIntersections } from '../../../hooks/useIntersections';
import { useCongestionGeometry } from '../../../hooks/useCongestionGeometry';
import { useCongestionState } from '../../../hooks/useCongestionState';
import { useCongestionSeries } from '../../../hooks/useCongestionSeries';
import { useCongestionPrediction } from '../../../hooks/useCongestionPrediction';
import { useActiveStrategy } from '../../../hooks/useActiveStrategy';
import { useVisionStream } from '../../../hooks/useVisionStream';
import { usePredictionHistory } from '../../../hooks/usePredictionHistory';
import type { StreamConnectionState } from '../../../hooks/types';

// FASE 2 rediseño UI — sección "Datos en vivo (backend local)" del ui-lab.
//
// A diferencia del resto de la galería, esta sección consume el backend REAL
// (core 8001 vía httpClient con JWT) — por eso NO lleva DemoBadge. Es la
// verificación manual de la capa de hooks de la fase, sin esperar Fase 3:
// cada tarjeta monta un hook y muestra conteos, lastUpdated, isStale,
// connection y errores elegantes si el motor está apagado. Vive solo en el
// chunk lazy DEV-only de UiLabView.

/** Nodo seed del smoke de CLAUDE.md (CA-01.6 / HU-05). */
const KNOWN_NODE_ID = 'larco_schell';

const CONNECTION_STATUS: Record<StreamConnectionState, Status> = {
  connecting: 'warn',
  open: 'ok',
  retrying: 'warn',
  closed: 'bad',
};

function ConnectionChip({ connection }: { connection: StreamConnectionState }) {
  return (
    <StatusChip status={CONNECTION_STATUS[connection]}>
      SSE {connection.toUpperCase()}
    </StatusChip>
  );
}

function timeLabel(epochMs: number | null): string {
  return epochMs === null ? '—' : new Date(epochMs).toLocaleTimeString();
}

function HookCard({
  title,
  loading,
  error,
  onRetry,
  aside,
  children,
}: {
  title: string;
  loading: boolean;
  error: string | null;
  onRetry?: () => void;
  aside?: React.ReactNode;
  children?: React.ReactNode;
}) {
  return (
    <div className="space-y-3 rounded-card border border-line bg-panel-2 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-[11px] font-bold uppercase tracking-[0.08em] text-ink-2">{title}</h3>
        {aside}
      </div>
      {loading && <p className="text-[12px] text-ink-2">Cargando…</p>}
      {error && (
        <div className="space-y-2 rounded-ctl border border-bad/40 bg-bad/10 p-3">
          <p className="text-[12px] leading-relaxed text-bad">{error}</p>
          {onRetry && <Button onClick={onRetry}>Reintentar</Button>}
        </div>
      )}
      {children}
    </div>
  );
}

function Fact({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <p className="text-[12px] text-ink-2">
      {k}: <span className="num font-semibold text-ink">{v}</span>
    </p>
  );
}

export function LiveDataSection() {
  // Día local YYYY-MM-DD para la serie (initializer: sin Date en render).
  const [today] = useState(() => {
    const d = new Date();
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    return `${d.getFullYear()}-${mm}-${dd}`;
  });
  const [visionOn, setVisionOn] = useState(false);

  const intersections = useIntersections();
  const geometry = useCongestionGeometry();
  const congestion = useCongestionState({ staleAfterMs: 90_000 });
  const series = useCongestionSeries(today);
  const prediction = useCongestionPrediction();
  const strategy = useActiveStrategy(KNOWN_NODE_ID);

  // La tarjeta de history/visión reusa el primer id de la tarjeta 1; sin
  // lista todavía, los hooks quedan disabled ('' = no consulta).
  const firstCameraId = intersections.data?.[0]?.id ?? '';
  const history = usePredictionHistory(firstCameraId, 5);
  const vision = useVisionStream(firstCameraId, { enabled: visionOn });

  return (
    <div className="space-y-4">
      <p className="text-[12px] leading-relaxed text-ink-2">
        Esta sección consume el backend REAL (core 8001 vía httpClient con JWT) — a diferencia
        del resto de la galería, acá NO aplica el badge Demo. Requiere el motor levantado
        (<span className="num">invoke up</span>); con el backend apagado cada tarjeta muestra su
        error de red sin afectar a las demás. Verificar con rol operator; otros roles verán 403
        en las tarjetas de congestión.
      </p>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <HookCard
          title="useIntersections()"
          loading={intersections.loading}
          error={intersections.error}
          onRetry={() => void intersections.refetch()}
          aside={<Button onClick={() => void intersections.refetch()}>Actualizar</Button>}
        >
          {intersections.data && (
            <>
              <Fact k="Intersecciones" v={intersections.data.length} />
              <Fact
                k="Primeras"
                v={intersections.data.slice(0, 3).map((i) => i.name).join(' · ') || '—'}
              />
              <Fact k="Última actualización" v={timeLabel(intersections.lastUpdated)} />
            </>
          )}
        </HookCard>

        <HookCard
          title="useCongestionGeometry()"
          loading={geometry.loading}
          error={geometry.error}
          onRetry={() => void geometry.refetch()}
        >
          {geometry.data && (
            <>
              <Fact k="Tramos" v={geometry.data.features.length} />
              <p className="text-[11px] text-ink-3">
                Cache de sesión: remontar esta tarjeta no repite el GET (1660 tramos estáticos).
              </p>
            </>
          )}
        </HookCard>

        <HookCard
          title="useCongestionState({ staleAfterMs: 90 s })"
          loading={congestion.loading}
          error={congestion.error}
          onRetry={() => void congestion.refetch()}
          aside={
            <span className="flex items-center gap-2">
              <StatusChip status={congestion.isStale ? 'warn' : 'ok'}>
                {congestion.isStale ? 'STALE > 90 s' : 'FRESCO'}
              </StatusChip>
              <ConnectionChip connection={congestion.connection} />
            </span>
          }
        >
          {congestion.data && (
            <>
              <Fact k="Aristas con estado" v={congestion.data.edges.length} />
              <Fact k="Última actualización" v={timeLabel(congestion.lastUpdated)} />
              <Button onClick={() => void congestion.refetch()}>Refrescar</Button>
            </>
          )}
        </HookCard>

        <HookCard
          title={`useCongestionSeries('${today}')`}
          loading={series.loading}
          error={series.error}
          onRetry={() => void series.refetch()}
        >
          {series.data && (
            <>
              <Fact k="Aristas en serie" v={series.data.count} />
              <Fact k="t0" v={series.data.t0 ?? '—'} />
              <Fact k="Paso (s)" v={series.data.step_s ?? '—'} />
              <p className="text-[11px] text-ink-3">Día sin datos = count 0 (no es error).</p>
            </>
          )}
        </HookCard>

        <HookCard
          title="useCongestionPrediction()"
          loading={prediction.loading}
          error={prediction.error}
          onRetry={() => void prediction.refetch()}
        >
          {prediction.data && (
            <>
              <Fact k="base_timestep" v={prediction.data.base_timestep} />
              <Fact k="Horizonte" v={prediction.data.horizon} />
              <Fact k="Aristas" v={prediction.data.count} />
              <Fact k="Fuente" v={prediction.data.source} />
            </>
          )}
        </HookCard>

        <HookCard
          title={`useActiveStrategy('${KNOWN_NODE_ID}')`}
          loading={strategy.loading}
          error={strategy.error}
          onRetry={() => void strategy.refetch()}
          aside={<ConnectionChip connection={strategy.connection} />}
        >
          {strategy.data && (
            <>
              <Fact k="Modo" v={strategy.data.strategy_mode} />
              <Fact k="Ciclo (s)" v={strategy.data.cycle_seconds} />
              <Fact k="Activada" v={strategy.data.activated_at} />
            </>
          )}
        </HookCard>

        <HookCard
          title="usePredictionHistory(1ª cámara, 5 min)"
          loading={history.loading}
          error={history.error}
          onRetry={() => void history.refetch()}
        >
          {firstCameraId === '' && (
            <p className="text-[12px] text-ink-2">Esperando la lista de cámaras (tarjeta 1)…</p>
          )}
          {history.data && (
            <>
              <Fact k="Cámara" v={history.data.camera_id} />
              <Fact k="Puntos de historia" v={history.data.history.length} />
              <Fact
                k="Congestión +15 min"
                v={history.data.prediction?.predicted_congestion_15min ?? '—'}
              />
            </>
          )}
        </HookCard>

        <HookCard
          title="useVisionStream(1ª cámara)"
          loading={false}
          error={null}
          aside={visionOn ? <ConnectionChip connection={vision.connection} /> : undefined}
        >
          <Chip on={visionOn} onToggle={setVisionOn}>
            Conectar al edge
          </Chip>
          {visionOn && vision.data && (
            <>
              <Fact k="Vehículos únicos" v={vision.data.metrics.unique_vehicles} />
              <Fact k="Flujo (veh/h)" v={vision.data.metrics.flow_vehicles_per_hour} />
              <Fact k="Fin de ventana" v={vision.data.window.end} />
              <Fact k="Último payload" v={timeLabel(vision.lastUpdated)} />
            </>
          )}
          {visionOn && !vision.data && (
            <p className="text-[12px] text-ink-2">
              Sin payloads aún — el edge solo emite tras un alta on-demand (vista de detalle v1).
            </p>
          )}
          <p className="text-[11px] text-ink-3">
            SSE del edge sin JWT — deuda backend (CLAUDE.md §Rediseño UI). Default apagado: sin
            alta previa el edge no emite y EventSource reintentaría en loop.
          </p>
        </HookCard>
      </div>
    </div>
  );
}
