// FASE 3 (B3) — drawer de intersección del Centro de Comando (spec: .drawer
// del prototipo). Estado en la URL: abierto ⟺ ?nodo=<camera_id> (D3.2b: SIN
// flujo de visión on-demand — eso es Fase 4; la señal viene del agregador
// compartido useVisionAggregates).
//
// Política de paridad por sección:
// · Cámara: HlsPlayer REAL con stream_url; sin stream → vacío honesto.
// · Estado actual: SOLO métricas reales — velocidad/flujo de la muestra de
//   visión del nodo (REAL-CON-CAVEAT, mismos caveats del strip) + nivel del
//   tramo si la geometry/state lo resuelven. Cola/Ocupación del prototipo NO
//   se muestran (sin fuente) — decisión D6.
// · Ciclo semafórico: REAL cuando resoluble — nodeId = id sin prefijo "cam_"
//   (convención del seed, assertada en test backend) ∈ KNOWN_NODE_IDS →
//   useActiveStrategy + adapter para TrafficLightCycle TAL CUAL; 404
//   `no_active_state` → "Sin estrategia activa" honesto — decisión D5.
// · Recomendación IA + historial: MOCK con DemoBadge; acciones → toasts demo,
//   "Aplicar" pasa a done local SIN tocar mapa/KPIs (D7 — PROHIBIDO
//   /control/__internal/activate desde acá).
//
// El contenido va DENTRO de un componente hijo montado solo con el drawer
// abierto: HlsPlayer y useActiveStrategy no deben quedar vivos al cerrar
// (ui/Drawer mantiene children montados para animar el translate).
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Camera, ChevronDown, Sparkles } from 'lucide-react';

import { Drawer } from '../../ui/Drawer';
import { Button } from '../../ui/Button';
import { ConfBar } from '../../ui/ConfBar';
import { DemoBadge } from '../../ui/DemoBadge';
import { useToast } from '../../ui/Toast';
import { HlsPlayer } from '../../HlsPlayer';
import { TrafficLightCycle } from '../control/TrafficLightCycle';
import { KNOWN_NODE_IDS } from '../control/controlTypes';
import { useActiveStrategy } from '../../../hooks/useActiveStrategy';
import type { VisionSample } from '../../../hooks/useVisionAggregates';
import type { IntersectionSummary } from '../../../types/intersections';
import { adaptActiveState, drawerStatusFor } from './derive';
import { DRAWER_AI_MOCK, DRAWER_HISTORY_MOCK } from './mockData';

interface IntersectionDrawerProps {
  /** ?nodo= de la URL; null = cerrado. */
  cameraId: string | null;
  intersections: IntersectionSummary[] | null;
  intersectionsLoading: boolean;
  /** Última muestra de visión del nodo (agregador compartido). */
  vision: VisionSample | undefined;
  /** Nivel real del tramo que toca el nodo (derive.edgeLevelForNode); null = sin dato. */
  edgeLevel: number | null;
  onClose: () => void;
}

function SectionTitle({ children }: { children: string }) {
  return (
    <div className="flex items-center gap-[9px] text-[10px] font-extrabold uppercase tracking-[0.12em] text-ink-2">
      {children}
      <span aria-hidden="true" className="h-px flex-1 bg-line" />
    </div>
  );
}

function Stat({
  label,
  value,
  unit,
  title,
}: {
  label: string;
  value: string;
  unit?: string;
  title?: string;
}) {
  return (
    <div title={title} className="rounded-[11px] border border-line bg-white/2 px-3 py-2.5">
      <span className="block text-[9px] font-bold uppercase tracking-[0.1em] text-ink-2">
        {label}
      </span>
      <span className="num mt-[3px] flex items-baseline gap-[5px] text-lg font-bold">
        {value}
        {unit !== undefined && (
          <em className="text-[10.5px] font-semibold not-italic text-ink-2">{unit}</em>
        )}
      </span>
    </div>
  );
}

/** Cuerpo del drawer — montado SOLO con el drawer abierto (HLS/SSE vivos). */
function DrawerContent({
  camera,
  vision,
  edgeLevel,
  applied,
  onApplied,
}: {
  camera: IntersectionSummary;
  vision: VisionSample | undefined;
  edgeLevel: number | null;
  applied: boolean;
  onApplied: () => void;
}) {
  const navigate = useNavigate();
  const { push } = useToast();
  const [whyOpen, setWhyOpen] = useState(false);

  // Ciclo real resoluble (D5): convención cam_<node_id> del seed.
  const nodeId = camera.id.replace(/^cam_/, '');
  const instrumented = KNOWN_NODE_IDS.includes(nodeId);
  const strategy = useActiveStrategy(instrumented ? nodeId : '');

  const demoToast = (label: string) => {
    push({ kind: 'info', title: 'Acción de demo', body: `«${label}» · sin efecto` });
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Cámara (HLS real) */}
      <div className="relative aspect-video overflow-hidden rounded-ctl border border-line bg-canvas-2">
        {camera.stream_url !== null ? (
          <HlsPlayer src={camera.stream_url} />
        ) : (
          <div className="grid h-full place-items-center text-[11.5px] text-ink-3">
            <span className="flex items-center gap-2">
              <Camera size={14} aria-hidden="true" /> Sin stream registrado para esta cámara
            </span>
          </div>
        )}
      </div>
      <Button onClick={() => navigate(`/camara/${camera.id}`)}>
        Detalle de cámara
      </Button>

      {/* Estado actual — SOLO métricas reales (D6) */}
      <SectionTitle>Estado actual</SectionTitle>
      <div className="grid grid-cols-2 gap-[9px]">
        <Stat
          label="Velocidad"
          value={vision?.speed != null ? vision.speed.toFixed(1) : '—'}
          unit="km/h"
          title="Visión · velocidad experimental sin calibrar (DEUDA-SPEED-CALIB)."
        />
        <Stat
          label="Flujo"
          value={vision !== undefined ? String(Math.round(vision.flow)) : '—'}
          unit="veh/h"
          title="Visión · presencia extrapolada, no line-crossing (spike de flujo)."
        />
        <Stat
          label="Tramo"
          value={edgeLevel !== null ? `nivel ${edgeLevel}` : '—'}
          unit="/5"
          title="Máximo congestion_level (Waze) de las aristas que tocan el nodo."
        />
        <Stat
          label="Señal de visión"
          value={vision !== undefined ? 'activa' : 'sin señal'}
          title="Última muestra del SSE de visión del edge para esta cámara."
        />
      </div>

      {/* Ciclo semafórico — REAL cuando resoluble (D5) */}
      <SectionTitle>Ciclo semafórico</SectionTitle>
      {!instrumented ? (
        <p className="text-[11.5px] text-ink-2">
          Nodo sin instrumentación de control (fuera de los {KNOWN_NODE_IDS.length} nodos del
          plano de control del seed).
        </p>
      ) : strategy.errorStatus === 404 ? (
        <p className="text-[11.5px] text-ink-2">
          Sin estrategia activa para este nodo — el motor adaptativo no registró ninguna
          decisión vigente.
        </p>
      ) : strategy.error !== null ? (
        <div className="flex flex-wrap items-center gap-3 text-[11.5px]">
          <span className="text-bad">{strategy.error}</span>
          <Button onClick={() => void strategy.refetch()}>Reintentar</Button>
        </div>
      ) : (
        <TrafficLightCycle
          status={strategy.loading ? 'loading' : 'success'}
          data={strategy.data !== null ? adaptActiveState(strategy.data) : null}
          error={null}
        />
      )}

      {/* Recomendación IA — MOCK señalizado (D7) */}
      <div className="relative overflow-hidden rounded-card border border-brand/40 bg-linear-to-br from-brand/14 to-accent/5 p-3.5">
        <div className="mb-2 flex items-center gap-2 text-[10px] font-extrabold uppercase tracking-[0.12em] text-brand-2">
          <Sparkles size={12} aria-hidden="true" /> Recomendación del motor adaptativo
          <DemoBadge className="ml-auto" />
        </div>
        <p className="text-sm font-semibold leading-[1.45]">{DRAWER_AI_MOCK.recommendation}</p>
        <div className="my-2.5 flex flex-wrap gap-[7px]">
          {DRAWER_AI_MOCK.impacts.map((impact) => (
            <span
              key={impact}
              className="rounded-full border border-ok/40 bg-ok/13 px-2.5 py-1 text-[10.5px] font-extrabold text-ok"
            >
              {impact}
            </span>
          ))}
        </div>
        <ConfBar percent={DRAWER_AI_MOCK.confidence} />
        <button
          type="button"
          aria-expanded={whyOpen}
          onClick={() => setWhyOpen((v) => !v)}
          className="mt-2.5 flex w-full items-center gap-1.5 py-1 text-left text-[11.5px] font-semibold text-brand-2"
        >
          ¿Por qué esta decisión?
          <ChevronDown
            size={13}
            aria-hidden="true"
            className={`transition-transform duration-300 ease-fluid ${whyOpen ? 'rotate-180' : ''}`}
          />
        </button>
        <div
          className={`grid transition-[grid-template-rows] duration-[350ms] ease-fluid ${
            whyOpen ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'
          }`}
        >
          <div className="overflow-hidden">
            <ul className="mb-1 ml-4 mt-2 flex list-disc flex-col gap-[5px] text-[11.5px] text-ink-2">
              {DRAWER_AI_MOCK.why.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-[7px]">
          <Button
            variant={applied ? 'done' : 'pri'}
            onClick={() => {
              onApplied();
              // D7: SOLO toast + estado local — sin cascade sobre mapa/KPIs
              // reales y sin tocar /control/__internal/activate.
              push({
                kind: 'ok',
                title: 'Plan aplicado',
                body: 'Extender N–S +12 s · monitoreando efecto… · demo',
              });
            }}
          >
            {applied ? '✓ Aplicado · ciclo 1/3' : 'Aplicar'}
          </Button>
          <Button onClick={() => demoToast('Simular')}>Simular</Button>
          <Button onClick={() => demoToast('Rechazar')}>Rechazar</Button>
        </div>
      </div>

      {camera.status !== 'critical' && !applied && (
        <p className="rounded-ctl border border-dashed border-ok/40 bg-ok/5 px-3.5 py-3 text-xs leading-[1.5] text-ok">
          ✓ Operando en plan adaptativo · sin acciones pendientes para este cruce.
        </p>
      )}

      {/* Historial — MOCK señalizado */}
      <SectionTitle>Historial reciente</SectionTitle>
      <div>
        <DemoBadge className="mb-2" />
        {DRAWER_HISTORY_MOCK.map((row) => (
          <div
            key={row.time}
            className="flex gap-[11px] border-b border-dashed border-white/7 px-0.5 py-2 text-[11px] leading-[1.4] text-ink-2 last:border-0"
          >
            <span className="num shrink-0 pt-px text-[10.5px] text-ink-3">{row.time}</span>
            <span>{row.event}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function IntersectionDrawer({
  cameraId,
  intersections,
  intersectionsLoading,
  vision,
  edgeLevel,
  onClose,
}: IntersectionDrawerProps) {
  const open = cameraId !== null;
  const camera = open ? (intersections?.find((i) => i.id === cameraId) ?? null) : null;
  // Estado demo "aplicado" por nodo abierto: se resetea al cambiar de nodo.
  const [appliedFor, setAppliedFor] = useState<string | null>(null);
  const applied = appliedFor !== null && appliedFor === cameraId;

  const chip =
    camera !== null ? drawerStatusFor(camera.status, applied) : null;

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title={camera?.name ?? (intersectionsLoading ? 'Cargando intersección…' : 'Intersección')}
      status={chip?.status}
      statusLabel={chip?.label}
    >
      {/* children null al cerrar: HLS y SSE del ciclo no quedan vivos. */}
      {!open ? null : camera !== null ? (
        <DrawerContent
          camera={camera}
          vision={vision}
          edgeLevel={edgeLevel}
          applied={applied}
          onApplied={() => setAppliedFor(camera.id)}
        />
      ) : intersectionsLoading ? (
        <p className="text-[12.5px] text-ink-2" role="status">
          Cargando inventario de intersecciones…
        </p>
      ) : (
        <div className="flex flex-col items-start gap-3">
          <p className="text-[12.5px] text-ink-2">
            La intersección <span className="num">{cameraId}</span> no existe en el inventario.
          </p>
          <Button onClick={onClose}>Cerrar panel</Button>
        </div>
      )}
    </Drawer>
  );
}
