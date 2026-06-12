// FASE 3 rediseño UI — strip de 6 KPI del Centro de Comando (spec: .kpis del
// prototipo; grid 2 col en 390px → 3 → 6 en ≥1240px).
//
// Política de paridad por card:
// · idx (índice de congestión) — REAL (useCongestionState); ventana vacía → "—".
// · vel/flu — REAL-CON-CAVEAT (agregación del SSE de visión): el dato es real
//   pero limitado y lo dice en la card — "visión · sin calibrar"
//   (DEUDA-SPEED-CALIB) y "visión · presencia extrapolada" (resultado del
//   spike de flujo). Sin DemoBadge: no es mock. Ver useVisionAggregates.
// · dem — MOCK con DemoBadge (sin fuente real de demora).
// · sem — REAL-PARCIAL (one-shot active-state × 5 nodos instrumentados).
// · pred — REAL degradado: hoy el endpoint responde 503 → "Servicio no
//   disponible" + Reintentar (NUNCA mock tapando un servicio caído). Sin
//   ConfBar (la respuesta real no publica confianza — D8). Click → modo
//   predicción del mapa (así lo hace el prototipo, no abre modal).
import type { ReactNode } from 'react';

import { Button } from '../../ui/Button';
import { DemoBadge } from '../../ui/DemoBadge';
import { KpiCard } from '../../ui/KpiCard';
import { Sparkline } from '../../ui/Sparkline';
import { KNOWN_NODE_IDS } from '../control/controlTypes';
import { DEM_KPI } from './mockData';

export type KpiKind = 'idx' | 'vel' | 'dem' | 'flu' | 'sem';

interface KpiStripProps {
  idx: { value: number | null; spark: readonly number[] };
  vel: { value: number | null; spark: readonly number[] };
  flu: { value: number | null; spark: readonly number[] };
  sem: { active: number | null };
  pred: { label: string | null; unavailable: boolean; onRetry: () => void };
  /** Card pred → cambia el mapa a modo predicción (+flash). */
  onShowPrediction: () => void;
  /** 3B: abre el KPI modal. Sin handler las cards no son clickeables. */
  onOpenKpi?: (kind: KpiKind) => void;
}

// Caveat de UNA línea (B0): el texto visible es corto y el `title` lleva la
// explicación completa — una card vacía se ve diseñada, no rota.
function Caveat({ children, title }: { children: ReactNode; title?: string }) {
  return (
    <span
      title={title}
      className="mt-1.5 block truncate text-[9.5px] font-semibold uppercase tracking-[0.08em] text-ink-3"
    >
      {children}
    </span>
  );
}

// Sparkline del strip (36px de alto, como el viewBox del prototipo). Con <3
// muestras todavía no hay tendencia que mostrar → placeholder deliberado:
// línea base tenue al 20% (no un hueco).
function spark(data: readonly number[], colorClass: string): ReactNode {
  if (data.length < 3) {
    return (
      <div aria-hidden="true" className={`mt-2 flex h-9 w-full items-end pb-[3px] ${colorClass}`}>
        <div className="h-px w-full bg-current opacity-20" />
      </div>
    );
  }
  return <Sparkline data={data} className={`mt-2 h-9 w-full ${colorClass}`} />;
}

// Densidad B0: min-h común nivela la fila; en ≥lg (fila de 6) los paddings se
// compactan para entrar en anchos lógicos retina (~1190px) calcando la
// densidad del prototipo.
const CARD_CLASS = 'min-h-[122px] lg:px-3 lg:pb-2 lg:pt-2.5';

export function KpiStrip({ idx, vel, flu, sem, pred, onShowPrediction, onOpenKpi }: KpiStripProps) {
  const open = (kind: KpiKind) => (onOpenKpi ? () => onOpenKpi(kind) : undefined);

  return (
    <section className="mb-[13px] grid grid-cols-2 gap-[11px] md:grid-cols-3 lg:grid-cols-6">
      <KpiCard
        label="Índice de congestión · red"
        value={idx.value}
        unit={idx.value !== null ? '/100' : undefined}
        sparkClassName="text-warn"
        footer={
          idx.value === null ? (
            <Caveat title="El estado de congestión no trae aristas en la ventana actual (feed Waze sin datos recientes).">
              sin datos en ventana
            </Caveat>
          ) : (
            spark(idx.spark, 'text-warn')
          )
        }
        onClick={open('idx')}
        className={CARD_CLASS}
      />

      <KpiCard
        label="Velocidad media"
        value={vel.value}
        unit={vel.value !== null ? 'km/h' : undefined}
        decimals={1}
        footer={
          vel.value === null ? (
            <Caveat title="Sin señal de visión del edge. La velocidad es experimental, sin calibración píxel→metro (DEUDA-SPEED-CALIB).">
              sin señal · sin calibrar
            </Caveat>
          ) : (
            <>
              {spark(vel.spark, 'text-bad')}
              <Caveat title="Velocidad experimental del pipeline de visión, sin calibración píxel→metro (DEUDA-SPEED-CALIB).">
                visión · sin calibrar
              </Caveat>
            </>
          )
        }
        onClick={open('vel')}
        className={CARD_CLASS}
      />

      <KpiCard
        label="Demora media · críticos"
        value={DEM_KPI.value}
        unit="s/veh"
        delta={{ text: '▲ +6', tone: 'neg' }}
        footer={
          <>
            {spark(DEM_KPI.spark, 'text-warn')}
            <DemoBadge className="mt-1.5" />
          </>
        }
        onClick={open('dem')}
        className={CARD_CLASS}
      />

      <KpiCard
        label="Flujo total"
        value={flu.value}
        unit={flu.value !== null ? 'veh/h' : undefined}
        footer={
          flu.value === null ? (
            <Caveat title="Sin señal de visión del edge. El flujo es presencia extrapolada, no conteo por line-crossing (resultado del spike de flujo).">
              sin señal · presencia extrapolada
            </Caveat>
          ) : (
            <>
              {spark(flu.spark, 'text-brand')}
              <Caveat title="Flujo por presencia extrapolada del pipeline de visión, no conteo por line-crossing (resultado del spike de flujo).">
                visión · presencia extrapolada
              </Caveat>
            </>
          )
        }
        onClick={open('flu')}
        className={CARD_CLASS}
      />

      <KpiCard
        label="Cruces en modo adaptativo"
        value={sem.active}
        unit={sem.active !== null ? `/${KNOWN_NODE_IDS.length}` : undefined}
        footer={
          <Caveat title="Nodos del plano de control con consulta de estrategia activa (KNOWN_NODE_IDS del seed).">
            nodos instrumentados
          </Caveat>
        }
        onClick={open('sem')}
        className={CARD_CLASS}
      />

      <KpiCard
        label="Predicción 15 min · red"
        warn
        value={null}
        valueLabel={
          pred.unavailable ? (
            <span className="text-[14px] font-semibold text-ink-2">Servicio no disponible</span>
          ) : pred.label !== null ? (
            <span className="text-[23px] text-warn">{pred.label}</span>
          ) : undefined
        }
        footer={
          pred.unavailable ? (
            <Button className="mt-2" onClick={pred.onRetry}>
              Reintentar
            </Button>
          ) : (
            <Caveat>media de red al horizonte +15 min</Caveat>
          )
        }
        // Degradada (503) la card NO es clickeable: evita el botón anidado con
        // Reintentar y el modo predicción sigue accesible por el SegmentedControl.
        onClick={pred.unavailable ? undefined : onShowPrediction}
        className={CARD_CLASS}
      />
    </section>
  );
}
