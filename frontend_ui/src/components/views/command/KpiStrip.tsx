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

function Caveat({ children }: { children: ReactNode }) {
  return (
    <span className="mt-1.5 block text-[9.5px] font-semibold uppercase tracking-[0.08em] text-ink-3">
      {children}
    </span>
  );
}

function spark(data: readonly number[], colorClass: string): ReactNode {
  return data.length > 1 ? (
    <Sparkline data={data} className={`mt-2 h-8 w-full ${colorClass}`} />
  ) : null;
}

export function KpiStrip({ idx, vel, flu, sem, pred, onShowPrediction, onOpenKpi }: KpiStripProps) {
  const open = (kind: KpiKind) => (onOpenKpi ? () => onOpenKpi(kind) : undefined);

  return (
    <section className="mb-[13px] grid grid-cols-2 gap-[11px] md:grid-cols-3 min-[1240px]:grid-cols-6">
      <KpiCard
        label="Índice de congestión · red"
        value={idx.value}
        unit={idx.value !== null ? '/100' : undefined}
        sparkClassName="text-warn"
        footer={
          idx.value === null ? (
            <Caveat>sin datos en ventana</Caveat>
          ) : (
            spark(idx.spark, 'text-warn')
          )
        }
        onClick={open('idx')}
      />

      <KpiCard
        label="Velocidad media"
        value={vel.value}
        unit={vel.value !== null ? 'km/h' : undefined}
        decimals={1}
        footer={
          <>
            {vel.value === null ? <Caveat>sin señal de visión</Caveat> : spark(vel.spark, 'text-bad')}
            <Caveat>visión · sin calibrar</Caveat>
          </>
        }
        onClick={open('vel')}
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
      />

      <KpiCard
        label="Flujo total"
        value={flu.value}
        unit={flu.value !== null ? 'veh/h' : undefined}
        footer={
          <>
            {flu.value === null ? <Caveat>sin señal de visión</Caveat> : spark(flu.spark, 'text-brand')}
            <Caveat>visión · presencia extrapolada</Caveat>
          </>
        }
        onClick={open('flu')}
      />

      <KpiCard
        label="Cruces en modo adaptativo"
        value={sem.active}
        unit={sem.active !== null ? `/${KNOWN_NODE_IDS.length}` : undefined}
        footer={<Caveat>nodos instrumentados</Caveat>}
        onClick={open('sem')}
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
      />
    </section>
  );
}
