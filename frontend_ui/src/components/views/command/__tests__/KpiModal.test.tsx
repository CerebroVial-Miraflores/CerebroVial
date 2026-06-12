/**
 * KpiModal (FASE 3 B3) — política D2 por card: idx REAL (serie del día en
 * contexto, vacío honesto), vel/flu/dem MOCK + DemoBadge, sem lista
 * REAL-PARCIAL sin "Reactivar".
 */
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';

import { KpiModal, type KpiModalIdxData } from '../KpiModal';
import type { UseAdaptiveNodesResult } from '../useAdaptiveNodes';

function idxData(overrides: Partial<KpiModalIdxData> = {}): KpiModalIdxData {
  return {
    series: null,
    loading: false,
    error: null,
    onRetry: vi.fn(),
    day: '2026-06-05',
    predictionUnavailable: true,
    ...overrides,
  };
}

const ADAPTIVE: UseAdaptiveNodesResult = {
  loading: false,
  activeCount: 1,
  refetch: vi.fn(),
  nodes: [
    {
      nodeId: 'larco_schell',
      kind: 'active',
      state: {
        node_id: 'larco_schell',
        strategy_mode: 'webster',
        cycle_seconds: 90,
        phase_timings: [{ phase_id: 'NS', green: 42, yellow: 4, all_red: 2 }],
        decided_at: 'x',
        activated_at: 'y',
        activated_by: null,
      },
    },
    { nodeId: 'larco_benavides', kind: 'no-strategy', state: null },
    { nodeId: 'ejercito_sucre', kind: 'failed', state: null },
  ],
};

function mountModal(kind: 'idx' | 'vel' | 'dem' | 'flu' | 'sem' | null, idx = idxData()) {
  const onClose = vi.fn();
  render(<KpiModal kind={kind} onClose={onClose} idx={idx} adaptive={ADAPTIVE} />);
  return { onClose };
}

describe('KpiModal — idx (serie REAL)', () => {
  it('con serie del día → BigChart + mstats Actual/Pico y nota de predicción no disponible', () => {
    mountModal(
      'idx',
      idxData({
        series: { points: [10, 40, 30], tLabels: ['08:00', '08:01', '08:02'] },
      }),
    );
    expect(screen.getByText('Índice de congestión · red')).toBeInTheDocument();
    expect(screen.getByText(/Día 2026-06-05 · serie real/)).toBeInTheDocument();
    expect(screen.getByText(/sin punto de predicción/)).toBeInTheDocument();
    expect(screen.getByText('30 /100')).toBeInTheDocument(); // Actual (último punto)
    expect(screen.getByText('40 /100')).toBeInTheDocument(); // Pico
    // Serie real SIN DemoBadge.
    expect(screen.queryByText('Demo · datos simulados')).not.toBeInTheDocument();
  });

  it('día sin datos → vacío honesto, sin chart', () => {
    mountModal('idx', idxData({ series: null }));
    expect(screen.getByText(/no tiene datos de congestión/)).toBeInTheDocument();
    expect(document.querySelector('polyline')).toBeNull();
  });

  it('error de la serie → mensaje + Reintentar', () => {
    const onRetry = vi.fn();
    mountModal('idx', idxData({ error: 'El servidor respondió 500.', onRetry }));
    fireEvent.click(screen.getByRole('button', { name: 'Reintentar' }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});

describe('KpiModal — vel/flu/dem (serie MOCK señalizada)', () => {
  it.each(['vel', 'flu', 'dem'] as const)('%s: BigChart mock con forecast + DemoBadge', (kind) => {
    mountModal(kind);
    expect(screen.getByText('Demo · datos simulados')).toBeInTheDocument();
    expect(screen.getByTestId('bigchart-forecast')).toBeInTheDocument();
    expect(screen.getByText(/Últimas 3 h · línea punteada/)).toBeInTheDocument();
    expect(screen.getByText('Predicción +45 min')).toBeInTheDocument();
  });
});

describe('KpiModal — sem (lista REAL-PARCIAL)', () => {
  it('lista los nodos con su estrategia/estado y SIN botón Reactivar (D2)', () => {
    mountModal('sem');
    expect(screen.getByText('Larco × Schell')).toBeInTheDocument();
    expect(screen.getByText(/Webster \(off-peak\) · 90s/)).toBeInTheDocument();
    expect(screen.getByText('Sin estrategia activa')).toBeInTheDocument();
    expect(screen.getByText('Sin dato (error al consultar)')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Reactivar/ })).not.toBeInTheDocument();
  });
});

describe('KpiModal — cierre', () => {
  it('kind null no renderiza nada; abierto cierra con Esc', () => {
    mountModal(null);
    expect(document.querySelector('[role="dialog"]')).toBeNull();

    const { onClose } = mountModal('sem');
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
