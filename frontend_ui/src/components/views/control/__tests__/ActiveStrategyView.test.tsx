// HU-05 / CA-05.1 — Guard rail DHU-006.
//
// El backend expone strategy_mode crudo ("webster" / "max_pressure"); el
// frontend DEBE mapearlo a etiqueta de dominio antes de mostrarlo al
// Operador. Este test falla si el render incluye el código técnico crudo
// en CUALQUIERA de los estados de la vista pasiva (loading, success, stale,
// error) o cuando el modo no está mapeado (fallback de dominio).
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

import { ActiveStrategyView } from '../ActiveStrategyView';
import { controlActiveStateService } from '../../../../services/controlActiveStateService';
import * as sseClientModule from '../../../../services/sseClient';
import type { ActiveStateResponse } from '../../../../services/controlActiveStateService';

vi.mock('../../../../services/controlActiveStateService', () => ({
  controlActiveStateService: {
    getActiveState: vi.fn(),
  },
}));

vi.mock('../../../../services/sseClient', () => ({
  openControlActiveStateStream: vi.fn(),
}));

const successResponse: ActiveStateResponse = {
  node_id: 'larco_schell',
  strategy_mode: 'webster',
  cycle_seconds: 78.5,
  phase_timings: [
    { phase_id: 'NS', green: 39.1, yellow: 3, all_red: 2 },
    { phase_id: 'EW', green: 31.4, yellow: 3, all_red: 2 },
  ],
  decided_at: '2026-05-26T20:30:00Z',
  activated_at: '2026-05-26T20:31:00Z',
  activated_by: 'cli-test',
};

const RAW_CODES = /\bwebster\b|\bmax_pressure\b/i;

/** Tipado del callback onError que sseClient publica al consumidor. */
type SSECapture = {
  onError?: (err: unknown) => void;
};

function captureSseCallbacks(): SSECapture {
  const captured: SSECapture = {};
  vi.mocked(sseClientModule.openControlActiveStateStream).mockImplementation(
    (_nodeId, opts) => {
      captured.onError = opts.onError;
      return new AbortController();
    },
  );
  return captured;
}

describe('ActiveStrategyView — DHU-006 guard rail (4 estados + fallback)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('LOADING — pending fetch, nunca renderiza códigos crudos', () => {
    captureSseCallbacks();
    // Promise que nunca resuelve mantiene la vista en loading.
    vi.mocked(controlActiveStateService.getActiveState).mockReturnValue(
      new Promise<ActiveStateResponse>(() => {}),
    );

    const { container } = render(<ActiveStrategyView />);

    expect(screen.getByText(/Cargando/i)).toBeInTheDocument();
    expect(container.textContent).not.toMatch(RAW_CODES);
  });

  it('SUCCESS / webster — muestra etiqueta de dominio, nunca el código crudo', async () => {
    captureSseCallbacks();
    vi.mocked(controlActiveStateService.getActiveState).mockResolvedValue(
      successResponse,
    );

    const { container } = render(<ActiveStrategyView />);

    await waitFor(() =>
      expect(
        screen.getByText('Optimización por demanda'),
      ).toBeInTheDocument(),
    );
    expect(container.textContent).not.toMatch(RAW_CODES);
  });

  it('SUCCESS / max_pressure — muestra etiqueta de dominio, nunca el código crudo', async () => {
    captureSseCallbacks();
    vi.mocked(controlActiveStateService.getActiveState).mockResolvedValue({
      ...successResponse,
      strategy_mode: 'max_pressure',
    });

    const { container } = render(<ActiveStrategyView />);

    await waitFor(() =>
      expect(
        screen.getByText('Prioridad por congestión'),
      ).toBeInTheDocument(),
    );
    expect(container.textContent).not.toMatch(RAW_CODES);
  });

  it('STALE — onError del SSE mantiene última etiqueta + marca "no confirmada", sin códigos crudos', async () => {
    const captured = captureSseCallbacks();
    vi.mocked(controlActiveStateService.getActiveState).mockResolvedValue(
      successResponse,
    );

    const { container } = render(<ActiveStrategyView />);

    // Espera al render exitoso primero.
    await waitFor(() =>
      expect(
        screen.getByText('Optimización por demanda'),
      ).toBeInTheDocument(),
    );

    // Dispara error de SSE → la vista debería transicionar a 'stale'.
    expect(captured.onError).toBeDefined();
    act(() => {
      captured.onError?.(new Error('stream caído'));
    });

    await waitFor(() =>
      expect(
        screen.getByTestId('active-strategy-stale-marker'),
      ).toBeInTheDocument(),
    );
    // La etiqueta de dominio sigue ahí + marca "No confirmada".
    expect(screen.getByText('Optimización por demanda')).toBeInTheDocument();
    expect(screen.getByText(/No confirmada/i)).toBeInTheDocument();
    expect(container.textContent).not.toMatch(RAW_CODES);
  });

  it('ERROR — fetch falla sin datos previos: mensaje neutral, nunca códigos crudos', async () => {
    captureSseCallbacks();
    vi.mocked(controlActiveStateService.getActiveState).mockRejectedValue(
      new Error('network'),
    );

    const { container } = render(<ActiveStrategyView />);

    await waitFor(() =>
      expect(
        screen.getByText(/No se pudo cargar la estrategia vigente/i),
      ).toBeInTheDocument(),
    );
    expect(container.textContent).not.toMatch(RAW_CODES);
  });

  it('FALLBACK — strategy_mode desconocido cae a etiqueta de dominio, nunca al crudo', async () => {
    captureSseCallbacks();
    vi.mocked(controlActiveStateService.getActiveState).mockResolvedValue({
      ...successResponse,
      // Forzamos un valor que no está mapeado. La firma TS lo prohíbe, lo
      // bypassamos con un cast porque el guard rail existe precisamente para
      // cubrir el caso "el backend devolvió algo nuevo que el frontend
      // todavía no conoce".
      strategy_mode: 'gru_optimal' as ActiveStateResponse['strategy_mode'],
    });

    const { container } = render(<ActiveStrategyView />);

    await waitFor(() =>
      expect(
        screen.getByText('Estrategia no reconocida'),
      ).toBeInTheDocument(),
    );
    // No aparece el crudo "gru_optimal" ni los reservados webster / max_pressure.
    expect(container.textContent).not.toMatch(/gru_optimal/i);
    expect(container.textContent).not.toMatch(RAW_CODES);
  });
});
