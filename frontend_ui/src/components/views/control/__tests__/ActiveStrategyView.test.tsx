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

  // === CA-05.4 (DHU-005 Caso B) — las TRES aserciones textuales =========
  // El CA dice: "el sistema MANTIENE la última estrategia conocida, la
  // MARCA como NO CONFIRMADA, e INDICA el TIEMPO transcurrido desde la
  // última confirmación". Las 3 partes se verifican por separado abajo —
  // si una se diluye, el test correspondiente falla.

  it('CA-05.4 (1/3): MANTIENE en pantalla la última estrategia conocida tras caída del stream', async () => {
    const captured = captureSseCallbacks();
    vi.mocked(controlActiveStateService.getActiveState).mockResolvedValue(
      successResponse,
    );

    render(<ActiveStrategyView />);
    await waitFor(() =>
      expect(
        screen.getByText('Optimización por demanda'),
      ).toBeInTheDocument(),
    );

    // Snapshot del contenido visible pre-error: subtítulo, ciclo, fases.
    expect(
      screen.getByText(/Reparte el verde según la demanda/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/ciclo 78\.5 s/i)).toBeInTheDocument();
    expect(screen.getByText('NS')).toBeInTheDocument();
    expect(screen.getByText('EW')).toBeInTheDocument();

    // Cae el stream.
    act(() => captured.onError?.(new Error('stream caído')));

    // Todo lo anterior sigue visible: la última estrategia conocida NO se
    // borra ni se reemplaza por vacío.
    expect(
      screen.getByText('Optimización por demanda'),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Reparte el verde según la demanda/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/ciclo 78\.5 s/i)).toBeInTheDocument();
    expect(screen.getByText('NS')).toBeInTheDocument();
    expect(screen.getByText('EW')).toBeInTheDocument();
  });

  it('CA-05.4 (2/3): MARCA visualmente como NO CONFIRMADA', async () => {
    const captured = captureSseCallbacks();
    vi.mocked(controlActiveStateService.getActiveState).mockResolvedValue(
      successResponse,
    );

    render(<ActiveStrategyView />);
    await waitFor(() =>
      expect(
        screen.getByText('Optimización por demanda'),
      ).toBeInTheDocument(),
    );

    // Antes del error la marca no está.
    expect(
      screen.queryByTestId('active-strategy-stale-marker'),
    ).not.toBeInTheDocument();

    act(() => captured.onError?.(new Error('stream caído')));

    const marker = screen.getByTestId('active-strategy-stale-marker');
    expect(marker).toBeInTheDocument();
    expect(marker).toHaveTextContent(/No confirmada/i);
  });

  it('CA-05.4 (3/3): INDICA el tiempo transcurrido desde la última confirmación, y AVANZA con el reloj', async () => {
    vi.useFakeTimers();
    try {
      const t0 = new Date(2026, 4, 26, 12, 0, 0);
      vi.setSystemTime(t0);

      const captured = captureSseCallbacks();
      vi.mocked(controlActiveStateService.getActiveState).mockResolvedValue(
        successResponse,
      );

      render(<ActiveStrategyView />);

      // Flush microtasks para que el fetch mockeado resuelva y se setee
      // lastConfirmedAt = t0.
      await act(async () => {
        await Promise.resolve();
      });

      expect(
        screen.getByText('Optimización por demanda'),
      ).toBeInTheDocument();

      // Caída del stream a t0 → entra a stale.
      act(() => captured.onError?.(new Error('stream caído')));

      // El contador debe existir y mostrar tiempo "hace 0 s" recién
      // entrado en stale.
      let elapsedEl = screen.getByTestId('active-strategy-stale-elapsed');
      expect(elapsedEl).toBeInTheDocument();
      expect(elapsedEl).toHaveTextContent(/Última confirmación hace/i);
      expect(elapsedEl).toHaveTextContent(/0\s?s/);

      // Avanzo 5 segundos: el setInterval del componente tickea y el
      // contador refleja el avance.
      await act(async () => {
        vi.advanceTimersByTime(5_000);
      });
      elapsedEl = screen.getByTestId('active-strategy-stale-elapsed');
      expect(elapsedEl).toHaveTextContent(/5\s?s/);

      // Avanzo otros 60 s para entrar en formato "min".
      await act(async () => {
        vi.advanceTimersByTime(60_000);
      });
      elapsedEl = screen.getByTestId('active-strategy-stale-elapsed');
      expect(elapsedEl).toHaveTextContent(/1\s?min/);
    } finally {
      vi.useRealTimers();
    }
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
