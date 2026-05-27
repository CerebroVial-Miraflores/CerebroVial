// HU-05 / CA-05.4 — Integración: robustez ante pings de transporte SSE.
//
// Bug detectado en demo manual post-Fase 8: sse-starlette envía un PING
// (comment line) cada DEFAULT_PING_INTERVAL=15s para keep-alive de proxies.
// @microsoft/fetch-event-source dispatcha onmessage por la empty-line
// terminadora con ``{event:'', data:''}``. ANTES del fix esto disparaba
// fetchAndSet en ActiveStrategyView → reset del staleTimer → el banner
// "no confirmada" oscilaba (aparecía a +10s, desaparecía al ping de +15s,
// reaparecía a +25s, etc.) aunque el motor estuviera caído.
//
// Este test fija la conducta correcta integrada (View + sseClient real +
// fetch-event-source mockeado): pings llegando NO curan el stale; solo un
// active-state-changed real lo hace.
//
// Diferencia con ActiveStrategyView.test.tsx: ese archivo mockea ../sseClient
// DIRECTAMENTE — el filtro de sseClient no participa. Acá mockeamos la
// LIBRERÍA debajo, dejando el sseClient real en el medio. Por eso vive en
// archivo separado: dos vi.mock incompatibles del mismo eje no conviven.
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import type { EventSourceMessage } from '@microsoft/fetch-event-source';

vi.mock('@microsoft/fetch-event-source', () => ({
  fetchEventSource: vi.fn(),
  EventStreamContentType: 'text/event-stream',
}));

vi.mock('../../../../services/controlActiveStateService', () => ({
  controlActiveStateService: {
    getActiveState: vi.fn(),
  },
}));

import { fetchEventSource } from '@microsoft/fetch-event-source';
import { ActiveStrategyView } from '../ActiveStrategyView';
import { controlActiveStateService } from '../../../../services/controlActiveStateService';
import type { ActiveStateResponse } from '../../../../services/controlActiveStateService';

const successResponse: ActiveStateResponse = {
  node_id: 'larco_schell',
  strategy_mode: 'webster',
  cycle_seconds: 78.5,
  phase_timings: [
    { phase_id: 'NS', green: 39.1, yellow: 3, all_red: 2 },
    { phase_id: 'EW', green: 31.4, yellow: 3, all_red: 2 },
  ],
  decided_at: '2026-05-27T05:00:00Z',
  activated_at: '2026-05-27T05:01:00Z',
  activated_by: 'cli-test',
};

interface CapturedFES {
  onmessage?: (ev: EventSourceMessage) => void;
}

function captureFetchEventSourceCallbacks(): CapturedFES {
  const captured: CapturedFES = {};
  vi.mocked(fetchEventSource).mockImplementation((_url, opts) => {
    captured.onmessage = opts.onmessage;
    // La promise no resuelve nunca; el stream "queda abierto" durante el
    // test. El componente la cierra vía AbortController en su cleanup,
    // pero el test no llega a desmontar.
    return new Promise<void>(() => {});
  });
  return captured;
}

describe('ActiveStrategyView — pings de transporte SSE (CA-05.4 robustez real)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('motor caído + pings llegando: "no confirmada" PERSISTE (no oscila)', async () => {
    vi.useFakeTimers();
    try {
      vi.setSystemTime(new Date(2026, 4, 27, 12, 0, 0));
      const captured = captureFetchEventSourceCallbacks();
      vi.mocked(controlActiveStateService.getActiveState).mockResolvedValue(
        successResponse,
      );

      render(<ActiveStrategyView />);
      // Flush microtasks para que el fetch inicial resuelva y se renderice
      // el estado success.
      await act(async () => {
        await Promise.resolve();
      });

      expect(
        screen.getByText('Optimización por demanda'),
      ).toBeInTheDocument();
      const initialFetchCalls = vi.mocked(
        controlActiveStateService.getActiveState,
      ).mock.calls.length;

      // Avanzo 10 s → el staleTimer dispara → kind='stale'.
      await act(async () => {
        vi.advanceTimersByTime(10_000);
      });
      expect(
        screen.getByTestId('active-strategy-stale-marker'),
      ).toBeInTheDocument();

      // Simulo 5 pings sucesivos. Si el filtro de sseClient funciona, NINGUNO
      // debería disparar fetchAndSet ni cambiar el estado.
      for (let i = 0; i < 5; i++) {
        await act(async () => {
          captured.onmessage?.({
            event: '',
            data: '',
            id: '',
            retry: undefined,
          } as EventSourceMessage);
          await Promise.resolve();
        });
        // Avanzo 15 s entre pings (cadencia real de sse-starlette).
        await act(async () => {
          vi.advanceTimersByTime(15_000);
        });
      }

      // 1) El service NO se llamó más allá del fetch inicial — ningún ping
      //    disparó re-fetch.
      expect(
        vi.mocked(controlActiveStateService.getActiveState).mock.calls.length,
      ).toBe(initialFetchCalls);

      // 2) El banner "no confirmada" sigue ahí — NO osciló a lo largo de
      //    los 5 ciclos de ping × 15 s = 75 s.
      expect(
        screen.getByTestId('active-strategy-stale-marker'),
      ).toBeInTheDocument();
      expect(
        screen.getByText('Optimización por demanda'),
      ).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it('un evento active-state-changed REAL cura el stale (no regresión)', async () => {
    vi.useFakeTimers();
    try {
      vi.setSystemTime(new Date(2026, 4, 27, 12, 0, 0));
      const captured = captureFetchEventSourceCallbacks();
      vi.mocked(controlActiveStateService.getActiveState).mockResolvedValue(
        successResponse,
      );

      render(<ActiveStrategyView />);
      await act(async () => {
        await Promise.resolve();
      });

      // Llevo a stale.
      await act(async () => {
        vi.advanceTimersByTime(10_000);
      });
      expect(
        screen.getByTestId('active-strategy-stale-marker'),
      ).toBeInTheDocument();
      const callsBeforeRealEvent = vi.mocked(
        controlActiveStateService.getActiveState,
      ).mock.calls.length;

      // Llega un evento REAL del dominio.
      await act(async () => {
        captured.onmessage?.({
          event: 'active-state-changed',
          data: JSON.stringify({
            type: 'active-state-changed',
            node_id: 'larco_schell',
            decision_id: 'd-real',
            activated_at: '2026-05-27T12:00:30Z',
          }),
          id: '',
          retry: undefined,
        } as EventSourceMessage);
        // Dos ticks: uno para la promise del fetch, otro para flush del state.
        await Promise.resolve();
        await Promise.resolve();
      });

      // 1) El service SÍ se llamó tras el evento real.
      expect(
        vi.mocked(controlActiveStateService.getActiveState).mock.calls.length,
      ).toBeGreaterThan(callsBeforeRealEvent);

      // 2) El banner "no confirmada" desapareció — volvió a kind='success'.
      expect(
        screen.queryByTestId('active-strategy-stale-marker'),
      ).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});
