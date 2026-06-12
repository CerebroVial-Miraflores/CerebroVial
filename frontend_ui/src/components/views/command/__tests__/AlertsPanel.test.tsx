/**
 * AlertsPanel — feed mock señalizado del comando (FASE 3).
 *
 * Fake timers (receta Toast.test): la entrada dinámica de los ~9 s, el
 * encadenado de "Reiniciar nodo" (2.8 s) y el cleanup al desmontar. El panel
 * monta bajo ToastProvider en el harness (el provider real vive en AppShell).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';

import { AlertsPanel } from '../AlertsPanel';
import { ToastProvider } from '../../../ui/Toast';
import { INCOMING_ALERT_DELAY_MS, INITIAL_ALERTS } from '../mockData';

function mountPanel() {
  const onOpenNode = vi.fn();
  const onShowPrediction = vi.fn();
  const view = render(
    <ToastProvider>
      <AlertsPanel onOpenNode={onOpenNode} onShowPrediction={onShowPrediction} />
    </ToastProvider>,
  );
  return { onOpenNode, onShowPrediction, view };
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('AlertsPanel — feed inicial', () => {
  it('muestra las alertas del mock con su DemoBadge y el conteo en el header', () => {
    mountPanel();
    expect(screen.getByText('Alertas priorizadas por IA')).toBeInTheDocument();
    expect(screen.getByText('Demo · datos simulados')).toBeInTheDocument();
    expect(screen.getByText(String(INITIAL_ALERTS.length))).toBeInTheDocument();
    expect(screen.getByText('Congestión crítica')).toBeInTheDocument();
    expect(screen.getByText('Hardware · Latencia alta')).toBeInTheDocument();
  });

  it('expande y colapsa un item (cuerpo con stats y acciones)', () => {
    mountPanel();
    const header = screen.getByRole('button', { name: /Congestión crítica/ });
    expect(header).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(header);
    expect(header).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('142 m')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Ver intersección' })).toBeInTheDocument();
  });
});

describe('AlertsPanel — entrada dinámica (~9 s, como el prototipo)', () => {
  it('a los 9 s entra la alerta nueva con animación, sube el conteo y dispara el toast pred', () => {
    mountPanel();
    expect(screen.queryByText('IA · Cola creciente')).not.toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(INCOMING_ALERT_DELAY_MS);
    });

    expect(screen.getByText('IA · Cola creciente')).toBeInTheDocument();
    expect(screen.getByText(String(INITIAL_ALERTS.length + 1))).toBeInTheDocument();
    expect(screen.getByText('Nueva alerta IA')).toBeInTheDocument(); // toast
    const item = screen.getByText('IA · Cola creciente').closest('.animate-new-alert');
    expect(item).not.toBeNull();
  });

  it('desmontar antes de los 9 s limpia el timer (sin toast fantasma ni setState)', () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const { view } = mountPanel();
    view.unmount();
    act(() => {
      vi.advanceTimersByTime(INCOMING_ALERT_DELAY_MS + 1000);
    });
    expect(screen.queryByText('Nueva alerta IA')).not.toBeInTheDocument();
    expect(errorSpy.mock.calls.filter((args) => String(args[0]).includes('act'))).toHaveLength(0);
    errorSpy.mockRestore();
  });
});

describe('AlertsPanel — acciones (toasts demo, D7)', () => {
  it('"Ver intersección" abre el detalle real del nodo del seed', () => {
    const { onOpenNode } = mountPanel();
    fireEvent.click(screen.getByRole('button', { name: /Congestión crítica/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Ver intersección' }));
    expect(onOpenNode).toHaveBeenCalledWith('cam_larco_benavides');
  });

  it('"Ver predicción" delega al mapa (modo predicción + flash)', () => {
    const { onShowPrediction } = mountPanel();
    fireEvent.click(screen.getByRole('button', { name: /Bloqueo probable/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Ver predicción' }));
    expect(onShowPrediction).toHaveBeenCalledTimes(1);
  });

  it('"Aplicar plan IA" → toast ok y botón en estado done (sin tocar capas reales)', () => {
    mountPanel();
    fireEvent.click(screen.getByRole('button', { name: /Congestión crítica/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Aplicar plan IA' }));

    expect(screen.getByText('Plan IA aplicado')).toBeInTheDocument();
    const done = screen.getByRole('button', { name: '✓ Aplicado' });
    expect(done).toBeDisabled();
  });

  it('"Reiniciar nodo" encadena toast info → toast ok a los 2.8 s', () => {
    mountPanel();
    fireEvent.click(screen.getByRole('button', { name: /Hardware · Latencia alta/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Reiniciar nodo' }));

    expect(screen.getByText('Comando enviado')).toBeInTheDocument();
    expect(screen.queryByText(/en línea/)).not.toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(2800);
    });
    expect(screen.getByText('Nodo Edge #4 en línea')).toBeInTheDocument();
  });

  it('acción genérica → toast «Acción de demo»', () => {
    mountPanel();
    fireEvent.click(screen.getByRole('button', { name: /Congestión crítica/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Escalar' }));
    expect(screen.getByText('Acción de demo')).toBeInTheDocument();
    expect(screen.getByText('«Escalar» · sin efecto')).toBeInTheDocument();
  });
});
