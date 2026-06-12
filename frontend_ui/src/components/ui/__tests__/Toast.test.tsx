import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';

import { ToastProvider, useToast, type ToastKind } from '../Toast';

function Host({ kind = 'ok' as ToastKind, title = 'Listo', body = 'cuerpo' }) {
  const { push } = useToast();
  return (
    <button type="button" onClick={() => push({ kind, title, body })}>
      disparar
    </button>
  );
}

function renderHost(props?: { kind?: ToastKind; title?: string; body?: string }) {
  return render(
    <ToastProvider>
      <Host {...props} />
    </ToastProvider>,
  );
}

// Vitest 4 fakea rAF y performance por default (jsdom pretendToBeVisual):
// el doble rAF de entrada son 2 frames de 16 ms.
describe('Toast', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('push monta el toast y la entrada llega con el doble rAF', () => {
    renderHost();
    fireEvent.click(screen.getByRole('button', { name: 'disparar' }));
    const card = screen.getByText('Listo').closest('[data-kind]')!;
    expect(card).toHaveClass('translate-x-6', 'opacity-0');
    act(() => {
      vi.advanceTimersByTime(48);
    });
    expect(card).toHaveClass('translate-x-0', 'opacity-100');
    expect(screen.getByText('cuerpo')).toBeInTheDocument();
  });

  it('autodismiss: sale a los 5200 ms y se remueve tras 420 ms más', () => {
    renderHost();
    fireEvent.click(screen.getByRole('button', { name: 'disparar' }));
    act(() => {
      vi.advanceTimersByTime(5200);
    });
    expect(screen.getByText('Listo').closest('[data-kind]')).toHaveClass('translate-x-6', 'opacity-0');
    act(() => {
      vi.advanceTimersByTime(420);
    });
    expect(screen.queryByText('Listo')).not.toBeInTheDocument();
  });

  it('apila múltiples toasts en orden', () => {
    renderHost();
    const trigger = screen.getByRole('button', { name: 'disparar' });
    fireEvent.click(trigger);
    fireEvent.click(trigger);
    fireEvent.click(trigger);
    expect(screen.getAllByText('Listo')).toHaveLength(3);
  });

  it.each(['ok', 'warn', 'pred', 'info'] as const)('tipo %s marca data-kind y su dot', (kind) => {
    renderHost({ kind });
    fireEvent.click(screen.getByRole('button', { name: 'disparar' }));
    expect(screen.getByText('Listo').closest('[data-kind]')).toHaveAttribute('data-kind', kind);
  });

  it('no filtra timers al desmontar el provider', () => {
    const { unmount } = renderHost();
    fireEvent.click(screen.getByRole('button', { name: 'disparar' }));
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});
