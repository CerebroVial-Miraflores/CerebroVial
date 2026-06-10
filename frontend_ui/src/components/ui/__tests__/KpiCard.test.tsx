import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';

import { KpiCard } from '../KpiCard';
import { formatNumber, useCountUp } from '../useCountUp';

function Probe({ target, decimals = 0, reduced = false }: { target: number; decimals?: number; reduced?: boolean }) {
  const value = useCountUp(target, { decimals, reducedMotion: reduced });
  return <span data-testid="probe">{value}</span>;
}

// Vitest 4 fakea rAF + performance por default: el loop de countUp (950 ms)
// avanza con advanceTimersByTime (frames de 16 ms con timestamp fake).
describe('useCountUp', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('arranca en 0 y llega al target', () => {
    render(<Probe target={1245} />);
    expect(screen.getByTestId('probe')).toHaveTextContent('0');
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByTestId('probe')).toHaveTextContent('1,245');
  });

  it('en un frame intermedio está entre 0 y el target', () => {
    render(<Probe target={1000} />);
    act(() => {
      vi.advanceTimersByTime(200);
    });
    const text = screen.getByTestId('probe').textContent!.replace(',', '');
    const value = Number(text);
    expect(value).toBeGreaterThan(0);
    expect(value).toBeLessThan(1000);
  });

  it('respeta decimales', () => {
    render(<Probe target={22.4} decimals={1} />);
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByTestId('probe')).toHaveTextContent('22.4');
  });

  it('reducedMotion salta directo al target sin animar', () => {
    render(<Probe target={1245} reduced />);
    expect(screen.getByTestId('probe')).toHaveTextContent('1,245');
  });
});

describe('formatNumber', () => {
  it('separa miles y usa el signo − (U+2212)', () => {
    expect(formatNumber(1245)).toBe('1,245');
    expect(formatNumber(-1245)).toBe('−1,245');
    expect(formatNumber(-18.3, 1)).toBe('−18.3');
    expect(formatNumber(0)).toBe('0');
  });
});

describe('KpiCard', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('muestra label, valor con countUp, unidad y delta', () => {
    render(
      <KpiCard
        label="Flujo total"
        value={1245}
        unit="veh/h"
        delta={{ text: '＋4% vs ayer', tone: 'neu' }}
        spark={[1, 2, 3]}
      />,
    );
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByText('Flujo total')).toBeInTheDocument();
    expect(screen.getByText('1,245')).toBeInTheDocument();
    expect(screen.getByText('veh/h')).toBeInTheDocument();
    expect(screen.getByText('＋4% vs ayer')).toHaveClass('text-ink-2');
  });

  it.each([
    ['pos', 'text-ok'],
    ['neg', 'text-bad'],
  ] as const)('delta %s usa su paleta', (tone, expected) => {
    render(<KpiCard label="X" value={1} delta={{ text: 'Δ', tone }} />);
    expect(screen.getByText('Δ')).toHaveClass(expected);
  });

  it('footer reemplaza al sparkline', () => {
    const { container } = render(
      <KpiCard label="Pred" value={94} spark={[1, 2, 3]} footer={<p>pie custom</p>} />,
    );
    expect(screen.getByText('pie custom')).toBeInTheDocument();
    expect(container.querySelector('svg')).toBeNull();
  });

  it('variante warn cambia el borde', () => {
    const { container } = render(<KpiCard label="Pred" value={94} warn />);
    expect(container.firstChild).toHaveClass('border-warn/40');
  });

  it('con onClick es botón y dispara; sin onClick no es botón', () => {
    const onClick = vi.fn();
    const { rerender } = render(<KpiCard label="X" value={1} onClick={onClick} />);
    fireEvent.click(screen.getByRole('button'));
    expect(onClick).toHaveBeenCalledTimes(1);
    rerender(<KpiCard label="X" value={1} />);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });
});
