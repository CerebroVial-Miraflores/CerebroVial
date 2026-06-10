import { describe, expect, it } from 'vitest';
import { render } from '@testing-library/react';

import { buildSparklinePoints, Sparkline } from '../Sparkline';

describe('buildSparklinePoints', () => {
  it('normaliza min/max al viewBox 120x36 (margen 4 abajo / 6 arriba)', () => {
    // min=0 → y = 36-4-0 = 32; max=10 → y = 36-4-26 = 6
    expect(buildSparklinePoints([0, 10], 120, 36)).toBe('0.0,32.0 120.0,6.0');
  });

  it('serie plana no divide por cero', () => {
    expect(buildSparklinePoints([5, 5, 5], 120, 36)).toBe('0.0,32.0 60.0,32.0 120.0,32.0');
  });

  it('con menos de 2 puntos devuelve vacío', () => {
    expect(buildSparklinePoints([7])).toBe('');
  });
});

describe('Sparkline', () => {
  it('renderiza área + línea + punto final con currentColor', () => {
    const { container } = render(<Sparkline data={[1, 3, 2, 5]} className="text-warn" />);
    const polylines = container.querySelectorAll('polyline');
    expect(polylines).toHaveLength(2);
    expect(polylines[0]).toHaveAttribute('fill', 'currentColor');
    expect(polylines[1]).toHaveAttribute('stroke', 'currentColor');
    expect(container.querySelector('circle')).toHaveAttribute('fill', 'currentColor');
  });

  it('con un solo punto no renderiza nada', () => {
    const { container } = render(<Sparkline data={[7]} />);
    expect(container).toBeEmptyDOMElement();
  });
});
