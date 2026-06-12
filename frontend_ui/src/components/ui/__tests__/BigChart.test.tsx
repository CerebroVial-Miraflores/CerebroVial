/**
 * BigChart (FASE 3 B2) — geometría pura + render del forecast punteado y el
 * divisor "ahora" (patrón buildSparklinePoints escalado).
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import { BigChart, buildBigChartGeometry } from '../BigChart';

describe('buildBigChartGeometry', () => {
  it('serie de menos de 2 puntos → null', () => {
    expect(buildBigChartGeometry([])).toBeNull();
    expect(buildBigChartGeometry([5])).toBeNull();
  });

  it('4 gridlines equiespaciadas dentro del padding', () => {
    const g = buildBigChartGeometry([1, 2, 3], [], 720, 248, 26)!;
    expect(g.gridY).toHaveLength(4);
    expect(g.gridY[0]).toBe(26);
    expect(g.gridY[3]).toBe(248 - 26);
  });

  it('el divisor "ahora" cae en el último punto OBSERVADO, no en el del forecast', () => {
    const sinForecast = buildBigChartGeometry([10, 20, 30], [], 720, 248, 26)!;
    expect(sinForecast.nowX).toBeCloseTo(720 - 26, 1);

    const conForecast = buildBigChartGeometry([10, 20, 30], [40, 50], 720, 248, 26)!;
    // Con 2 puntos de forecast, el "ahora" (índice 2 de 4) queda al 50% del ancho útil.
    expect(conForecast.nowX).toBeCloseTo(26 + 0.5 * (720 - 52), 1);
    expect(conForecast.nowX).toBeLessThan(sinForecast.nowX);
  });

  it('sin forecast → forecastPoints null; con forecast arranca en el último observado', () => {
    expect(buildBigChartGeometry([1, 2, 3])!.forecastPoints).toBeNull();
    const g = buildBigChartGeometry([1, 2, 3], [4])!;
    expect(g.forecastPoints!.startsWith(g.linePoints.split(' ').at(-1)!)).toBe(true);
  });

  it('normaliza min/max sobre serie + forecast juntos (el forecast no se sale del viewBox)', () => {
    const g = buildBigChartGeometry([10, 20], [100], 720, 248, 26)!;
    const ys = g.forecastPoints!.split(' ').map((p) => Number(p.split(',')[1]));
    expect(Math.min(...ys)).toBeGreaterThanOrEqual(26);
  });
});

describe('BigChart — render', () => {
  it('dibuja la serie y la polyline punteada SOLO cuando hay forecast', () => {
    const { container, rerender } = render(
      <BigChart series={[1, 2, 3]} forecast={[4, 5]} xLabels={['−3h', 'ahora', '+45m']} />,
    );
    expect(screen.getByTestId('bigchart-forecast')).toHaveAttribute('stroke-dasharray', '6 6');
    expect(container.querySelectorAll('polyline')).toHaveLength(3); // área + línea + forecast

    rerender(<BigChart series={[1, 2, 3]} xLabels={['−3h', 'ahora']} />);
    expect(screen.queryByTestId('bigchart-forecast')).not.toBeInTheDocument();
    expect(container.querySelectorAll('polyline')).toHaveLength(2);
  });

  it('renderiza el marcador "ahora" y los labels del eje', () => {
    render(<BigChart series={[1, 2, 3]} xLabels={['−3h', '−2h', '−1h', 'ahora', '+45m']} />);
    // 'ahora' aparece como marcador del divisor Y como label del eje.
    expect(screen.getAllByText('ahora').length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText('−3h')).toBeInTheDocument();
    expect(screen.getByText('+45m')).toBeInTheDocument();
  });

  it('serie insuficiente → no renderiza nada (el vacío lo comunica el modal)', () => {
    const { container } = render(<BigChart series={[1]} />);
    expect(container.firstChild).toBeNull();
  });
});
