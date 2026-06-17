import { describe, expect, it } from 'vitest';

import { JAM_LEVEL_LEGEND, jamLevelPaint } from '../edgeStyle';

// FASE 4 migración MapLibre — el estilo es un paint data-driven (expresión
// `match`) en vez de PathOptions/className de Leaflet. Helpers para leer un valor
// por nivel de una expresión ['match', input, l0, v0, l1, v1, ..., fallback].
function matchInput(expr: unknown): unknown {
  return (expr as unknown[])[1];
}
function matchFallback(expr: unknown): unknown {
  const arr = expr as unknown[];
  return arr[arr.length - 1];
}
function matchValue(expr: unknown, level: number): unknown {
  const arr = expr as unknown[];
  for (let i = 2; i < arr.length - 1; i += 2) {
    if (arr[i] === level) return arr[i + 1];
  }
  return undefined;
}

describe('jamLevelPaint — color', () => {
  const paint = jamLevelPaint();
  const color = paint?.['line-color'];

  it('lee congestion_level por default', () => {
    expect(matchInput(color)).toEqual(['get', 'congestion_level']);
  });

  it.each([
    [0, '#0fae79'],
    [1, '#0fae79'],
    [2, '#f59e0b'],
    [3, '#f59e0b'],
    [4, '#ef4444'],
    [5, '#a855f7'],
  ])('nivel %i → %s (mapeo cromático histórico)', (level, hex) => {
    expect(matchValue(color, level)).toBe(hex);
  });

  it('sin dato (fallback) → neutro ink-3', () => {
    expect(matchFallback(color)).toBe('#5b6275');
  });

  it('acepta otra propiedad (demo /ui-lab usa jam_level)', () => {
    expect(matchInput(jamLevelPaint('jam_level')?.['line-color'])).toEqual(['get', 'jam_level']);
  });
});

describe('jamLevelPaint — grosor (escala fina 2→5, CA-22.3)', () => {
  const width = jamLevelPaint()?.['line-width'];

  it('arranca en 2 y crece hasta 5', () => {
    expect(matchValue(width, 0)).toBe(2);
    expect(matchValue(width, 5)).toBe(5);
    expect(matchFallback(width)).toBe(2);
  });

  it('los pesos crecen (o se mantienen) con la severidad — redundancia no-cromática', () => {
    const weights = [0, 1, 2, 3, 4, 5].map((l) => matchValue(width, l) as number);
    for (let i = 1; i < weights.length; i++) {
      expect(weights[i]).toBeGreaterThanOrEqual(weights[i - 1]);
    }
  });
});

describe('jamLevelPaint — opacidad', () => {
  const opacity = jamLevelPaint()?.['line-opacity'];

  it('nivel crítico opaco; sin dato tenue', () => {
    expect(matchValue(opacity, 4)).toBe(1);
    expect(matchValue(opacity, 5)).toBe(1);
    expect(matchFallback(opacity)).toBe(0.35);
  });
});

describe('JAM_LEVEL_LEGEND', () => {
  it('expone la escala semántica de 4 clases con swatches de tokens', () => {
    expect(JAM_LEVEL_LEGEND.map((i) => i.swatchClass)).toEqual([
      'bg-ok-road',
      'bg-warn',
      'bg-bad',
      'bg-sev',
    ]);
  });
});
