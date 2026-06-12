import { describe, expect, it } from 'vitest';

import { JAM_LEVEL_LEGEND, jamLevelPathOptions, jamLevelStyle } from '../edgeStyle';

describe('jamLevelStyle', () => {
  it.each([
    [0, 'edge-jam-0'],
    [1, 'edge-jam-1'],
    [2, 'edge-jam-2'],
    [3, 'edge-jam-3'],
    [4, 'edge-jam-4'],
    [5, 'edge-jam-5'],
  ])('nivel %i → %s', (level, className) => {
    expect(jamLevelStyle(level).className).toBe(className);
  });

  it.each([null, undefined, NaN, 2.5, -1, 6])('inválido (%s) → neutro tenue', (level) => {
    const style = jamLevelStyle(level as number | null | undefined);
    expect(style.className).toBe('edge-jam-none');
    expect(style.opacity).toBe(0.35);
    expect(style.weight).toBe(3);
  });

  it('los pesos crecen con la severidad (3 → 9)', () => {
    const weights = [0, 1, 2, 3, 4, 5].map((l) => jamLevelStyle(l).weight);
    for (let i = 1; i < weights.length; i++) {
      expect(weights[i]).toBeGreaterThanOrEqual(weights[i - 1]);
    }
    expect(weights[0]).toBe(3);
    expect(weights[5]).toBe(9);
  });
});

describe('jamLevelPathOptions', () => {
  it('traduce a PathOptions con className y sin fill', () => {
    expect(jamLevelPathOptions(5)).toEqual({
      className: 'edge-jam-5',
      weight: 9,
      opacity: 1,
      fill: false,
    });
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
