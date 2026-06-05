import { describe, it, expect } from 'vitest';
import { markerVisual, congestionColorClass } from '../markerVisual';

describe('congestionColorClass', () => {
  it('mapea el status de congestión a color', () => {
    expect(congestionColorClass('critical')).toBe('bg-red-500');
    expect(congestionColorClass('moderate')).toBe('bg-amber-500');
    expect(congestionColorClass('fluid')).toBe('bg-emerald-500');
    expect(congestionColorClass('lo-que-sea')).toBe('bg-emerald-500'); // default seguro
  });
});

describe('markerVisual — congestión (color) + salud (pulso/tachado)', () => {
  it('offline confirmado: NO pulsa y se tacha, conservando el color de congestión', () => {
    const v = markerVisual('critical', 'offline');
    expect(v).toEqual({ color: 'bg-red-500', pulse: false, struck: true });
  });

  it('desconocido (health undefined): pulsa normal, NUNCA offline', () => {
    const v = markerVisual('moderate', undefined);
    expect(v).toEqual({ color: 'bg-amber-500', pulse: true, struck: false });
  });

  it('playing: pulsa y no se tacha', () => {
    expect(markerVisual('fluid', 'playing')).toEqual({
      color: 'bg-emerald-500',
      pulse: true,
      struck: false,
    });
  });

  it('loading: tratado como vivo (pulsa, no tachado) — solo offline confirmado tacha', () => {
    expect(markerVisual('fluid', 'loading')).toEqual({
      color: 'bg-emerald-500',
      pulse: true,
      struck: false,
    });
  });
});
