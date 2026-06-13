/**
 * Tests de DetectionOverlay (FASE 4 Mitad A, D-019).
 *
 * Verifica el mapeo coords normalizadas → user-units del viewBox, el stroke por
 * tipo desde tokens, y que NO dibuja sin frame o sin cajas (overlay oculto).
 */
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';

import { DetectionOverlay } from '../DetectionOverlay';
import type { DetectionBox } from '../../../../types/detections';

const frame = { width: 1280, height: 720 };

function box(over: Partial<DetectionBox> = {}): DetectionBox {
  return { id: 'v1', type: 'car', confidence: 0.9, bbox: [0.1, 0.2, 0.3, 0.5], ...over };
}

describe('DetectionOverlay', () => {
  it('no dibuja sin frame', () => {
    const { container } = render(<DetectionOverlay boxes={[box()]} frame={null} />);
    expect(container.querySelector('svg')).toBeNull();
  });

  it('no dibuja sin cajas (stale/vacío → overlay oculto)', () => {
    const { container } = render(<DetectionOverlay boxes={[]} frame={frame} />);
    expect(container.querySelector('svg')).toBeNull();
  });

  it('viewBox usa las dims del frame (aspecto de la fuente, para el letterbox)', () => {
    const { container } = render(<DetectionOverlay boxes={[box()]} frame={frame} />);
    expect(container.querySelector('svg')?.getAttribute('viewBox')).toBe('0 0 1280 720');
    expect(container.querySelector('svg')?.getAttribute('preserveAspectRatio')).toBe('xMidYMid meet');
  });

  it('mapea bbox normalizado a user-units del viewBox', () => {
    const { container } = render(<DetectionOverlay boxes={[box({ bbox: [0.1, 0.2, 0.3, 0.5] })]} frame={frame} />);
    const rect = container.querySelector('rect')!;
    expect(rect.getAttribute('x')).toBe(String(0.1 * 1280)); // 128
    expect(rect.getAttribute('y')).toBe(String(0.2 * 720)); // 144
    expect(rect.getAttribute('width')).toBe(String((0.3 - 0.1) * 1280)); // 256
    expect(rect.getAttribute('height')).toBe(String((0.5 - 0.2) * 720)); // 216
    expect(rect.getAttribute('vector-effect')).toBe('non-scaling-stroke');
  });

  it('una caja por detección con stroke por tipo desde tokens', () => {
    const boxes = [box({ id: 'a', type: 'car' }), box({ id: 'b', type: 'bus' }), box({ id: 'c', type: 'truck' })];
    const { container } = render(<DetectionOverlay boxes={boxes} frame={frame} />);
    const rects = container.querySelectorAll('rect');
    expect(rects).toHaveLength(3);
    expect(rects[0].getAttribute('class')).toContain('stroke-ok'); // car
    expect(rects[1].getAttribute('class')).toContain('stroke-warn'); // bus
    expect(rects[2].getAttribute('class')).toContain('stroke-bad'); // truck
  });

  it('tipo desconocido cae al stroke por defecto', () => {
    const { container } = render(<DetectionOverlay boxes={[box({ type: 'bicycle' })]} frame={frame} />);
    expect(container.querySelector('rect')?.getAttribute('class')).toContain('stroke-brand');
  });
});
