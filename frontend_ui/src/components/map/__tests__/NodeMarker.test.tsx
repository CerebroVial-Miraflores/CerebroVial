import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';

import { buildNodeIconHtml } from '../nodeIcon';
import { NodeMarker } from '../NodeMarker';

// Mock per-file de react-leaflet/leaflet con captura de props (patrón
// CongestionMapView.test — ver CLAUDE.md "Rediseño UI").
const captured = vi.hoisted(() => ({
  divIconArgs: null as Record<string, unknown> | null,
}));

vi.mock('react-leaflet', () => ({
  Marker: ({
    icon,
    eventHandlers,
  }: {
    icon: { html: string };
    eventHandlers?: { click?: () => void };
  }) => (
    <button type="button" data-testid="marker" onClick={eventHandlers?.click}>
      {icon.html}
    </button>
  ),
}));

vi.mock('leaflet', () => ({
  divIcon: vi.fn((args: Record<string, unknown>) => {
    captured.divIconArgs = args;
    return args;
  }),
}));

describe('buildNodeIconHtml (puro)', () => {
  it.each([
    ['ok', 'var(--color-ok)'],
    ['warn', 'var(--color-warn)'],
    ['bad', 'var(--color-bad)'],
  ] as const)('estado %s usa su token en style inline', (status, token) => {
    const html = buildNodeIconHtml(status);
    expect(html).toContain(`stroke: ${token}`);
    expect(html).toContain('var(--color-canvas-2)');
    expect(html).toContain('overflow="visible"');
    expect(html).not.toContain('animate-halo-ping');
  });

  it('critical agrega el halo animado con transform-box', () => {
    const html = buildNodeIconHtml('bad', { critical: true });
    expect(html).toContain('animate-halo-ping');
    expect(html).toContain('transform-box: fill-box');
  });
});

describe('NodeMarker', () => {
  it('construye el divIcon sin fondo de leaflet y con anchor centrado', () => {
    render(<NodeMarker position={[-12.12, -77.03]} status="warn" />);
    expect(captured.divIconArgs).toMatchObject({
      className: '',
      iconSize: [48, 48],
      iconAnchor: [24, 24],
    });
    expect(String(captured.divIconArgs?.html)).toContain('var(--color-warn)');
  });

  it('click dispara el handler', () => {
    const onClick = vi.fn();
    render(<NodeMarker position={[-12.12, -77.03]} status="bad" critical onClick={onClick} />);
    fireEvent.click(screen.getByTestId('marker'));
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
