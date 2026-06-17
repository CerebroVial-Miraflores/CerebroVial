import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import type { ReactNode } from 'react';

import { buildNodeIconHtml } from '../nodeIcon';
import { NodeMarker } from '../NodeMarker';

// FASE 4 migración MapLibre — mock per-file de react-map-gl/maplibre con captura
// de props del <Marker> (patrón heredado del mock de react-leaflet). El Marker
// stub invoca onClick con un evento sintético (originalEvent.stopPropagation),
// como hace MapLibre.
const captured = vi.hoisted(() => ({
  markerProps: null as Record<string, unknown> | null,
}));

vi.mock('react-map-gl/maplibre', () => ({
  Marker: ({
    children,
    onClick,
    ...props
  }: {
    children?: ReactNode;
    onClick?: (e: { originalEvent: { stopPropagation: () => void } }) => void;
  }) => {
    captured.markerProps = props;
    return (
      <button
        type="button"
        data-testid="marker"
        onClick={() => onClick?.({ originalEvent: { stopPropagation: () => {} } })}
      >
        {children}
      </button>
    );
  },
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
  it('ancla centrado y convierte position [lat,lng] → longitude/latitude', () => {
    render(<NodeMarker position={[-12.12, -77.03]} status="warn" />);
    expect(captured.markerProps).toMatchObject({
      longitude: -77.03,
      latitude: -12.12,
      anchor: 'center',
    });
  });

  it('renderiza el mismo SVG de nodeIcon con el token del estado', () => {
    const { container } = render(<NodeMarker position={[-12.12, -77.03]} status="warn" />);
    expect(container.innerHTML).toContain('var(--color-warn)');
  });

  it('click dispara el handler', () => {
    const onClick = vi.fn();
    render(<NodeMarker position={[-12.12, -77.03]} status="bad" critical onClick={onClick} />);
    fireEvent.click(screen.getByTestId('marker'));
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
