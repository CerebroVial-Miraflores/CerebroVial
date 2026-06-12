import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { ReactNode } from 'react';

import { MapCanvas } from '../MapCanvas';

// Mock per-file de react-leaflet con captura de props (patrón CongestionMapView.test).
const captured = vi.hoisted(() => ({
  map: null as Record<string, unknown> | null,
  tile: null as Record<string, unknown> | null,
}));

vi.mock('react-leaflet', () => ({
  MapContainer: ({ children, ...props }: { children?: ReactNode }) => {
    captured.map = props;
    return <div data-testid="map-container">{children}</div>;
  },
  TileLayer: (props: Record<string, unknown>) => {
    captured.tile = props;
    return <div data-testid="tile-layer" />;
  },
}));

describe('MapCanvas', () => {
  it('usa el TileLayer CartoDB dark con atribución OSM + CARTO', () => {
    render(<MapCanvas center={[-12.12, -77.03]} zoom={14} />);
    expect(String(captured.tile?.url)).toContain('basemaps.cartocdn.com/dark_all');
    expect(String(captured.tile?.attribution)).toContain('OpenStreetMap');
    expect(String(captured.tile?.attribution)).toContain('CARTO');
    expect(captured.map?.center).toEqual([-12.12, -77.03]);
    expect(captured.map?.zoom).toBe(14);
  });

  it('renderiza children dentro del mapa y los slots overlay', () => {
    render(
      <MapCanvas
        center={[-12.12, -77.03]}
        zoom={14}
        legend={<span>leyenda</span>}
        modeBadge={<span>EN VIVO</span>}
        coachTip={<span>tip</span>}
      >
        <span>capa-hija</span>
      </MapCanvas>,
    );
    expect(screen.getByTestId('map-container')).toHaveTextContent('capa-hija');
    expect(screen.getByText('leyenda')).toBeInTheDocument();
    expect(screen.getByText('EN VIVO')).toBeInTheDocument();
    expect(screen.getByText('tip')).toBeInTheDocument();
  });

  it('altura por flex (anti-calc) y stacking context propio (isolate)', () => {
    const { container } = render(<MapCanvas center={[0, 0]} zoom={10} />);
    expect(container.firstChild).toHaveClass('isolate', 'flex-1', 'min-h-0', 'relative');
  });
});
