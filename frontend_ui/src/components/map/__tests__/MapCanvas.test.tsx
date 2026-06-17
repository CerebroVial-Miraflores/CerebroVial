import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { ReactNode } from 'react';

import { MapCanvas } from '../MapCanvas';

// FASE 4 migración MapLibre — mock per-file de react-map-gl/maplibre con captura
// de props del <Map> (initialViewState + mapStyle). El basemap CartoDB dark vive
// ahora en mapStyle (raster source), no en un TileLayer.
const captured = vi.hoisted(() => ({
  map: null as Record<string, unknown> | null,
}));

vi.mock('react-map-gl/maplibre', () => ({
  Map: ({ children, ...props }: { children?: ReactNode }) => {
    captured.map = props;
    return <div data-testid="map">{children}</div>;
  },
}));

function rasterTiles(style: unknown): string[] {
  const s = style as { sources: Record<string, { tiles: string[] }> };
  return s.sources['carto-dark'].tiles;
}

function rasterAttribution(style: unknown): string {
  const s = style as { sources: Record<string, { attribution: string }> };
  return s.sources['carto-dark'].attribution;
}

describe('MapCanvas', () => {
  it('usa el basemap raster CartoDB dark con atribución OSM + CARTO', () => {
    render(<MapCanvas center={[-12.12, -77.03]} zoom={14} />);
    expect(rasterTiles(captured.map?.mapStyle)[0]).toContain('basemaps.cartocdn.com/dark_all');
    expect(rasterAttribution(captured.map?.mapStyle)).toContain('OpenStreetMap');
    expect(rasterAttribution(captured.map?.mapStyle)).toContain('CARTO');
  });

  it('convierte center [lat,lng] → initialViewState [lng,lat] con el zoom', () => {
    render(<MapCanvas center={[-12.12, -77.03]} zoom={14} />);
    expect(captured.map?.initialViewState).toEqual({
      longitude: -77.03,
      latitude: -12.12,
      zoom: 14,
    });
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
    expect(screen.getByTestId('map')).toHaveTextContent('capa-hija');
    expect(screen.getByText('leyenda')).toBeInTheDocument();
    expect(screen.getByText('EN VIVO')).toBeInTheDocument();
    expect(screen.getByText('tip')).toBeInTheDocument();
  });

  it('forwarda interactiveLayerIds y cursor para el tooltip de arista', () => {
    render(
      <MapCanvas
        center={[0, 0]}
        zoom={10}
        interactiveLayerIds={['command-edges']}
        cursor="pointer"
      />,
    );
    expect(captured.map?.interactiveLayerIds).toEqual(['command-edges']);
    expect(captured.map?.cursor).toBe('pointer');
  });

  it('altura por flex (anti-calc) y stacking context propio (isolate)', () => {
    const { container } = render(<MapCanvas center={[0, 0]} zoom={10} />);
    expect(container.firstChild).toHaveClass('isolate', 'flex-1', 'min-h-0', 'relative');
  });
});
