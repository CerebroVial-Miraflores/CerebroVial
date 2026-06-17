/**
 * TomTomView (track EXPERIMENTAL, Fase A) — smoke de render.
 *
 * FASE 4 migración MapLibre: react-map-gl/maplibre se stubea (jsdom no monta
 * WebGL). La display key se lee EN EL RENDER, así que el env se controla por
 * test con vi.stubEnv (determinista, sin depender del .env local). Esto cierra el
 * rojo ambiental que tenía el .env con la key: stubeando vacío, la degradación
 * vuelve a verde.
 */
import React from 'react';
import { afterEach, describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TomTomView } from '../TomTomView';

vi.mock('react-map-gl/maplibre', () => ({
  Map: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="map">{children}</div>
  ),
  Source: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="tomtom-source">{children}</div>
  ),
  Layer: () => <div data-testid="tomtom-layer" />,
}));

afterEach(() => {
  vi.unstubAllEnvs();
});

describe('TomTomView — smoke (Fase A)', () => {
  it('monta sin crashear y muestra el rótulo de la vista', () => {
    render(<TomTomView />);
    expect(screen.getByTestId('map')).toBeInTheDocument();
    expect(screen.getByText('Tráfico en vivo')).toBeInTheDocument();
    expect(screen.getByText('EXPERIMENTAL')).toBeInTheDocument();
  });

  it('muestra la atribución TomTom obligatoria y visible (ToS 17.3)', () => {
    render(<TomTomView />);
    // El link de atribución (© AÑO TomTom → www.tomtom.com) es el requisito legal.
    const attribution = screen.getByRole('link', { name: /TomTom/ });
    expect(attribution).toBeInTheDocument();
    expect(attribution).toHaveAttribute('href', 'https://www.tomtom.com');
  });

  it('con VITE_TOMTOM_KEY monta la capa raster de flujo', () => {
    vi.stubEnv('VITE_TOMTOM_KEY', 'fake-display-key');
    render(<TomTomView />);
    expect(screen.getByTestId('tomtom-layer')).toBeInTheDocument();
    expect(screen.queryByText('VITE_TOMTOM_KEY')).not.toBeInTheDocument();
  });

  it('degrada con gracia sin VITE_TOMTOM_KEY (aviso visible, sin romper)', () => {
    vi.stubEnv('VITE_TOMTOM_KEY', '');
    render(<TomTomView />);
    expect(screen.getByText('VITE_TOMTOM_KEY')).toBeInTheDocument();
    expect(screen.queryByTestId('tomtom-layer')).not.toBeInTheDocument();
  });
});
