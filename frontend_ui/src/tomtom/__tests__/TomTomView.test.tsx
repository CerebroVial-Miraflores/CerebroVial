/**
 * TomTomView (track EXPERIMENTAL, Fase A) — smoke de render.
 *
 * No hay lógica pura que testear en Fase A (solo composición visual). Este smoke
 * afirma que la vista monta sin crashear y cumple lo no-negociable: el rótulo de
 * la vista y la atribución TomTom visibles (ToS 17.3). `react-leaflet` se stubea
 * (jsdom no monta Leaflet real), siguiendo el precedente de CongestionMapView.test.
 *
 * En test `VITE_TOMTOM_KEY` no está definida → la capa Flow degrada con gracia
 * (no monta) y se muestra el aviso; el smoke confirma esa degradación.
 */
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TomTomView } from '../TomTomView';

vi.mock('react-leaflet', () => ({
  MapContainer: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="map-container">{children}</div>
  ),
  TileLayer: () => <div data-testid="tile-layer" />,
}));

describe('TomTomView — smoke (Fase A)', () => {
  it('monta sin crashear y muestra el rótulo de la vista', () => {
    render(<TomTomView />);
    expect(screen.getByTestId('map-container')).toBeInTheDocument();
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

  it('degrada con gracia sin VITE_TOMTOM_KEY (aviso visible, sin romper)', () => {
    render(<TomTomView />);
    expect(screen.getByText('VITE_TOMTOM_KEY')).toBeInTheDocument();
  });
});
