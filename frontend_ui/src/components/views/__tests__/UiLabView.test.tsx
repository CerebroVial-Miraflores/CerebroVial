import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { ReactNode } from 'react';

import UiLabView from '../UiLabView';
import { ToastProvider } from '../../ui/Toast';

// Smoke de la galería /ui-lab (solo DEV). Mock per-file de react-map-gl/maplibre
// (FASE 4 migración MapLibre): acá solo importa que todo monte.
vi.mock('react-map-gl/maplibre', () => ({
  Map: ({ children }: { children?: ReactNode }) => <div data-testid="map">{children}</div>,
  Source: ({ children }: { children?: ReactNode }) => <div data-testid="edge-source">{children}</div>,
  Layer: () => <div data-testid="edge-layer" />,
  Marker: ({ children }: { children?: ReactNode }) => <div data-testid="marker">{children}</div>,
  Popup: ({ children }: { children?: ReactNode }) => <div data-testid="popup">{children}</div>,
}));

// FASE 2: la sección "Datos en vivo" consume hooks de datos REALES
// (axios/SSE) — se mockea para que este smoke no dispare red en jsdom.
vi.mock('../uilab/LiveDataSection', () => ({
  LiveDataSection: () => <div data-testid="live-data-section" />,
}));

describe('UiLabView', () => {
  it('monta todas las secciones de la galería', () => {
    // FASE 3: el ToastProvider vive en AppShell; el harness lo repone porque
    // la vista usa useToast y acá se monta standalone.
    render(
      <ToastProvider>
        <UiLabView />
      </ToastProvider>,
    );
    expect(screen.getByRole('heading', { name: 'UI Lab' })).toBeInTheDocument();
    for (const section of [
      'Tokens',
      'Button',
      'Chip · HChip · SegmentedControl',
      'StatusChip · Pill · CountBadge · DemoBadge',
      'ConfBar',
      'Panel',
      'Sparkline',
      'KpiCard',
      'Toasts',
      'Modal',
      'Drawer',
      'Mapa (MapCanvas + mockGeo)',
      'Datos en vivo (backend local)',
    ]) {
      expect(screen.getByRole('heading', { name: section })).toBeInTheDocument();
    }
    expect(screen.getByTestId('edge-layer')).toBeInTheDocument();
    expect(screen.getAllByTestId('marker').length).toBeGreaterThanOrEqual(3);
    expect(screen.getByTestId('live-data-section')).toBeInTheDocument();
  });
});
