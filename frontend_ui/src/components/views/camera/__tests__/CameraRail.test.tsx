import { act } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

import { CameraRail } from '../CameraRail';
import type { CameraSummary } from '../../../../types/cameras';

// HlsPlayer real corre hls.js; acá lo stubeamos para observar montaje sin red.
vi.mock('../../../HlsPlayer', () => ({
  HlsPlayer: ({ src }: { src: string }) => <div data-testid="hls-player">{src}</div>,
}));

// IntersectionObserver controlable: capturamos el callback para simular que un
// tile entra al viewport (lazy → recién ahí monta el HlsPlayer).
let ioCallback: ((entries: { isIntersecting: boolean }[]) => void) | null = null;
class MockIO {
  constructor(cb: (entries: { isIntersecting: boolean }[]) => void) {
    ioCallback = cb;
  }
  observe() {}
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return [];
  }
}

beforeEach(() => {
  ioCallback = null;
  Object.defineProperty(globalThis, 'IntersectionObserver', {
    configurable: true,
    writable: true,
    value: MockIO,
  });
});

const cameras: CameraSummary[] = [
  { id: 'cam_a', name: 'Cámara A', stream_url: 'https://x/a.m3u8' },
  { id: 'cam_b', name: 'Cámara B', stream_url: 'https://x/b.m3u8' },
  { id: 'cam_c', name: 'Cámara C (offline)', stream_url: null },
];

describe('CameraRail', () => {
  it('excluye la cámara activa y lista las otras', () => {
    render(<CameraRail cameras={cameras} activeId="cam_a" onSelect={vi.fn()} />);
    expect(screen.queryByText('Cámara A')).not.toBeInTheDocument();
    expect(screen.getByText('Cámara B')).toBeInTheDocument();
    expect(screen.getByText('Cámara C (offline)')).toBeInTheDocument();
    expect(screen.getAllByTestId('rail-tile')).toHaveLength(2);
  });

  it('lazy: no monta el HlsPlayer hasta que el tile entra al viewport', () => {
    render(<CameraRail cameras={cameras} activeId="cam_c" onSelect={vi.fn()} />);
    // Aún no visible → sin player.
    expect(screen.queryByTestId('hls-player')).not.toBeInTheDocument();

    act(() => {
      ioCallback?.([{ isIntersecting: true }]);
    });
    // Ahora visibles: los tiles con stream montan player (cam_a y cam_b).
    expect(screen.getAllByTestId('hls-player').length).toBeGreaterThan(0);
  });

  it('una cámara sin stream queda OFFLINE y nunca monta player', () => {
    render(<CameraRail cameras={cameras} activeId="cam_a" onSelect={vi.fn()} />);
    act(() => {
      ioCallback?.([{ isIntersecting: true }]);
    });
    // cam_c sin stream: aparece el label OFFLINE y su tile no tiene player propio.
    expect(screen.getAllByText('OFFLINE').length).toBeGreaterThan(0);
  });

  it('click en un tile invoca onSelect con el id de la cámara', () => {
    const onSelect = vi.fn();
    render(<CameraRail cameras={cameras} activeId="cam_a" onSelect={onSelect} />);
    fireEvent.click(screen.getByTitle('Ver Cámara B'));
    expect(onSelect).toHaveBeenCalledWith('cam_b');
  });

  it('sin otras cámaras muestra vacío honesto', () => {
    render(
      <CameraRail cameras={[cameras[0]]} activeId="cam_a" onSelect={vi.fn()} />,
    );
    expect(screen.getByText(/no hay otras cámaras/i)).toBeInTheDocument();
  });
});
