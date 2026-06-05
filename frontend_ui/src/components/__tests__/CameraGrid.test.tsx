import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock de IntersectionObserver (jsdom no lo trae). Guarda los callbacks por instancia
// para que el test simule entrada/salida de viewport.
const io = vi.hoisted(() => ({
  cbs: [] as ((entries: { isIntersecting: boolean }[]) => void)[],
  observe: vi.fn(),
  disconnect: vi.fn(),
}));

class MockIntersectionObserver {
  constructor(cb: (entries: { isIntersecting: boolean }[]) => void) {
    io.cbs.push(cb);
  }
  observe = io.observe;
  disconnect = io.disconnect;
  unobserve = vi.fn();
  takeRecords = vi.fn();
}

// Mock de HlsPlayer: no corre hls; expone botones para simular onStatusChange.
vi.mock('../HlsPlayer', () => ({
  HlsPlayer: ({
    src,
    onStatusChange,
  }: {
    src: string;
    onStatusChange?: (s: 'loading' | 'playing' | 'offline') => void;
  }) => (
    <div data-testid="hls-player">
      <span data-testid="hls-src">{src}</span>
      <button data-testid="go-offline" onClick={() => onStatusChange?.('offline')} />
      <button data-testid="go-playing" onClick={() => onStatusChange?.('playing')} />
    </div>
  ),
}));

import { CameraGrid } from '../CameraGrid';

const setVisible = (visible: boolean) =>
  act(() => {
    io.cbs.forEach((cb) => cb([{ isIntersecting: visible }]));
  });

describe('CameraGrid', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    io.cbs = [];
    // @ts-expect-error: inyectamos el mock en el global
    globalThis.IntersectionObserver = MockIntersectionObserver;
  });

  it('monta el HlsPlayer al entrar en viewport y lo desmonta al salir', () => {
    render(<CameraGrid cameras={[{ id: 'c1', name: 'Larco Benavides', stream_url: 'http://x/c1.m3u8' }]} />);

    // Fuera de viewport: placeholder, sin player.
    expect(screen.queryByTestId('hls-player')).not.toBeInTheDocument();

    setVisible(true);
    expect(screen.getByTestId('hls-player')).toBeInTheDocument();
    expect(screen.getByTestId('hls-src')).toHaveTextContent('http://x/c1.m3u8');

    setVisible(false);
    expect(screen.queryByTestId('hls-player')).not.toBeInTheDocument();
  });

  it('muestra "CÁMARA OFFLINE" cuando el player reporta offline (sticky)', () => {
    render(<CameraGrid cameras={[{ id: 'c1', name: 'Larco Benavides', stream_url: 'http://x/c1.m3u8' }]} />);
    setVisible(true);

    act(() => {
      fireEvent.click(screen.getByTestId('go-offline'));
    });

    expect(screen.getByText('CÁMARA OFFLINE')).toBeInTheDocument();
    expect(screen.queryByTestId('hls-player')).not.toBeInTheDocument();

    // Sticky: aunque siga visible, no remonta el player.
    setVisible(true);
    expect(screen.queryByTestId('hls-player')).not.toBeInTheDocument();
  });

  it('muestra "CÁMARA OFFLINE" cuando stream_url es null (sin montar player)', () => {
    render(<CameraGrid cameras={[{ id: 'c2', name: 'Sin Stream', stream_url: null }]} />);
    setVisible(true);

    expect(screen.getByText('CÁMARA OFFLINE')).toBeInTheDocument();
    expect(screen.queryByTestId('hls-player')).not.toBeInTheDocument();
  });

  it('click en una celda dispara onSelectCamera con id y nombre', () => {
    const onSelect = vi.fn();
    render(
      <CameraGrid
        cameras={[{ id: 'c1', name: 'Larco Benavides', stream_url: 'http://x/c1.m3u8' }]}
        onSelectCamera={onSelect}
      />,
    );

    fireEvent.click(screen.getByTestId('camera-cell'));
    expect(onSelect).toHaveBeenCalledWith('c1', 'Larco Benavides');
  });
});
