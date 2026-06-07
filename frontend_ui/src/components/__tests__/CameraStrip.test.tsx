import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock de IntersectionObserver (jsdom no lo trae). Guarda los callbacks por instancia
// para simular entrada/salida de viewport.
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
    </div>
  ),
}));

import { CameraStrip } from '../CameraStrip';

const setVisible = (visible: boolean) =>
  act(() => {
    io.cbs.forEach((cb) => cb([{ isIntersecting: visible }]));
  });

describe('CameraStrip', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    io.cbs = [];
    // @ts-expect-error: inyectamos el mock en el global
    globalThis.IntersectionObserver = MockIntersectionObserver;
  });

  it('muestra un mensaje cuando no hay otras cámaras', () => {
    render(<CameraStrip cameras={[]} onSelect={vi.fn()} />);
    expect(screen.getByText(/no hay otras cámaras/i)).toBeInTheDocument();
  });

  it('monta el HlsPlayer al entrar en viewport y lo desmonta al salir (lazy)', () => {
    render(
      <CameraStrip
        cameras={[{ id: 'c1', name: 'Larco Benavides', stream_url: 'http://x/c1.m3u8' }]}
        onSelect={vi.fn()}
      />,
    );

    // Fuera de viewport: placeholder, sin player.
    expect(screen.queryByTestId('hls-player')).not.toBeInTheDocument();

    setVisible(true);
    expect(screen.getByTestId('hls-player')).toBeInTheDocument();
    expect(screen.getByTestId('hls-src')).toHaveTextContent('http://x/c1.m3u8');

    setVisible(false);
    expect(screen.queryByTestId('hls-player')).not.toBeInTheDocument();
  });

  it('click en un tile dispara onSelect con id, nombre y stream_url', () => {
    const onSelect = vi.fn();
    render(
      <CameraStrip
        cameras={[{ id: 'c1', name: 'Larco Benavides', stream_url: 'http://x/c1.m3u8' }]}
        onSelect={onSelect}
      />,
    );

    fireEvent.click(screen.getByTestId('strip-tile'));
    expect(onSelect).toHaveBeenCalledWith('c1', 'Larco Benavides', 'http://x/c1.m3u8');
  });

  it('cámara sin stream_url se muestra OFFLINE sin montar player', () => {
    render(
      <CameraStrip
        cameras={[{ id: 'c2', name: 'Sin Stream', stream_url: null }]}
        onSelect={vi.fn()}
      />,
    );
    setVisible(true);

    // 'OFFLINE' aparece en el badge y en el cuerpo del tile.
    expect(screen.getAllByText('OFFLINE').length).toBeGreaterThan(0);
    expect(screen.queryByTestId('hls-player')).not.toBeInTheDocument();
  });

  it('renderiza un tile por cámara', () => {
    render(
      <CameraStrip
        cameras={[
          { id: 'c1', name: 'A', stream_url: 'http://x/a.m3u8' },
          { id: 'c2', name: 'B', stream_url: 'http://x/b.m3u8' },
          { id: 'c3', name: 'C', stream_url: 'http://x/c.m3u8' },
        ]}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getAllByTestId('strip-tile')).toHaveLength(3);
  });
});
