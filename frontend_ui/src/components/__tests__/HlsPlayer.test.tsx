import { render, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock de hls.js: el constructor devuelve una instancia controlable; capturamos los
// handlers registrados con .on(evt, cb) para dispararlos desde el test.
const hlsMock = vi.hoisted(() => {
  const handlers: Record<string, (e: unknown, d: unknown) => void> = {};
  const instance = {
    on: vi.fn((evt: string, cb: (e: unknown, d: unknown) => void) => {
      handlers[evt] = cb;
    }),
    loadSource: vi.fn(),
    attachMedia: vi.fn(),
    destroy: vi.fn(),
  };
  return { handlers, instance, isSupported: vi.fn(() => true) };
});

vi.mock('hls.js', () => {
  class Hls {
    static isSupported() {
      return hlsMock.isSupported();
    }
    constructor() {
      return hlsMock.instance as unknown as Hls;
    }
  }
  return {
    default: Hls,
    Events: { ERROR: 'hlsError', MANIFEST_PARSED: 'hlsManifestParsed' },
  };
});

import { HlsPlayer, type PlayerStatus } from '../HlsPlayer';

describe('HlsPlayer', () => {
  let canPlayTypeSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.clearAllMocks();
    Object.keys(hlsMock.handlers).forEach((k) => delete hlsMock.handlers[k]);
    hlsMock.isSupported.mockReturnValue(true);
    // jsdom no implementa play/load; los stubeamos para no romper.
    HTMLMediaElement.prototype.play = vi.fn(() => Promise.resolve());
    HTMLMediaElement.prototype.load = vi.fn();
    // Por defecto: navegador sin HLS nativo → camino hls.js.
    canPlayTypeSpy = vi.spyOn(HTMLMediaElement.prototype, 'canPlayType').mockReturnValue('');
  });

  afterEach(() => {
    canPlayTypeSpy.mockRestore();
    vi.useRealTimers();
  });

  it('camino hls.js: un error FATAL marca offline', () => {
    const statuses: PlayerStatus[] = [];
    render(<HlsPlayer src="http://x/p.m3u8" onStatusChange={(s) => statuses.push(s)} />);
    expect(statuses.at(-1)).toBe('loading');

    act(() => {
      hlsMock.handlers['hlsError'](
        { type: 'hlsError' },
        { fatal: true, type: 'networkError', details: 'manifestLoadError', response: { code: 404 } },
      );
    });

    expect(statuses.at(-1)).toBe('offline');
  });

  it('camino hls.js: un error NO-fatal NO marca offline', () => {
    const statuses: PlayerStatus[] = [];
    render(<HlsPlayer src="http://x/p.m3u8" onStatusChange={(s) => statuses.push(s)} />);

    act(() => {
      hlsMock.handlers['hlsError'](
        { type: 'hlsError' },
        { fatal: false, type: 'mediaError', details: 'bufferStalledError' },
      );
    });

    expect(statuses).not.toContain('offline');
    expect(statuses.at(-1)).toBe('loading');
  });

  it('camino nativo (Safari): el evento error del <video> marca offline', () => {
    canPlayTypeSpy.mockReturnValue('maybe'); // fuerza el camino nativo
    const statuses: PlayerStatus[] = [];
    const { container } = render(
      <HlsPlayer src="http://x/p.m3u8" onStatusChange={(s) => statuses.push(s)} />,
    );
    const video = container.querySelector('video')!;

    act(() => {
      fireEvent.error(video);
    });

    expect(statuses.at(-1)).toBe('offline');
  });

  it('timeout sin primer frame marca offline', () => {
    vi.useFakeTimers();
    const statuses: PlayerStatus[] = [];
    render(
      <HlsPlayer
        src="http://x/p.m3u8"
        offlineTimeoutMs={1000}
        onStatusChange={(s) => statuses.push(s)}
      />,
    );

    act(() => {
      vi.advanceTimersByTime(1000);
    });

    expect(statuses.at(-1)).toBe('offline');
  });
});
