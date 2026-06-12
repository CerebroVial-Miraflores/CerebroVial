import { useEffect, useRef, useState } from 'react';

import { HlsPlayer, type PlayerStatus } from '../../HlsPlayer';
import type { CameraSummary } from '../../../types/cameras';

// FASE 4 rediseño UI — carril horizontal de previews HLS del detalle de cámara
// (reemplaza tokenizado a CameraStrip v1). Mismo lazy-por-viewport: cada tile
// monta su HlsPlayer y reproduce SOLO al entrar en viewport; al salir, el
// cleanup destruye el player → nunca más streams vivos que los visibles.
// Muestra las OTRAS cámaras navegables; click → navega a /camara/<id> (la URL
// es la fuente de verdad de la cámara activa).

interface CameraRailProps {
  /** Lista completa; el carril excluye la cámara activa. */
  cameras: CameraSummary[];
  activeId: string;
  onSelect: (id: string) => void;
}

export function CameraRail({ cameras, activeId, onSelect }: CameraRailProps) {
  const others = cameras.filter((c) => c.id !== activeId);
  if (others.length === 0) {
    return (
      <p className="px-1 py-3 text-[11.5px] text-ink-3">
        No hay otras cámaras para mostrar.
      </p>
    );
  }
  return (
    <div
      data-testid="camera-rail"
      className="flex gap-3 overflow-x-auto pb-2 snap-x scroll-smooth"
    >
      {others.map((cam) => (
        <RailTile key={cam.id} camera={cam} onSelect={onSelect} />
      ))}
    </div>
  );
}

function RailTile({
  camera,
  onSelect,
}: {
  camera: CameraSummary;
  onSelect: (id: string) => void;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  const [visible, setVisible] = useState(false);
  const [status, setStatus] = useState<PlayerStatus>('loading');

  // Lazy por viewport: los tiles fuera del scroll horizontal no montan player.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) setVisible(entry.isIntersecting);
      },
      { threshold: 0.1 },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Offline STICKY: una vez marcado, no remonta el player.
  const offline = status === 'offline' || !camera.stream_url;

  const badge = offline
    ? { text: 'OFFLINE', cls: 'border-bad/40 bg-bad/15 text-bad' }
    : status === 'playing'
      ? { text: 'EN VIVO', cls: 'border-ok/40 bg-ok/13 text-ok' }
      : { text: 'CARGANDO', cls: 'border-line-2 bg-panel-2 text-ink-2' };

  return (
    <button
      ref={ref}
      type="button"
      data-testid="rail-tile"
      onClick={() => onSelect(camera.id)}
      title={`Ver ${camera.name}`}
      className="group w-44 shrink-0 snap-start overflow-hidden rounded-ctl border border-line bg-canvas-2 text-left transition-colors hover:border-brand/60"
    >
      <div className="flex items-center justify-between gap-2 border-b border-line px-2 py-1.5">
        <span className="truncate text-[11px] font-semibold text-ink">{camera.name}</span>
        <span
          className={`shrink-0 rounded-full border px-1.5 py-0.5 text-[8.5px] font-extrabold tracking-[0.08em] ${badge.cls}`}
        >
          {badge.text}
        </span>
      </div>
      <div className="grid aspect-video place-items-center bg-black">
        {offline ? (
          <span className="text-[10.5px] font-bold tracking-wide text-bad">OFFLINE</span>
        ) : visible ? (
          <HlsPlayer src={camera.stream_url as string} controls={false} onStatusChange={setStatus} />
        ) : (
          <span className="text-[11px] text-ink-3">…</span>
        )}
      </div>
    </button>
  );
}
