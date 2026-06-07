import { useEffect, useRef, useState } from 'react';
import { HlsPlayer, type PlayerStatus } from './HlsPlayer';
import type { GridCamera } from './CameraGrid';

/**
 * Carril horizontal de previews HLS (filmstrip). Mismo lazy-por-viewport que `CameraGrid`
 * (B1): cada tile monta su HlsPlayer y reproduce SOLO cuando entra en viewport; al salir,
 * desmonta el player (cleanup → hls.destroy()), así nunca hay más streams vivos que los
 * visibles. Pensado para el detalle de cámara: muestra las OTRAS cámaras navegables en
 * horizontal; al click, el detalle se reemplaza por la cámara elegida.
 *
 * No se reutiliza `CameraGrid` directo porque su layout es grilla (celdas full-width,
 * aspect-video) y sus tests están acoplados a ese DOM; el tile de acá es de ancho fijo.
 */
interface CameraStripProps {
  cameras: GridCamera[];
  onSelect: (id: string, name: string, streamUrl: string | null) => void;
}

export function CameraStrip({ cameras, onSelect }: CameraStripProps) {
  if (cameras.length === 0) {
    return (
      <div className="text-slate-500 text-xs px-4 py-3">
        No hay otras cámaras para mostrar.
      </div>
    );
  }
  return (
    <div
      data-testid="camera-strip"
      className="flex gap-3 overflow-x-auto pb-2 px-1 snap-x scroll-smooth"
    >
      {cameras.map((cam) => (
        <StripTile key={cam.id} camera={cam} onSelect={onSelect} />
      ))}
    </div>
  );
}

function StripTile({
  camera,
  onSelect,
}: {
  camera: GridCamera;
  onSelect: (id: string, name: string, streamUrl: string | null) => void;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  const [visible, setVisible] = useState(false);
  const [status, setStatus] = useState<PlayerStatus>('loading');

  // Lazy por viewport. El observer corre contra el viewport; los tiles fuera del scroll
  // horizontal quedan fuera de él y no montan player. setState dentro del callback (async) → ok.
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

  // Offline es STICKY (igual que la grilla): una vez marcado, no remonta el player.
  const offline = status === 'offline' || !camera.stream_url;

  const badge = offline
    ? { text: 'OFFLINE', cls: 'bg-rose-500/20 text-rose-300' }
    : status === 'playing'
      ? { text: 'EN VIVO', cls: 'bg-emerald-500/20 text-emerald-300' }
      : { text: 'CARGANDO', cls: 'bg-slate-500/20 text-slate-300' };

  return (
    <button
      ref={ref}
      type="button"
      data-testid="strip-tile"
      onClick={() => onSelect(camera.id, camera.name, camera.stream_url)}
      title={`Ver ${camera.name}`}
      className="group shrink-0 w-48 snap-start bg-slate-800 rounded-lg border border-slate-700 hover:border-indigo-500 overflow-hidden cursor-pointer transition-colors text-left"
    >
      <div className="flex items-center justify-between px-2 py-1.5 border-b border-slate-700">
        <span className="font-medium text-white text-xs truncate">{camera.name}</span>
        <span className={`text-[9px] px-1.5 py-0.5 rounded-full shrink-0 ${badge.cls}`}>{badge.text}</span>
      </div>
      <div className="aspect-video bg-black flex items-center justify-center">
        {offline ? (
          <div className="text-rose-400 text-[11px] font-bold tracking-wide">OFFLINE</div>
        ) : visible ? (
          <HlsPlayer src={camera.stream_url as string} onStatusChange={setStatus} />
        ) : (
          <div className="text-slate-600 text-xs">…</div>
        )}
      </div>
    </button>
  );
}
