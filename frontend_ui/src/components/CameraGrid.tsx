import { useEffect, useRef, useState } from 'react';
import { HlsPlayer, type PlayerStatus } from './HlsPlayer';

/**
 * Grilla de previews HLS de las cámaras (B1). Reproducción LAZY por viewport: cada celda
 * monta su HlsPlayer y reproduce SOLO cuando está en viewport; al salir, desmonta el player
 * (su cleanup hace hls.destroy()), así nunca hay más streams vivos que los visibles.
 * Cámara offline (error fatal / 404 / timeout) o sin stream_url → label "CÁMARA OFFLINE".
 */
export interface GridCamera {
  id: string;
  name: string;
  stream_url: string | null;
}

interface CameraGridProps {
  cameras: GridCamera[];
  onSelectCamera?: (id: string, name: string, streamUrl: string | null) => void;
  /** B2: reporta hacia arriba la salud (playing/offline/loading) de cada cámara, para
   *  que el marcador del mapa la comparta. */
  onStatusChange?: (id: string, status: PlayerStatus) => void;
  /** B2: id resaltado por la cross-selección (hover en mapa/lista/grilla). */
  selectedId?: string | null;
  /** B2: hover sobre una celda → propaga el id (o null al salir). */
  onHover?: (id: string | null) => void;
}

export function CameraGrid({
  cameras,
  onSelectCamera,
  onStatusChange,
  selectedId,
  onHover,
}: CameraGridProps) {
  if (cameras.length === 0) {
    return (
      <div className="text-slate-400 text-sm p-6 text-center">
        No hay cámaras para mostrar.
      </div>
    );
  }
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {cameras.map((cam) => (
        <CameraCell
          key={cam.id}
          camera={cam}
          onSelect={onSelectCamera}
          onStatusChange={onStatusChange}
          selected={cam.id === selectedId}
          onHover={onHover}
        />
      ))}
    </div>
  );
}

function CameraCell({
  camera,
  onSelect,
  onStatusChange,
  selected,
  onHover,
}: {
  camera: GridCamera;
  onSelect?: (id: string, name: string, streamUrl: string | null) => void;
  onStatusChange?: (id: string, status: PlayerStatus) => void;
  selected?: boolean;
  onHover?: (id: string | null) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [status, setStatus] = useState<PlayerStatus>('loading');

  // Lazy por viewport. setState va dentro del callback del observer (async) → ok.
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

  // Cámara sin stream_url: offline INMEDIATO, sin montar player. Se reporta hacia arriba
  // (el player nunca correrá su onStatusChange para este caso).
  useEffect(() => {
    if (!camera.stream_url) onStatusChange?.(camera.id, 'offline');
  }, [camera.stream_url, camera.id, onStatusChange]);

  // Reporta cada cambio de estado del player hacia arriba además de guardarlo local.
  const handleStatus = (s: PlayerStatus) => {
    setStatus(s);
    onStatusChange?.(camera.id, s);
  };

  // Offline es STICKY: una vez marcado, la celda no remonta el player aunque siga visible.
  const offline = status === 'offline' || !camera.stream_url;

  const badge = offline
    ? { text: 'OFFLINE', cls: 'bg-rose-500/20 text-rose-300' }
    : status === 'playing'
      ? { text: 'EN VIVO', cls: 'bg-emerald-500/20 text-emerald-300' }
      : { text: 'CARGANDO', cls: 'bg-slate-500/20 text-slate-300' };

  return (
    <div
      ref={ref}
      data-testid="camera-cell"
      onClick={() => onSelect?.(camera.id, camera.name, camera.stream_url)}
      onMouseEnter={() => onHover?.(camera.id)}
      onMouseLeave={() => onHover?.(null)}
      className={`bg-slate-800 rounded-xl border overflow-hidden cursor-pointer transition-colors ${
        selected ? 'border-indigo-400 ring-2 ring-indigo-500/60' : 'border-slate-700 hover:border-indigo-500'
      }`}
    >
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-700">
        <span className="font-medium text-white text-sm truncate">{camera.name}</span>
        <span className={`text-[10px] px-2 py-0.5 rounded-full ${badge.cls}`}>{badge.text}</span>
      </div>
      <div className="aspect-video bg-black flex items-center justify-center">
        {offline ? (
          <div className="text-rose-400 text-sm font-bold tracking-wide">CÁMARA OFFLINE</div>
        ) : visible ? (
          <HlsPlayer src={camera.stream_url as string} onStatusChange={handleStatus} />
        ) : (
          <div className="text-slate-500 text-xs">…</div>
        )}
      </div>
    </div>
  );
}
