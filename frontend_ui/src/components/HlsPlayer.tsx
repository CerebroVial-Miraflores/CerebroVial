import { useEffect, useMemo, useRef, useState } from 'react';
import Hls, { Events, type ErrorData } from 'hls.js';

/**
 * Reproductor HLS de un .m3u8 (cámaras de Claro). Base validada en el spike B0
 * (CORS OK con withCredentials=false). Expone su estado vía onStatusChange para que
 * la grilla decida mostrar el player o el label "CÁMARA OFFLINE".
 */
export type PlayerStatus = 'loading' | 'playing' | 'offline';

interface HlsPlayerProps {
  src: string;
  onStatusChange?: (status: PlayerStatus) => void;
  /** ms sin primer frame antes de marcar offline (timeout de carga). Default 9000. */
  offlineTimeoutMs?: number;
}

type Engine = 'native' | 'hlsjs' | 'unsupported';

export function HlsPlayer({ src, onStatusChange, offlineTimeoutMs = 9000 }: HlsPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [status, setStatus] = useState<PlayerStatus>('loading');

  // Capacidad del navegador, decidida en RENDER (no en effect → sin setState síncrono).
  // Safari/iOS reproducen HLS nativo; el resto via hls.js sobre MSE.
  const engine = useMemo<Engine>(() => {
    const probe = document.createElement('video');
    if (probe.canPlayType('application/vnd.apple.mpegurl')) return 'native';
    if (Hls.isSupported()) return 'hlsjs';
    return 'unsupported';
  }, []);

  // 'unsupported' es offline derivado (sin setState en el cuerpo del effect).
  const effectiveStatus: PlayerStatus = engine === 'unsupported' ? 'offline' : status;

  // Notificar al padre (llama una prop, no setState propio → no viola la regla de hooks).
  useEffect(() => {
    onStatusChange?.(effectiveStatus);
  }, [effectiveStatus, onStatusChange]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || engine === 'unsupported') return;

    // Timeout de carga: si no hay primer frame a tiempo, offline. Solo se setea desde
    // callbacks async (timer/eventos) → respeta react-hooks/set-state-in-effect.
    let timer: ReturnType<typeof setTimeout> | null = setTimeout(() => {
      setStatus((s) => (s === 'playing' ? s : 'offline'));
    }, offlineTimeoutMs);
    const clearTimer = () => {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
    };

    const onPlaying = () => {
      clearTimer();
      setStatus('playing');
    };
    // CAMINO NATIVO / media error: el <video> emite 'error' (video.error.code).
    const onNativeError = () => {
      clearTimer();
      setStatus('offline');
    };
    video.addEventListener('playing', onPlaying);
    video.addEventListener('error', onNativeError);

    // 1) Safari/iOS: HLS nativo, sin hls.js.
    if (engine === 'native') {
      video.src = src;
      video.play().catch(() => {/* autoplay puede requerir gesto del usuario */});
      return () => {
        clearTimer();
        video.removeEventListener('playing', onPlaying);
        video.removeEventListener('error', onNativeError);
        video.removeAttribute('src');
        video.load();
      };
    }

    // 2) Chrome/Firefox/Edge/Brave: hls.js sobre MSE.
    const hls = new Hls({
      // DECISIÓN B0 — requests SIN credenciales (default, explícito). Claro puede mandar
      // Allow-Origin:* + Allow-Credentials:true; una request credentialed haría que el
      // navegador rechace el '*'. No-credentialed → el '*' es aceptado.
      xhrSetup: (xhr) => {
        xhr.withCredentials = false;
      },
    });

    // CAMINO HLS.JS: solo los errores FATALES marcan offline (los no-fatales son ruido
    // normal de carga/buffer y no deben tumbar la celda).
    hls.on(Events.ERROR, (_evt, data: ErrorData) => {
      if (!data.fatal) return;
      const code = data.response?.code;
      console.error(
        '[HlsPlayer]',
        `fatal type=${data.type} details=${data.details}` +
          (code !== undefined ? ` response.code=${code}` : ''),
      );
      clearTimer();
      setStatus('offline');
    });

    hls.on(Events.MANIFEST_PARSED, () => {
      video.play().catch(() => {/* autoplay puede requerir gesto del usuario */});
    });

    hls.loadSource(src);
    hls.attachMedia(video);

    return () => {
      clearTimer();
      video.removeEventListener('playing', onPlaying);
      video.removeEventListener('error', onNativeError);
      hls.destroy();
    };
  }, [src, engine, offlineTimeoutMs]);

  return (
    <video
      ref={videoRef}
      controls
      muted
      playsInline
      style={{ width: '100%', height: '100%', objectFit: 'cover', background: '#000' }}
    />
  );
}
