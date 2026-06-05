// B2 — apariencia del marcador del mapa: dos dimensiones que NO se pisan.
//   · COLOR  = congestión real por intersección (Waze, agregado MAX en el backend).
//   · PULSO/TACHADO = salud de la cámara (online / offline / desconocido).
// El pulso significa "transmite" (B2 D1): online y DESCONOCIDO pulsan; solo el offline
// confirmado pierde el pulso y se tacha. `health === undefined` = estado desconocido (la
// celda de la grilla nunca montó su HlsPlayer: fuera de viewport o grilla desmontada) →
// pulsa normal, NUNCA se marca offline por no haberse reproducido todavía.
import type { PlayerStatus } from '../components/HlsPlayer';

export interface MarkerVisual {
  /** Clase Tailwind del color del punto (congestión Waze: rojo/ámbar/verde). */
  color: string;
  /** El marcador pulsa (animate-ping) = cámara transmite o estado aún desconocido. */
  pulse: boolean;
  /** Tachado diagonal = cámara offline CONFIRMADA. */
  struck: boolean;
}

/** Color del punto según el status de congestión (fluid|moderate|critical). */
export function congestionColorClass(status: string): string {
  if (status === 'critical') return 'bg-red-500';
  if (status === 'moderate') return 'bg-amber-500';
  return 'bg-emerald-500';
}

/**
 * Combina congestión (color) y salud (pulso/tachado) para el marcador. Ver nota de
 * cabecera para la semántica de "desconocido ≠ offline" (B2 D1).
 */
export function markerVisual(
  status: string,
  health: PlayerStatus | undefined,
): MarkerVisual {
  const offline = health === 'offline';
  return { color: congestionColorClass(status), pulse: !offline, struck: offline };
}
