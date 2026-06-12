// FASE 3 (B1) — stack mínimo de overlays (Drawer/Modal), singleton de módulo.
//
// Resuelve la deuda anotada en Fase 1: si conviven Drawer y Modal abiertos,
// Esc debe cerrar SOLO el de arriba (en Fase 1 cerraba ambos). Cada overlay
// pushea un handle al abrirse y lo libera al cerrar/desmontar; su handler de
// Esc consulta isTop() antes de cerrar. El scrim no necesita guard: el del
// overlay superior tapa al inferior (z-modal 100 > z-drawer 90).
//
// Sin React a propósito: el orden del stack es estado global de la página
// (los overlays portalan a document.body), no de un árbol de componentes.

const stack: symbol[] = [];

export interface OverlayHandle {
  /** true si este overlay es el de más arriba (el que debe responder a Esc). */
  isTop(): boolean;
  /** Saca el overlay del stack esté donde esté. Idempotente. */
  release(): void;
}

export function pushOverlay(): OverlayHandle {
  const id = Symbol('overlay');
  stack.push(id);
  return {
    isTop: () => stack.length > 0 && stack[stack.length - 1] === id,
    release: () => {
      const index = stack.indexOf(id);
      if (index !== -1) stack.splice(index, 1);
    },
  };
}

/** Para asserts en tests. */
export function overlayCount(): number {
  return stack.length;
}

/** Limpia el stack entre casos de test. */
export function resetOverlayStackForTests(): void {
  stack.length = 0;
}
