// FASE 1 rediseño UI — lock de scroll del body compartido por overlays
// (Modal/Drawer). Con refcount: si hay dos overlays abiertos, el unlock de uno
// no libera el del otro; cada acquire devuelve un release idempotente.

let lockCount = 0;

export function acquireBodyLock(): () => void {
  lockCount += 1;
  document.body.classList.add('overflow-hidden');
  let released = false;
  return () => {
    if (released) return;
    released = true;
    lockCount -= 1;
    if (lockCount === 0) {
      document.body.classList.remove('overflow-hidden');
    }
  };
}
