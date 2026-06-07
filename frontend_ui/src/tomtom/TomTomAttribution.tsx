/**
 * Atribución TomTom — visible y NO removible (ToS 17.3, track EXPERIMENTAL Fase A).
 *
 * Para una capa raster sin logo embebido, TomTom exige mostrar la atribución de
 * forma visible. Acá es texto estático "© TomTom" en la esquina del mapa, análogo
 * a cómo se muestra la atribución de OpenStreetMap. NO es cosmético ni opcional:
 * es requisito legal. No agregar lógica para ocultarlo/condicionarlo.
 *
 * Fase B (diferida): la Copyrights API de TomTom da el texto exacto por área vía
 * llamada HTTP — fuera de alcance en esta fase front-pura "cero fetch".
 */

export const TomTomAttribution = () => {
  const year = new Date().getFullYear();
  return (
    <div className="absolute bottom-2 right-2 z-[500] pointer-events-auto select-none rounded-md bg-slate-900/80 backdrop-blur px-2 py-1 text-[11px] leading-none text-slate-300 border border-slate-700 shadow">
      Tráfico{' '}
      <a
        href="https://www.tomtom.com"
        target="_blank"
        rel="noopener noreferrer"
        className="font-semibold text-slate-100 hover:text-white underline-offset-2 hover:underline"
      >
        © {year} TomTom
      </a>
    </div>
  );
};
