/**
 * Tipos locales del track EXPERIMENTAL TomTom (Fase A — capa visual raster).
 *
 * Aislados a la carpeta: NO se exportan al resto del front ni se mezclan con los
 * tipos de `congestion/` (Waze) ni de predicción. Si Fase B agrega flowSegmentData
 * + KPIs, sus tipos viven acá también, no en `types/` global.
 */

/**
 * Estilos de tile raster del Traffic Flow de TomTom (API v4).
 * Verificado contra la doc oficial de Raster Flow Tiles.
 * Fase A usa `relative0-dark` (elección de diseño sobre UI oscura `bg-slate-900`).
 */
export type TomTomFlowStyle =
  | 'absolute'
  | 'relative'
  | 'relative0'
  | 'relative0-dark'
  | 'relative-delay'
  | 'reduced-sensitivity';
