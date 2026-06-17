/**
 * Capa raster de Traffic Flow de TomTom (track EXPERIMENTAL, Fase A).
 *
 * FASE 4 migración MapLibre: <TileLayer> de Leaflet → <Source type="raster"> +
 * <Layer type="raster"> de react-map-gl. Los tiles los pide el NAVEGADOR directo
 * a api.tomtom.com — el backend NO interviene (ToS 11.4: los tiles son Results,
 * no se cachean ni se reenvían desde servidor). Sin estado, sin fetch manual.
 *
 * URL (doc oficial Raster Flow Tiles):
 *   https://api.tomtom.com/traffic/map/{version}/tile/flow/{style}/{z}/{x}/{y}.{fmt}?key={key}
 * version=4, fmt=png. MapLibre sustituye {z}/{x}/{y}.
 *
 * Degradación con gracia: si `VITE_TOMTOM_KEY` no está definida, NO se monta el
 * layer (devuelve null) y se avisa por consola. La key se lee EN EL RENDER (no a
 * nivel de módulo) para que sea determinista en test (vi.stubEnv). NUNCA se
 * hardcodea: la `VITE_TOMTOM_KEY` es una display key protegida por domain-whitelist
 * + QPS en el dashboard de TomTom (va al bundle a propósito), no un secret.
 */
import { Source, Layer } from 'react-map-gl/maplibre';
import type { TomTomFlowStyle } from './types';

interface TomTomFlowLayerProps {
  /** Estilo del tile. Fase A: 'relative0-dark' (UI oscura). */
  style?: TomTomFlowStyle;
  /** Opacidad de la capa sobre el basemap OSM. */
  opacity?: number;
}

export const TomTomFlowLayer = ({
  style = 'relative0-dark',
  opacity = 1,
}: TomTomFlowLayerProps) => {
  const key: string | undefined = import.meta.env.VITE_TOMTOM_KEY;
  if (!key) {
    // Degradación con gracia: la vista sigue funcionando (solo OSM, sin flujo).
    console.warn(
      '[tomtom] VITE_TOMTOM_KEY no definida — la capa de tráfico de TomTom no se monta. ' +
        'Definila en .env (display key con domain-whitelist) para ver el tráfico en vivo.',
    );
    return null;
  }

  const url =
    `https://api.tomtom.com/traffic/map/4/tile/flow/${style}/{z}/{x}/{y}.png` +
    `?key=${key}`;

  return (
    <Source
      id="tomtom-flow"
      type="raster"
      tiles={[url]}
      tileSize={256}
      // La atribución visible y no removible vive además en <TomTomAttribution/> (ToS 17.3).
      attribution='Traffic &copy; <a href="https://www.tomtom.com">TomTom</a>'
    >
      <Layer id="tomtom-flow" type="raster" paint={{ 'raster-opacity': opacity }} />
    </Source>
  );
};
