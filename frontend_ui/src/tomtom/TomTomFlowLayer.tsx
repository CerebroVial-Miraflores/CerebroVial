/**
 * Capa raster de Traffic Flow de TomTom (track EXPERIMENTAL, Fase A).
 *
 * Un único <TileLayer> de react-leaflet apuntando al endpoint raster oficial de
 * TomTom (API v4). Los tiles los pide el NAVEGADOR directo a api.tomtom.com — el
 * backend NO interviene (ToS 11.4: los tiles son Results, no se cachean ni se
 * reenvían desde servidor a múltiples clientes). Sin estado, sin fetch manual:
 * Leaflet pide los tiles solo a partir de la `url` plantilla.
 *
 * URL (doc oficial Raster Flow Tiles):
 *   https://api.tomtom.com/traffic/map/{version}/tile/flow/{style}/{z}/{x}/{y}.{fmt}?key={key}
 * version=4, fmt=png. Leaflet sustituye {z}/{x}/{y}.
 *
 * Degradación con gracia: si `VITE_TOMTOM_KEY` no está definida, NO se monta el
 * layer (devuelve null) y se avisa por consola. NUNCA se hardcodea una key: la
 * `VITE_TOMTOM_KEY` es una display key protegida por domain-whitelist + QPS en el
 * dashboard de TomTom (va al bundle a propósito), no un secret de servidor.
 */
import { TileLayer } from 'react-leaflet';
import type { TomTomFlowStyle } from './types';

const TOMTOM_KEY: string | undefined = import.meta.env.VITE_TOMTOM_KEY;

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
  if (!TOMTOM_KEY) {
    // Degradación con gracia: la vista sigue funcionando (solo OSM, sin flujo).
    console.warn(
      '[tomtom] VITE_TOMTOM_KEY no definida — la capa de tráfico de TomTom no se monta. ' +
        'Definila en .env (display key con domain-whitelist) para ver el tráfico en vivo.',
    );
    return null;
  }

  const url =
    `https://api.tomtom.com/traffic/map/4/tile/flow/${style}/{z}/{x}/{y}.png` +
    `?key=${TOMTOM_KEY}`;

  return (
    <TileLayer
      // La atribución de Leaflet también muestra "© TomTom"; la atribución
      // visible y no removible vive además en <TomTomAttribution/> (ToS 17.3).
      attribution='Traffic &copy; <a href="https://www.tomtom.com">TomTom</a>'
      url={url}
      opacity={opacity}
    />
  );
};
