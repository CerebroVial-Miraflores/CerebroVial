import { Marker } from 'react-map-gl/maplibre';

import { buildNodeIconHtml, NODE_ICON_SIZE, type NodeStatus } from './nodeIcon';

// FASE 4 migración MapLibre — marcador de nodo semáforo sobre el <Map>.
// Renderiza el MISMO SVG de nodeIcon.ts (buildNodeIconHtml) por
// dangerouslySetInnerHTML: var() y la animación animate-halo-ping funcionan
// igual en un marker DOM (vive en el árbol de la app). `position` se mantiene
// en [lat, lng] (contrato heredado de Leaflet); se convierte a longitude/latitude
// adentro. anchor="center" replica el iconAnchor centrado del divIcon.

interface NodeMarkerProps {
  position: [number, number];
  status: NodeStatus;
  critical?: boolean;
  onClick?: () => void;
}

export function NodeMarker({ position, status, critical = false, onClick }: NodeMarkerProps) {
  return (
    <Marker
      longitude={position[1]}
      latitude={position[0]}
      anchor="center"
      onClick={
        onClick
          ? (e) => {
              e.originalEvent.stopPropagation();
              onClick();
            }
          : undefined
      }
    >
      <div
        style={{ width: NODE_ICON_SIZE, height: NODE_ICON_SIZE, cursor: onClick ? 'pointer' : 'default' }}
        dangerouslySetInnerHTML={{ __html: buildNodeIconHtml(status, { critical }) }}
      />
    </Marker>
  );
}
