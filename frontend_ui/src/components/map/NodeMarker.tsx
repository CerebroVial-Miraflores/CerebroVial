import { divIcon } from 'leaflet';
import { Marker } from 'react-leaflet';

import { buildNodeIconHtml, NODE_ICON_SIZE, type NodeStatus } from './nodeIcon';

// FASE 1 rediseño UI — marcador de nodo semáforo sobre el MapCanvas.
// className: '' neutraliza el fondo blanco de .leaflet-div-icon de leaflet.css.

interface NodeMarkerProps {
  position: [number, number];
  status: NodeStatus;
  critical?: boolean;
  onClick?: () => void;
}

export function NodeMarker({ position, status, critical = false, onClick }: NodeMarkerProps) {
  const half = NODE_ICON_SIZE / 2;
  const icon = divIcon({
    html: buildNodeIconHtml(status, { critical }),
    className: '',
    iconSize: [NODE_ICON_SIZE, NODE_ICON_SIZE],
    iconAnchor: [half, half],
  });
  return (
    <Marker
      position={position}
      icon={icon}
      eventHandlers={onClick ? { click: onClick } : undefined}
    />
  );
}
