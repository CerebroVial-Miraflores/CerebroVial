import type { FeatureCollection, LineString } from 'geojson';

// FASE 1 rediseño UI — fixtures GeoJSON inventadas de Miraflores para validar
// edgeStyle/MapCanvas en /ui-lab (Demo · datos simulados). Coordenadas en orden
// GeoJSON [lng, lat]. Los jam_level cubren las 4 clases visuales + el neutro
// (null = tramo sin dato).

export interface MockEdgeProps {
  edge_id: string;
  name: string;
  jam_level: number | null;
}

export const MIRAFLORES_CENTER: [number, number] = [-12.1211, -77.0297];

function edge(
  edgeId: string,
  name: string,
  jamLevel: number | null,
  coordinates: [number, number][],
): FeatureCollection<LineString, MockEdgeProps>['features'][number] {
  return {
    type: 'Feature',
    properties: { edge_id: edgeId, name, jam_level: jamLevel },
    geometry: { type: 'LineString', coordinates },
  };
}

export const MOCK_EDGES: FeatureCollection<LineString, MockEdgeProps> = {
  type: 'FeatureCollection',
  features: [
    edge('mock-larco', 'Av. Larco', 5, [
      [-77.0299, -12.1211],
      [-77.0285, -12.1267],
      [-77.0273, -12.1318],
    ]),
    edge('mock-benavides', 'Av. Benavides', 4, [
      [-77.0392, -12.1296],
      [-77.0312, -12.1284],
      [-77.0231, -12.1273],
    ]),
    edge('mock-pardo', 'Av. Pardo', 3, [
      [-77.0381, -12.1191],
      [-77.0314, -12.1198],
      [-77.0258, -12.1204],
    ]),
    edge('mock-arequipa', 'Av. Arequipa', 1, [
      [-77.0287, -12.1043],
      [-77.0292, -12.1118],
      [-77.0297, -12.1189],
    ]),
    edge('mock-malecon', 'Malecón de la Reserva', null, [
      [-77.0405, -12.1222],
      [-77.0367, -12.1287],
      [-77.0312, -12.1338],
    ]),
  ],
};
