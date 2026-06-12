// FASE 2 rediseño UI — wire de GET /api/intersections (core).
//
// Shape relevado del consumidor v1 (DashboardView), que mantiene su interface
// inline hasta morir en su fase. Este tipo es la versión canónica para la capa
// services/hooks del rediseño.

export interface IntersectionSummary {
  id: string;
  name: string;
  speed: number;
  flow: number;
  /** Nivel de congestión agregado (constructo Waze, ver D-009). */
  status: string;
  lat: number;
  lng: number;
  stream_url: string | null;
}
