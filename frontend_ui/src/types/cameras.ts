// FASE 2 rediseño UI — wire de GET /api/cameras (core).
//
// Lista liviana de cámaras (solo CameraDB, sin el agregado Waze de
// /api/intersections). Mismo shape que el GridCamera del carril v1, pero los
// services no importan tipos desde components/ — esta es la versión canónica.

export interface CameraSummary {
  id: string;
  name: string;
  stream_url: string | null;
}
