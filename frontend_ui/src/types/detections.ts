// FASE 4 Mitad A — tipo del payload del tap de detecciones del edge
// (GET /detections/{id}/latest, D-019). Lo consume useDetections para el overlay
// de cajas sobre el video HLS.

export interface DetectionBox {
  /** id del vehículo (string limpio; tracking o detector). */
  id: string;
  /** car | bus | truck | motorcycle. */
  type: string;
  confidence: number;
  /** [x1, y1, x2, y2] NORMALIZADO a [0, 1] sobre el frame de inferencia. */
  bbox: [number, number, number, number];
}

export interface DetectionFrame {
  /** dims del frame de inferencia (espacio en que se normalizó el bbox). El
   *  overlay las usa como viewBox para que el letterbox coincida con el video. */
  width: number;
  height: number;
}

export interface DetectionsPayload {
  camera_id: string;
  /** null si la cámara aún no produjo detecciones. */
  frame: DetectionFrame | null;
  /** epoch s del instante de la detección (reloj del edge). null si vacío. */
  frame_timestamp: number | null;
  /** epoch s de cuándo el edge serializó la respuesta (su "ahora"). */
  server_timestamp: number;
  detection_ran: boolean;
  detections: DetectionBox[];
}
