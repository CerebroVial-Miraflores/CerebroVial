// Config del edge device (visión). Single-source de la base URL del edge, antes
// duplicada en CameraLivePanel, AnnotatedCameraStream, useVisionStream y
// useVisionAggregates. El edge va CRUDO (sin httpClient/JWT — su auth es deuda
// backend, ver "Rediseño UI" en CLAUDE.md): EventSource, MJPEG por fetch y las
// acciones de inferencia (POST/DELETE /cameras/{id}) pegan directo acá.
export const EDGE_API_URL: string =
  import.meta.env?.VITE_EDGE_API_URL || 'http://localhost:8000';
