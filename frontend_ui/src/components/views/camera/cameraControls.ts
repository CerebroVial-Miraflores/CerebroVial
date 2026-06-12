// FASE 4 rediseño UI — parámetros reales del endpoint /video del edge (MJPEG).
// Preservados de v1: el edge sirve el frame procesado (bboxes) o crudo, y la
// calidad mapea a width/quality del re-encode. Viven en el contenedor del
// detalle para persistir al navegar entre cámaras (el panel se re-monta por key).

export type StreamType = 'raw' | 'processed';
export type Quality = 'low' | 'medium' | 'high';

export function qualityParams(quality: Quality): string {
  switch (quality) {
    case 'low':
      return 'width=480&quality=50';
    case 'medium':
      return 'width=640&quality=70';
    case 'high':
      return 'width=1280&quality=85';
    default:
      return 'width=640&quality=70';
  }
}
