// FASE 4 (Opción B2) — parser puro del multipart MJPEG del edge
// (GET /video/{id}?type=processed). Sin React, sin red: recibe chunks de bytes y
// emite los JPEG completos. Es el punto frágil del stream anotado, por eso vive
// aislado y se testea como unidad.
//
// Formato BYTE-EXACTO del server (edge_device .../routes/video.py, cada yield =
//   b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + <JPEG> + b'\r\n'):
//
//   --frame\r\nContent-Type: image/jpeg\r\n\r\n<JPEG1>\r\n--frame\r\n…\r\n\r\n<JPEG2>\r\n--frame…
//
// → el PRIMER boundary está en el byte 0 SIN \r\n previo; los boundaries de CIERRE
// sí van precedidos por el \r\n final del body anterior. NO hay Content-Length, así
// que el corte se hace por boundary:
//
//   - APERTURA de parte: se busca `--frame` PELADO (detecta el de byte 0 y los
//     subsiguientes), seguido del fin de headers `\r\n\r\n`.
//   - CIERRE del body: se busca el siguiente `\r\n--frame` CON CRLF previo — el \r\n
//     no entra al JPEG (es el trailing del body) y exigir el CRLF descarta el falso
//     positivo de un `--frame` que aparezca PELADO dentro del payload.
//
// RE-SINCRONIZACIÓN: cada slice candidato se valida (SOI 0xFFD8 al inicio, EOI
// 0xFFD9 al final). Si valida, se emite; el parser SOLO emite JPEGs válidos. Si no
// valida (frame truncado por el server bajo carga, chunk raro, o un `\r\n--frame`
// dentro del payload que cortó de más — indistinguible de un cierre real sin
// Content-Length), se DESCARTA y se sigue buscando el próximo cierre válido: ese
// frame puntual se pierde, pero el parser no propaga el desfase ni pinta basura.

const BOUNDARY_OPEN = bytes('--frame'); // apertura: pelado
const BOUNDARY_CLOSE = bytes('\r\n--frame'); // cierre: con CRLF previo
const HEADER_END = bytes('\r\n\r\n');

// Cap del buffer: un JPEG 1280×720 pesa decenas-cientos de KB. 8 MB es holgado para
// un frame y corta el crecimiento sin fin si el stream se desincroniza.
const BUFFER_CAP_BYTES = 8 * 1024 * 1024;

function bytes(s: string): Uint8Array {
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 0xff;
  return out;
}

/** Índice de `needle` dentro de `haystack` desde `from`, o -1. Búsqueda byte a byte. */
function indexOfSeq(haystack: Uint8Array, needle: Uint8Array, from: number): number {
  const last = haystack.length - needle.length;
  for (let i = Math.max(0, from); i <= last; i++) {
    let match = true;
    for (let j = 0; j < needle.length; j++) {
      if (haystack[i + j] !== needle[j]) {
        match = false;
        break;
      }
    }
    if (match) return i;
  }
  return -1;
}

/** Un slice es un JPEG válido sii empieza con SOI (FF D8) y termina con EOI (FF D9). */
function isValidJpeg(slice: Uint8Array): boolean {
  const n = slice.length;
  return (
    n >= 4 &&
    slice[0] === 0xff &&
    slice[1] === 0xd8 &&
    slice[n - 2] === 0xff &&
    slice[n - 1] === 0xd9
  );
}

export class MjpegParser {
  private buf: Uint8Array = new Uint8Array(0);

  /**
   * Acumula `chunk` y devuelve los JPEG completos y VÁLIDOS que quedaron disponibles.
   * Cero, uno o varios por llamada. Un chunk puede traer un frame partido, el final
   * de uno + el inicio del siguiente, o varios frames completos.
   */
  push(chunk: Uint8Array): Uint8Array[] {
    this.append(chunk);

    const frames: Uint8Array[] = [];
    // Cursor de búsqueda de apertura. Avanza para re-sincronizar sin re-escanear
    // desde 0 tras un slice descartado.
    let searchFrom = 0;

    for (;;) {
      // 1) Apertura de parte: `--frame` pelado.
      const open = indexOfSeq(this.buf, BOUNDARY_OPEN, searchFrom);
      if (open === -1) {
        // Sin apertura visible: conservar solo una cola que pueda contener un
        // boundary partido entre chunks (el resto es ruido previo a la 1ª parte).
        this.dropBefore(Math.max(0, this.buf.length - (BOUNDARY_OPEN.length - 1)));
        break;
      }

      // 2) Fin de headers `\r\n\r\n` tras la apertura.
      const headerEnd = indexOfSeq(this.buf, HEADER_END, open + BOUNDARY_OPEN.length);
      if (headerEnd === -1) {
        // Headers incompletos: descartar lo previo a la apertura y esperar más bytes.
        this.dropBefore(open);
        break;
      }
      const jpegStart = headerEnd + HEADER_END.length;

      // 3) Cierre del body: siguiente `\r\n--frame` con CRLF.
      const close = indexOfSeq(this.buf, BOUNDARY_CLOSE, jpegStart);
      if (close === -1) {
        // Body incompleto: descartar lo previo a la apertura y esperar más bytes.
        this.dropBefore(open);
        break;
      }

      // 4) Slice candidato [jpegStart, close): validar o descartar+re-sync.
      const slice = this.buf.slice(jpegStart, close);
      if (isValidJpeg(slice)) {
        frames.push(slice);
      }
      // El `--frame` del cierre es la apertura de la próxima parte: avanzar el cursor
      // hasta ahí. Tanto si el slice fue válido como si se descartó, re-sincronizamos
      // sobre ese boundary.
      searchFrom = close + (BOUNDARY_CLOSE.length - BOUNDARY_OPEN.length);
    }

    this.enforceCap();
    return frames;
  }

  /** Estado interno (para tests/diagnóstico): bytes pendientes en el buffer. */
  get pendingBytes(): number {
    return this.buf.length;
  }

  private append(chunk: Uint8Array): void {
    if (chunk.length === 0) return;
    const merged = new Uint8Array(this.buf.length + chunk.length);
    merged.set(this.buf, 0);
    merged.set(chunk, this.buf.length);
    this.buf = merged;
  }

  /** Descarta los primeros `n` bytes del buffer (avance del cursor de parseo). */
  private dropBefore(n: number): void {
    if (n <= 0) return;
    this.buf = n >= this.buf.length ? new Uint8Array(0) : this.buf.slice(n);
  }

  /**
   * Si el buffer creció por encima del cap sin poder extraer un frame, el stream
   * está desincronizado: se descarta todo salvo una cola corta (por si hay un
   * boundary partido al final) y se re-sincroniza con el próximo `--frame`.
   */
  private enforceCap(): void {
    if (this.buf.length <= BUFFER_CAP_BYTES) return;
    const keep = BOUNDARY_CLOSE.length - 1;
    this.buf = this.buf.slice(this.buf.length - keep);
  }
}
