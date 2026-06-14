import { describe, expect, it } from 'vitest';

import { MjpegParser } from '../mjpegStreamParser';

// FASE 4 (Opción B2) — test del parser multipart MJPEG. El punto frágil del stream
// anotado, por eso el fixture es BYTE-EXACTO al formato del server (arranca con
// `--frame` SIN \r\n previo) y los troceos son adversariales.

function bytes(s: string): Uint8Array {
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 0xff;
  return out;
}

function concat(...parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

/** JPEG sintético: SOI (FF D8) + payload arbitrario + EOI (FF D9). */
function jpeg(payload: Uint8Array | number[] = []): Uint8Array {
  const body = payload instanceof Uint8Array ? payload : Uint8Array.from(payload);
  return concat(Uint8Array.from([0xff, 0xd8]), body, Uint8Array.from([0xff, 0xd9]));
}

/** Una parte multipart byte-exacta al server. */
function part(j: Uint8Array): Uint8Array {
  return concat(bytes('--frame\r\nContent-Type: image/jpeg\r\n\r\n'), j, bytes('\r\n'));
}

/**
 * Stream byte-exacto: concatena las partes y agrega un `--frame` final para que el
 * ÚLTIMO JPEG tenga su boundary de cierre (en el server siempre viene el próximo
 * frame; acá lo simulamos). El resultado arranca con `--frame` en el byte 0.
 */
function stream(...jpegs: Uint8Array[]): Uint8Array {
  return concat(...jpegs.map(part), bytes('--frame'));
}

/** Empuja `data` al parser en chunks de tamaño `chunkSize` y junta lo emitido. */
function feedInChunks(data: Uint8Array, chunkSize: number): Uint8Array[] {
  const parser = new MjpegParser();
  const out: Uint8Array[] = [];
  for (let i = 0; i < data.length; i += chunkSize) {
    out.push(...parser.push(data.subarray(i, i + chunkSize)));
  }
  return out;
}

function toArr(u: Uint8Array): number[] {
  return Array.from(u);
}

describe('MjpegParser', () => {
  it('emite el PRIMER frame del stream (boundary en byte 0, sin CRLF previo)', () => {
    const j1 = jpeg([1, 2, 3]);
    const parser = new MjpegParser();
    const frames = parser.push(stream(j1));
    expect(frames.map(toArr)).toEqual([toArr(j1)]);
  });

  it('(a) un JPEG partido entre múltiples chunks se reensambla', () => {
    const j1 = jpeg([10, 20, 30, 40, 50]);
    const data = stream(j1);
    // chunk chico fuerza el JPEG y los boundaries a partirse varias veces.
    expect(feedInChunks(data, 3).map(toArr)).toEqual([toArr(j1)]);
  });

  it('(b) un chunk con el final de un JPEG + el inicio del siguiente', () => {
    const j1 = jpeg([1, 1, 1]);
    const j2 = jpeg([2, 2, 2, 2]);
    const data = stream(j1, j2);
    // Cortar justo dentro del primer JPEG: el chunk 2 trae fin de j1 + inicio de j2.
    const cut = part(j1).length - 1;
    const parser = new MjpegParser();
    const out = [
      ...parser.push(data.subarray(0, cut)),
      ...parser.push(data.subarray(cut)),
    ];
    expect(out.map(toArr)).toEqual([toArr(j1), toArr(j2)]);
  });

  it('(c) varios JPEG completos en un solo chunk', () => {
    const j1 = jpeg([1]);
    const j2 = jpeg([2, 2]);
    const j3 = jpeg([3, 3, 3]);
    const parser = new MjpegParser();
    const frames = parser.push(stream(j1, j2, j3));
    expect(frames.map(toArr)).toEqual([toArr(j1), toArr(j2), toArr(j3)]);
  });

  it('(d) boundary partido entre chunks', () => {
    const j1 = jpeg([7, 7]);
    const j2 = jpeg([8, 8]);
    const data = stream(j1, j2);
    // Cortar en medio del `\r\n--frame` de cierre de j1.
    const closeIdx = part(j1).length - 2; // posición del \r del cierre
    const cut = closeIdx + 4; // dentro de `\r\n--frame`
    const parser = new MjpegParser();
    const out = [
      ...parser.push(data.subarray(0, cut)),
      ...parser.push(data.subarray(cut)),
    ];
    expect(out.map(toArr)).toEqual([toArr(j1), toArr(j2)]);
  });

  it('(e) un JPEG cuyo payload contiene `--frame` PELADO (sin CRLF) se emite completo', () => {
    // El cierre exige `\r\n--frame`; un `--frame` pelado en el payload NO se confunde
    // con un boundary → el frame no se corta.
    const j = jpeg(concat(bytes('xx--frame yy'), Uint8Array.from([0, 1, 2])));
    const parser = new MjpegParser();
    const frames = parser.push(stream(j));
    expect(frames.map(toArr)).toEqual([toArr(j)]);
  });

  it('(f) un slice con `\\r\\n--frame` en el payload se PIERDE (re-sync), los válidos se emiten', () => {
    const j1 = jpeg([1, 1]);
    // j_x lleva un `\r\n--frame` embebido → indistinguible de un cierre real sin
    // Content-Length. Tras el falso corte queda data sin `\r\n\r\n` espurio hasta el
    // header de la próxima parte, así el parser re-sincroniza limpio sobre j2.
    const jx = jpeg(concat(bytes('\r\n--frame'), bytes('basura-sin-crlf-crlf')));
    const j2 = jpeg([2, 2]);
    const parser = new MjpegParser();
    const frames = parser.push(stream(j1, jx, j2));
    // jx se pierde; j1 y j2 sí salen.
    expect(frames.map(toArr)).toEqual([toArr(j1), toArr(j2)]);
  });

  it('(f-bis) un slice truncado sin EOI entre dos válidos se descarta', () => {
    const j1 = jpeg([9]);
    const j2 = jpeg([5, 5, 5]);
    // Parte corrupta: SOI sin EOI (cuerpo cortado). Se arma a mano para que el cierre
    // `\r\n--frame` caiga sobre un slice que no termina en FF D9.
    const corrupt = Uint8Array.from([0xff, 0xd8, 0x42, 0x42]); // sin FF D9
    const data = concat(part(j1), part(corrupt), part(j2), bytes('--frame'));
    const parser = new MjpegParser();
    const frames = parser.push(data);
    expect(frames.map(toArr)).toEqual([toArr(j1), toArr(j2)]);
  });

  it('reensambla igual con cualquier tamaño de chunk (1..N)', () => {
    const j1 = jpeg([1, 2, 3, 4]);
    const j2 = jpeg([5, 6]);
    const j3 = jpeg([7, 8, 9]);
    const data = stream(j1, j2, j3);
    const expected = [toArr(j1), toArr(j2), toArr(j3)];
    for (const size of [1, 2, 5, 7, 13, 64, data.length]) {
      expect(feedInChunks(data, size).map(toArr)).toEqual(expected);
    }
  });

  it('no emite nada mientras el frame está incompleto (sin boundary de cierre)', () => {
    const j1 = jpeg([1, 2, 3]);
    const incomplete = part(j1); // sin el `--frame` de cierre
    const parser = new MjpegParser();
    expect(parser.push(incomplete)).toEqual([]);
    // Al llegar el cierre, recién ahí emite.
    expect(parser.push(bytes('--frame')).map(toArr)).toEqual([toArr(j1)]);
  });
});
