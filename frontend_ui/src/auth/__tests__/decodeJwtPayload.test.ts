// CT-01.12: decodeJwtPayload is UI-only, tolerates malformed input without throwing.
import { describe, it, expect } from 'vitest';
import { decodeJwtPayload } from '../decodeJwtPayload';

function base64url(obj: unknown): string {
  return btoa(JSON.stringify(obj))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

function makeToken(payload: unknown, signature = 'sig'): string {
  const header = base64url({ alg: 'HS256', typ: 'JWT' });
  return `${header}.${base64url(payload)}.${signature}`;
}

describe('decodeJwtPayload', () => {
  it('decodes a valid JWT and returns the payload with role and sub', () => {
    const token = makeToken({
      sub: '11111111-2222-3333-4444-555555555555',
      role: 'operator',
      exp: 9_999_999_999,
    });
    const payload = decodeJwtPayload(token);
    expect(payload).not.toBeNull();
    expect(payload?.sub).toBe('11111111-2222-3333-4444-555555555555');
    expect(payload?.role).toBe('operator');
    expect(payload?.exp).toBe(9_999_999_999);
  });

  it('returns null for null input', () => {
    expect(decodeJwtPayload(null)).toBeNull();
  });

  it('returns null for an empty string', () => {
    expect(decodeJwtPayload('')).toBeNull();
  });

  it('returns null for a token that is not three dot-separated parts', () => {
    expect(decodeJwtPayload('only.two')).toBeNull();
    expect(decodeJwtPayload('a.b.c.d')).toBeNull();
    expect(decodeJwtPayload('plain')).toBeNull();
  });

  it('returns null for a token whose payload segment is not valid base64', () => {
    expect(decodeJwtPayload('aaa.???.sig')).toBeNull();
  });

  it('returns null for a token whose payload is base64 but not valid JSON', () => {
    const garbage = btoa('not-json-at-all').replace(/=+$/, '');
    expect(decodeJwtPayload(`aaa.${garbage}.sig`)).toBeNull();
  });

  it('handles base64url padding correctly when payload length is not multiple of 4', () => {
    const payload = { sub: 'abc', role: 'admin' as const };
    const encoded = base64url(payload);
    expect(encoded.length % 4).not.toBe(0);
    const token = `header.${encoded}.sig`;
    const decoded = decodeJwtPayload(token);
    expect(decoded?.sub).toBe('abc');
    expect(decoded?.role).toBe('admin');
  });

  it('handles base64url chars - and _ that are not standard base64', () => {
    const payload = { data: '???>>>' };
    const encoded = base64url(payload);
    if (!encoded.includes('-') && !encoded.includes('_')) {
      const forced = `${base64url({ sub: 'xy' })}-_`;
      expect(decodeJwtPayload(`h.${forced}.s`)).toBeNull();
      return;
    }
    const token = `h.${encoded}.s`;
    const decoded = decodeJwtPayload(token);
    expect(decoded).not.toBeNull();
  });

  it('does not throw on a token whose payload lacks role or exp; returns what is present', () => {
    const token = makeToken({ sub: 'just-a-sub' });
    const decoded = decodeJwtPayload(token);
    expect(decoded?.sub).toBe('just-a-sub');
    expect(decoded?.role).toBeUndefined();
    expect(decoded?.exp).toBeUndefined();
  });
});
