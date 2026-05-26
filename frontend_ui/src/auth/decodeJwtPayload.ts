import type { JwtPayload } from './types';

/**
 * Decode the payload segment of a JWT for UI purposes only.
 *
 * SECURITY: this function does NOT validate the JWT signature, issuer, or
 * expiration. It must NEVER be used for authorization. The only authority on
 * whether a token is valid is the backend, which verifies the signature with
 * the shared secret on every protected request. We decode the payload in the
 * client solely to read display claims (role, sub) so the UI can render the
 * right shell. Treat the result as untrusted input.
 */
export function decodeJwtPayload(token: string | null): JwtPayload | null {
  if (typeof token !== 'string' || token.length === 0) return null;

  const parts = token.split('.');
  if (parts.length !== 3) return null;

  const payloadSegment = parts[1];
  if (!payloadSegment) return null;

  const base64 = payloadSegment.replace(/-/g, '+').replace(/_/g, '/');
  const padded = base64 + '='.repeat((4 - (base64.length % 4)) % 4);

  let json: string;
  try {
    json = atob(padded);
  } catch {
    return null;
  }

  try {
    const parsed: unknown = JSON.parse(json);
    if (parsed === null || typeof parsed !== 'object') return null;
    return parsed as JwtPayload;
  } catch {
    return null;
  }
}
