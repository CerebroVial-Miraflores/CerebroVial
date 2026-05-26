export type Role = 'operator' | 'manager' | 'admin';

export interface JwtPayload {
  sub?: string;
  role?: Role;
  exp?: number;
  [claim: string]: unknown;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: 'bearer';
  expires_in: number;
}

export interface SessionState {
  token: string | null;
  role: Role | null;
  userId: string | null;
  isAuthenticated: boolean;
}
