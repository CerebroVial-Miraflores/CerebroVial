import axios from 'axios';
import { httpClient } from '../services/httpClient';
import type { LoginCredentials, TokenResponse } from './types';

export class InvalidCredentialsError extends Error {
  constructor() {
    super('Credenciales incorrectas');
    this.name = 'InvalidCredentialsError';
  }
}

export const authService = {
  async login(creds: LoginCredentials): Promise<TokenResponse> {
    const body = new URLSearchParams();
    body.append('username', creds.email);
    body.append('password', creds.password);

    try {
      const response = await httpClient.post<TokenResponse>('/auth/login', body, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      });
      return response.data;
    } catch (err) {
      if (axios.isAxiosError(err) && err.response?.status === 401) {
        throw new InvalidCredentialsError();
      }
      throw err;
    }
  },
};
