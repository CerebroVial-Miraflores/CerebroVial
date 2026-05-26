import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AxiosError } from 'axios';
import { authService, InvalidCredentialsError } from '../authService';
import { httpClient } from '../../services/httpClient';

vi.mock('../../services/httpClient', () => ({
  httpClient: { post: vi.fn() },
}));

const postMock = httpClient.post as unknown as ReturnType<typeof vi.fn>;

function axiosErrorWithStatus(status: number): AxiosError {
  const err = new AxiosError(
    `Request failed with status code ${status}`,
    'ERR_BAD_REQUEST',
  );
  err.response = {
    status,
    statusText: '',
    data: { detail: 'x' },
    headers: {},
    config: {} as never,
  };
  return err;
}

describe('authService.login', () => {
  beforeEach(() => {
    postMock.mockReset();
  });

  it('posts urlencoded body with username=email and password to /auth/login', async () => {
    postMock.mockResolvedValue({
      data: { access_token: 't', token_type: 'bearer', expires_in: 28800 },
    });

    await authService.login({ email: 'alice@cv.pe', password: 's3cret' });

    expect(postMock).toHaveBeenCalledTimes(1);
    const [path, body, config] = postMock.mock.calls[0];
    expect(path).toBe('/auth/login');
    expect(body).toBeInstanceOf(URLSearchParams);
    expect((body as URLSearchParams).get('username')).toBe('alice@cv.pe');
    expect((body as URLSearchParams).get('password')).toBe('s3cret');
    expect(config?.headers?.['Content-Type']).toBe(
      'application/x-www-form-urlencoded',
    );
  });

  it('returns the TokenResponse on 200', async () => {
    postMock.mockResolvedValue({
      data: { access_token: 'jwt.value.here', token_type: 'bearer', expires_in: 28800 },
    });

    const result = await authService.login({ email: 'a@b.c', password: 'x' });

    expect(result).toEqual({
      access_token: 'jwt.value.here',
      token_type: 'bearer',
      expires_in: 28800,
    });
  });

  it('throws InvalidCredentialsError on 401', async () => {
    postMock.mockRejectedValue(axiosErrorWithStatus(401));

    await expect(
      authService.login({ email: 'a@b.c', password: 'wrong' }),
    ).rejects.toBeInstanceOf(InvalidCredentialsError);
  });

  it('rethrows non-401 errors as-is', async () => {
    const serverError = axiosErrorWithStatus(500);
    postMock.mockRejectedValue(serverError);

    await expect(
      authService.login({ email: 'a@b.c', password: 'x' }),
    ).rejects.toBe(serverError);
  });
});
