import { useState, type FormEvent } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Cpu } from 'lucide-react';

import { useSession } from './SessionContext';
import { InvalidCredentialsError } from './authService';

interface LocationState {
  from?: { pathname: string };
  reason?: 'session-expired';
}

export function LoginView() {
  const { login } = useSession();
  const navigate = useNavigate();
  const location = useLocation();
  const state = (location.state ?? null) as LocationState | null;

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const canSubmit = email.length > 0 && password.length > 0 && !submitting;

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      await login({ email, password });
      const target = state?.from?.pathname ?? '/';
      navigate(target, { replace: true });
    } catch (err) {
      if (err instanceof InvalidCredentialsError) {
        setError('Credenciales incorrectas');
      } else {
        setError('No pudimos iniciar sesión. Intentá de nuevo en unos segundos.');
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans flex items-center justify-center p-4">
      <div className="w-full max-w-sm bg-slate-900 border border-slate-800 rounded-2xl shadow-xl p-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-indigo-600 rounded-lg flex items-center justify-center shadow-[0_0_15px_rgba(79,70,229,0.5)]">
            <Cpu className="text-white w-6 h-6" />
          </div>
          <span className="font-bold text-xl text-white tracking-tight">CerebroVial</span>
        </div>

        {state?.reason === 'session-expired' && (
          <div
            role="status"
            className="mb-4 px-3 py-2 rounded-md bg-amber-500/10 border border-amber-500/40 text-amber-300 text-sm"
          >
            Tu sesión expiró. Por favor, ingresá de nuevo.
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="login-email" className="block text-sm text-slate-300 mb-1">
              Email
            </label>
            <input
              id="login-email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => {
                setEmail(e.target.value);
                if (error) setError(null);
              }}
              className="w-full px-3 py-2 rounded-md bg-slate-800 border border-slate-700 text-slate-100 focus:outline-none focus:border-indigo-500"
            />
          </div>

          <div>
            <label htmlFor="login-password" className="block text-sm text-slate-300 mb-1">
              Contraseña
            </label>
            <input
              id="login-password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => {
                setPassword(e.target.value);
                if (error) setError(null);
              }}
              className="w-full px-3 py-2 rounded-md bg-slate-800 border border-slate-700 text-slate-100 focus:outline-none focus:border-indigo-500"
            />
          </div>

          {error && (
            <div
              role="alert"
              className="px-3 py-2 rounded-md bg-red-500/10 border border-red-500/40 text-red-300 text-sm"
            >
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={!canSubmit}
            className="w-full py-2 rounded-md bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white font-medium transition-colors"
          >
            {submitting ? 'Ingresando…' : 'Ingresar'}
          </button>
        </form>
      </div>
    </div>
  );
}
