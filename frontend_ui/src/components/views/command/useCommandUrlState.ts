// FASE 3 rediseño UI — máquina de estados URL del Centro de Comando.
//
// La URL es la única fuente de verdad de: ?modo= (ahora|historico|prediccion),
// ?dia= (YYYY-MM-DD, histórico), ?t= (índice de timestep, histórico),
// ?nodo= (drawer de intersección), ?panel=alertas (one-shot scroll al panel).
//
// Reglas anti-loop (contrato del diseño):
// · Validación POR LECTURA, nunca por escritura: ?modo=basura se LEE como
//   'ahora'; jamás se "corrige" la URL en un effect (eso sí loopea). La URL
//   sucia es estable e inocua.
// · El CLAMP de t NO vive acá: el rango depende de la serie cargada (async).
//   El consumidor computa clampIndex(tRaw, len) en render — sin write-back.
// · Writers solo desde handlers, con la forma funcional setSearchParams(prev)
//   para no pisar params concurrentes (mover el slider con el drawer abierto
//   conserva ?nodo=).
// · El horizonte de predicción NO va a la URL (decisión D4): estado local.
import { useCallback, useState } from 'react';
import { useSearchParams } from 'react-router-dom';

import { todayIsoLocal, type CommandMode } from './derive';

const MODES: readonly string[] = ['ahora', 'historico', 'prediccion'];
const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;

export interface CommandUrlState {
  mode: CommandMode;
  /** Día válido del histórico (default: hoy local). */
  dia: string;
  /** Índice crudo de timestep (≥0); el consumidor lo clampea contra la serie. */
  tRaw: number;
  /** Nodo del drawer tal cual viene (la validación es del drawer). */
  nodo: string | null;
  panel: 'alertas' | null;
  setMode: (m: CommandMode) => void;
  setDia: (d: string) => void;
  setT: (i: number) => void;
  openNode: (id: string) => void;
  closeNode: () => void;
  clearPanel: () => void;
}

export function useCommandUrlState(): CommandUrlState {
  const [searchParams, setSearchParams] = useSearchParams();
  // Hoy local fijado al montar (patrón v1: useState initializer, render puro).
  const [today] = useState(() => todayIsoLocal());

  const rawMode = searchParams.get('modo') ?? '';
  const mode: CommandMode = MODES.includes(rawMode) ? (rawMode as CommandMode) : 'ahora';

  const rawDia = searchParams.get('dia');
  const dia = rawDia !== null && DAY_RE.test(rawDia) ? rawDia : today;

  const tParsed = Number.parseInt(searchParams.get('t') ?? '', 10);
  const tRaw = Number.isNaN(tParsed) || tParsed < 0 ? 0 : tParsed;

  const nodo = searchParams.get('nodo');
  const panel = searchParams.get('panel') === 'alertas' ? ('alertas' as const) : null;

  const setMode = useCallback(
    (m: CommandMode) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        if (m === 'ahora') next.delete('modo');
        else next.set('modo', m);
        // URL canónica: fuera de histórico, dia/t no significan nada.
        if (m !== 'historico') {
          next.delete('dia');
          next.delete('t');
        }
        return next;
      });
    },
    [setSearchParams],
  );

  const setDia = useCallback(
    (d: string) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set('dia', d);
        // Día nuevo → el índice anterior no aplica (la serie cambia de largo).
        next.delete('t');
        return next;
      });
    },
    [setSearchParams],
  );

  // replace: el slider no debe spamear el history (back = salir del modo, no
  // deshacer cada paso).
  const setT = useCallback(
    (i: number) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set('t', String(i));
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const openNode = useCallback(
    (id: string) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set('nodo', id);
        return next;
      });
    },
    [setSearchParams],
  );

  const closeNode = useCallback(() => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.delete('nodo');
      return next;
    });
  }, [setSearchParams]);

  // One-shot: el consumidor scrollea al panel y limpia el param (replace) —
  // el param desaparece y el effect no re-corre.
  const clearPanel = useCallback(() => {
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.delete('panel');
        return next;
      },
      { replace: true },
    );
  }, [setSearchParams]);

  return {
    mode,
    dia,
    tRaw,
    nodo,
    panel,
    setMode,
    setDia,
    setT,
    openNode,
    closeNode,
    clearPanel,
  };
}
