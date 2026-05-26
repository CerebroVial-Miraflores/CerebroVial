import React from 'react';
import { Loader2, AlertCircle, CheckCircle2, RotateCw } from 'lucide-react';
import { Card } from '../../ui/Card';
import type { ControlRecommendation } from '../../../services/controlService';
import { MODE_LABEL, PHASE_SUBTITLES } from './controlTypes';
import { TimingBar } from './TimingBar';
import { ModeSelector } from './ModeSelector';
import { TrafficLightCycle } from './TrafficLightCycle';
import { Pedagogical422Card } from './Pedagogical422Card';

export type RecommendationStatus = 'idle' | 'loading' | 'success' | 'error';

export type RecommendationErrorKind = 'webster_infeasible' | 'invalid_state' | 'generic';

export interface RecommendationError {
    kind: RecommendationErrorKind;
    message: string;
}

export interface ControlMetrics {
    flowTotal: number;
    Y: number;
    isPeak: boolean;
}

interface RecommendationPanelProps {
    status: RecommendationStatus;
    data: ControlRecommendation | null;
    error: RecommendationError | null;
    metrics: ControlMetrics;
    onRetry: () => void;
    onReset: () => void;
}

export const RecommendationPanel = ({ status, data, error, metrics, onRetry, onReset }: RecommendationPanelProps) => {
    if (status === 'idle') {
        return (
            <Card className="h-full flex items-center justify-center text-slate-500 text-sm">
                Configurá la intersección y pulsá <span className="mx-1 font-semibold text-indigo-400">Recomendar</span> para obtener el plan de tiempos.
            </Card>
        );
    }

    if (status === 'loading') {
        return (
            <Card className="h-full flex items-center justify-center gap-3 text-slate-300">
                <Loader2 size={20} className="animate-spin text-indigo-400" />
                <span>Calculando recomendación…</span>
            </Card>
        );
    }

    if (status === 'error' && error) {
        if (error.kind === 'webster_infeasible') {
            return (
                <Pedagogical422Card
                    status={status}
                    error={error}
                    metrics={metrics}
                    onReset={onReset}
                />
            );
        }

        if (error.kind === 'invalid_state') {
            return (
                <Card className="h-full border-amber-500/60 bg-amber-950/30">
                    <div className="flex items-start gap-3">
                        <AlertCircle className="text-amber-400 shrink-0 mt-0.5" size={22} />
                        <div className="flex-1">
                            <h3 className="text-amber-300 font-semibold mb-2">Estado inválido</h3>
                            <p className="text-sm text-slate-300 mb-3">{error.message}</p>
                            <p className="text-xs text-slate-400">Revisá los valores de <span className="text-slate-200">flow</span>, <span className="text-slate-200">saturation_flow</span> y <span className="text-slate-200">timestamp</span>.</p>
                        </div>
                    </div>
                </Card>
            );
        }

        return (
            <Card className="h-full border-red-500/60 bg-red-950/20">
                <div className="flex items-start gap-3">
                    <AlertCircle className="text-red-400 shrink-0 mt-0.5" size={22} />
                    <div className="flex-1">
                        <h3 className="text-red-300 font-semibold mb-2">Servicio no disponible</h3>
                        <p className="text-sm text-slate-300 mb-3">{error.message}</p>
                        <button
                            onClick={onRetry}
                            className="inline-flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-white text-sm px-3 py-1.5 rounded-lg transition-colors"
                        >
                            <RotateCw size={14} /> Reintentar
                        </button>
                    </div>
                </div>
            </Card>
        );
    }

    if (status === 'success' && data) {
        return (
            <Card className="h-full">
                <div className="flex items-start gap-3 mb-4">
                    <CheckCircle2 className="text-emerald-400 shrink-0 mt-0.5" size={22} />
                    <div className="flex-1">
                        <h3 className="text-white font-semibold text-lg mb-1">Recomendación generada</h3>
                        <p className="text-xs text-slate-400">Intersección <span className="text-slate-200 font-mono">{data.intersection_id}</span></p>
                    </div>
                </div>

                <ModeSelector status={status} data={data} error={error} />

                <div className="grid grid-cols-2 gap-3 mb-3">
                    <div className="bg-slate-900/60 rounded-lg p-3">
                        <p className="text-xs text-slate-500 uppercase tracking-wide">Modo</p>
                        <p className="text-indigo-300 font-semibold mt-1">{MODE_LABEL[data.mode]}</p>
                    </div>
                    <div className="bg-slate-900/60 rounded-lg p-3">
                        <p className="text-xs text-slate-500 uppercase tracking-wide">Ciclo</p>
                        <p className="text-emerald-300 font-semibold mt-1 font-mono">{data.cycle_seconds.toFixed(1)} s</p>
                    </div>
                </div>

                <div className="grid grid-cols-3 bg-slate-900/40 rounded-lg p-3 mb-3 divide-x divide-slate-800">
                    <div className="px-3 first:pl-0">
                        <p className="text-[10px] text-slate-500 uppercase tracking-wide">flow_total</p>
                        <p className="text-slate-100 font-mono font-semibold mt-0.5">{metrics.flowTotal.toFixed(0)} <span className="text-slate-500 text-xs font-normal">veh/h</span></p>
                        <p className="text-[10px] text-slate-600 mt-0.5">{metrics.isPeak ? '> 1500' : '< 1500'}</p>
                    </div>
                    <div className="px-3">
                        <p className="text-[10px] text-slate-500 uppercase tracking-wide">Y</p>
                        <p className="text-slate-100 font-mono font-semibold mt-0.5">{metrics.Y.toFixed(3)}</p>
                        <p className="text-[10px] text-slate-600 mt-0.5">Σ flow/sat</p>
                    </div>
                    <div className="px-3 flex flex-col items-start">
                        <p className="text-[10px] text-slate-500 uppercase tracking-wide">estado</p>
                        <span className={`mt-0.5 px-2 py-0.5 rounded text-xs font-bold border ${metrics.isPeak ? 'bg-amber-500/15 text-amber-300 border-amber-500/40' : 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40'}`}>
                            {metrics.isPeak ? 'PEAK' : 'OFF-PEAK'}
                        </span>
                    </div>
                </div>

                <TrafficLightCycle status={status} data={data} error={error} />

                <div className="mb-4">
                    <p className="text-xs text-slate-500 uppercase tracking-wide mb-2">Tiempos por fase</p>
                    {data.phase_timings.map(t => (
                        <TimingBar
                            key={t.phase_id}
                            timing={t}
                            cycleSeconds={data.cycle_seconds}
                            isNext={t.phase_id === data.next_phase}
                            subtitle={PHASE_SUBTITLES[t.phase_id] ?? ''}
                        />
                    ))}
                    <div className="flex items-center gap-3 text-[10px] text-slate-500 mt-2">
                        <span className="inline-flex items-center gap-1">
                            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500 inline-block" /> verde
                        </span>
                        <span className="inline-flex items-center gap-1">
                            <span className="w-2.5 h-2.5 rounded-sm bg-amber-400 inline-block" /> ámbar
                        </span>
                        <span className="inline-flex items-center gap-1">
                            <span className="w-2.5 h-2.5 rounded-sm bg-red-500 inline-block" /> all-red
                        </span>
                    </div>
                </div>

                <div className="mb-4">
                    <p className="text-xs text-slate-500 uppercase tracking-wide mb-2">Estado activo</p>
                    <p className="text-sm text-slate-200 leading-relaxed">
                        {data.mode === 'webster' ? (
                            <>
                                <span className="font-semibold text-indigo-300">Modo activo: Webster (off-peak).</span>{' '}
                                El motor detectó demanda baja (Σ flujos = <span className="font-mono">{metrics.flowTotal.toFixed(0)}</span> veh/h, debajo del umbral de 1500). Aplicó la fórmula de Webster (1958) para obtener un ciclo óptimo de <span className="font-mono">{data.cycle_seconds.toFixed(1)}</span>s, distribuyendo el verde proporcionalmente a la demanda crítica de cada fase. La capa MTC ajustó verdes que quedaban por debajo del mínimo legal de 7s.
                            </>
                        ) : (
                            <>
                                <span className="font-semibold text-indigo-300">Modo activo: Max Pressure (peak).</span>{' '}
                                El motor detectó demanda alta (Σ flujos = <span className="font-mono">{metrics.flowTotal.toFixed(0)}</span> veh/h, sobre el umbral de 1500). Eligió la fase <span className="font-mono text-indigo-300">{data.next_phase ?? '—'}</span> como próxima a entrar, según la presión de cada cola: la fase con más vehículos esperando entra primero. El ciclo base es <span className="font-mono">{data.cycle_seconds.toFixed(1)}</span>s.
                            </>
                        )}
                        {data.adjustments.length > 0 && (
                            <> Se aplicaron <span className="font-mono">{data.adjustments.length}</span> ajustes MTC (ver detalle abajo).</>
                        )}
                    </p>
                </div>

                <div className="mb-4">
                    <p className="text-[10px] text-slate-600 mb-1">─── Log técnico ──────────────────────────────────</p>
                    <p className="text-xs text-slate-400 font-mono leading-snug">{data.reasoning}</p>
                    <p className="text-[10px] text-slate-600 mt-1">──────────────────────────────────────────────────</p>
                </div>

                {data.adjustments.length > 0 && (
                    <div>
                        <p className="text-xs text-slate-500 uppercase tracking-wide mb-1">Ajustes MTC aplicados</p>
                        <ul className="text-sm text-slate-300 list-disc list-inside space-y-0.5">
                            {data.adjustments.map((adj, i) => (
                                <li key={i}>{adj}</li>
                            ))}
                        </ul>
                    </div>
                )}
            </Card>
        );
    }

    return null;
};
