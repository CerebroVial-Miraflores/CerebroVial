// DHU-020 — Playground interactivo del motor adaptativo, preservado como
// herramienta de Administrador.
//
// Este es el ControlView original (pre-HU-05): un panel interactivo donde el
// usuario edita la demanda de cada fase, dispara ``POST /control/recommend``
// y ve la recomendación con su razonamiento. Útil para diagnóstico, demo
// académica y exploración del comportamiento del motor.
//
// HU-05 introduce la vista pasiva del Operador (ActiveStrategyView). El nuevo
// ControlView.tsx decide cuál renderizar según el rol activo:
// - operator → ActiveStrategyView (pasiva, read-only, sin inputs)
// - admin    → ControlPlayground (este archivo, interactivo)
//
// El contenido se preserva BYTE-A-BYTE respecto al ControlView previo a la
// división — DHU-020 fue explícito en no degradar el playground.
import { useCallback, useMemo, useState } from 'react';
import { Play } from 'lucide-react';
import { Card } from '../../ui/Card';
import { PhaseEditor } from './PhaseEditor';
import { Slider } from './Slider';
import { PresetButtons } from './PresetButtons';
import { KNOWN_NODE_IDS, PRESETS, type PresetConfig } from './controlTypes';
import {
    RecommendationPanel,
    type RecommendationError,
    type RecommendationStatus,
} from './RecommendationPanel';
import {
    controlService,
    parseControlError,
    type ControlRecommendation,
    type IntersectionState,
    type PhaseFlow,
} from '../../../services/controlService';

const initialPhases: PhaseFlow[] = [
    { phase_id: 'NS', flow: 600, saturation_flow: 1800, queue: 5, has_pedestrian: true },
    { phase_id: 'EW', flow: 400, saturation_flow: 1800, queue: 3, has_pedestrian: false },
];

interface UseRecommendControlState {
    status: RecommendationStatus;
    data: ControlRecommendation | null;
    error: RecommendationError | null;
    mutate: (state: IntersectionState) => Promise<void>;
    reset: () => void;
}

function useRecommendControl(): UseRecommendControlState {
    const [status, setStatus] = useState<RecommendationStatus>('idle');
    const [data, setData] = useState<ControlRecommendation | null>(null);
    const [error, setError] = useState<RecommendationError | null>(null);

    const mutate = useCallback(async (state: IntersectionState) => {
        setStatus('loading');
        setError(null);
        try {
            const res = await controlService.recommend(state);
            setData(res.data);
            setStatus('success');
        } catch (err) {
            const detail = parseControlError(err);
            if (detail?.code === 'webster_infeasible') {
                setError({ kind: 'webster_infeasible', message: detail.message });
            } else if (detail?.code === 'invalid_state') {
                setError({ kind: 'invalid_state', message: detail.message });
            } else {
                setError({
                    kind: 'generic',
                    message: 'No se pudo obtener la recomendación. Verificá que el backend esté disponible.',
                });
            }
            setStatus('error');
        }
    }, []);

    const reset = useCallback(() => {
        setStatus('idle');
        setData(null);
        setError(null);
    }, []);

    return { status, data, error, mutate, reset };
}

function presetMatches(p: PresetConfig, lostTime: number, phases: PhaseFlow[]): boolean {
    if (p.lostTime !== lostTime) return false;
    if (p.phases.length !== phases.length) return false;
    return p.phases.every((pp, i) => {
        const ph = phases[i];
        return (
            pp.phase_id === ph.phase_id &&
            pp.flow === ph.flow &&
            pp.saturation_flow === ph.saturation_flow &&
            (pp.queue ?? 0) === (ph.queue ?? 0) &&
            (pp.has_pedestrian ?? false) === (ph.has_pedestrian ?? false)
        );
    });
}

const formIsValid = (intersectionId: string, phases: PhaseFlow[]): boolean => {
    if (intersectionId.trim().length === 0) return false;
    if (phases.length === 0) return false;
    return phases.every(
        p =>
            p.phase_id.trim().length > 0 &&
            p.flow >= 0 &&
            Number.isFinite(p.flow) &&
            p.saturation_flow > 0 &&
            Number.isFinite(p.saturation_flow),
    );
};

export const ControlPlayground = () => {
    // Default = primer nodo válido del seed (ver KNOWN_NODE_IDS). Antes
    // arrancaba con 'INT_001' como texto libre — un default heredado del
    // ControlView original (commit 52a33cb7, [Fase 10c][HU015]) que NO
    // existe en graph_nodes y hacía que el primer "Recomendar" del Admin
    // siempre devolviera 422 unknown_intersection. El selector reemplaza
    // el input de texto libre por una elección entre node_ids reales.
    const [intersectionId, setIntersectionId] = useState<string>(KNOWN_NODE_IDS[0]);
    const [lostTime, setLostTime] = useState(8);
    const [phases, setPhases] = useState<PhaseFlow[]>(initialPhases);
    const { status, data, error, mutate, reset } = useRecommendControl();

    const metrics = useMemo(() => {
        const flowTotal = phases.reduce((acc, p) => acc + (p.flow || 0), 0);
        const Y = phases.reduce(
            (acc, p) => acc + (p.saturation_flow > 0 ? p.flow / p.saturation_flow : 0),
            0,
        );
        return { flowTotal, Y, isPeak: flowTotal > 1500 };
    }, [phases]);

    const submit = () => {
        if (!formIsValid(intersectionId, phases)) return;
        const state: IntersectionState = {
            intersection_id: intersectionId.trim(),
            timestamp: new Date().toISOString(),
            phases: phases.map(p => ({
                phase_id: p.phase_id.trim(),
                flow: p.flow,
                saturation_flow: p.saturation_flow,
                queue: p.queue ?? 0,
                has_pedestrian: p.has_pedestrian ?? false,
            })),
            lost_time: lostTime,
        };
        void mutate(state);
    };

    const lastSubmittedState: IntersectionState | null =
        status === 'success' || status === 'error'
            ? {
                  intersection_id: intersectionId,
                  timestamp: new Date().toISOString(),
                  phases,
                  lost_time: lostTime,
              }
            : null;

    const retry = () => {
        if (lastSubmittedState) submit();
    };

    const canSubmit = formIsValid(intersectionId, phases) && status !== 'loading';

    const activePresetName = useMemo(
        () => PRESETS.find(p => presetMatches(p, lostTime, phases))?.name ?? null,
        [lostTime, phases],
    );

    const applyPreset = (preset: PresetConfig) => {
        setLostTime(preset.lostTime);
        setPhases(preset.phases.map(p => ({ ...p })));
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
                <PresetButtons activePresetName={activePresetName} onApply={applyPreset} />
                <h2 className="text-white font-semibold text-lg mb-1">Estado de la intersección</h2>
                <p className="text-xs text-slate-500 mb-4">
                    Define la demanda actual y los flujos de saturación. El motor decide entre Webster (off-peak) y
                    Max Pressure (peak) automáticamente.
                </p>

                <div className="space-y-3 mb-4">
                    <div>
                        <label className="text-[11px] uppercase tracking-wide text-slate-500">
                            Intersección
                        </label>
                        <select
                            value={intersectionId}
                            onChange={e => setIntersectionId(e.target.value)}
                            className="w-full bg-slate-900 border border-slate-700 rounded-md px-2 py-1.5 text-sm text-slate-100 font-mono focus:outline-none focus:border-indigo-500 mt-1"
                        >
                            {KNOWN_NODE_IDS.map(nodeId => (
                                <option key={nodeId} value={nodeId}>
                                    {nodeId}
                                </option>
                            ))}
                        </select>
                    </div>
                    <Slider
                        label="Tiempo perdido por ciclo"
                        hint="suma de amarillos + all-reds del ciclo"
                        min={2}
                        max={16}
                        step={0.5}
                        unit="s"
                        value={lostTime}
                        onChange={setLostTime}
                    />
                </div>

                <PhaseEditor phases={phases} onChange={setPhases} />

                <button
                    type="button"
                    onClick={submit}
                    disabled={!canSubmit}
                    className="mt-5 w-full inline-flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white font-semibold py-2.5 rounded-lg transition-colors"
                >
                    <Play size={16} />
                    {status === 'loading' ? 'Calculando…' : 'Recomendar'}
                </button>
            </Card>

            <RecommendationPanel
                status={status}
                data={data}
                error={error}
                metrics={metrics}
                onRetry={retry}
                onReset={reset}
            />
        </div>
    );
};
