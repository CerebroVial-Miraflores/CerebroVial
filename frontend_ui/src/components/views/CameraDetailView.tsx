import React, { useEffect, useMemo, useState } from 'react';
import {
    Users,
    Activity,
    Car,
    AlertTriangle,
    Zap,
    ArrowLeft
} from 'lucide-react';
import { Card } from '../ui/Card';
import { predictionService, type PredictionResult } from '../../services/predictionService';
import { TrafficHistoryWidget } from '../widgets/TrafficHistoryWidget';
import type { VisionStreamPayload } from '../../types/visionStream';
import { congestionLabel, densityPercent } from '../../utils/trafficLabels';
import { CameraStrip } from '../CameraStrip';
import type { GridCamera } from '../CameraGrid';

interface CameraDetailViewProps {
    cameraId: string;
    cameraName: string; // B1: nombre real desde /api/intersections (propagado por App)
    streamUrl: string | null; // C1/F1: HLS de Claro; se manda al edge como `source` del YOLO on-demand
    onBack: () => void; // Using onBack to match parent usage
}

const EDGE_API_URL = (import.meta.env?.VITE_EDGE_API_URL) || 'http://localhost:8000';
const CORE_API_URL = (import.meta.env?.VITE_CORE_API_URL) || 'http://localhost:8001';

type ViewMode = 'live' | 'history';
type StreamType = 'raw' | 'processed';
type Quality = 'low' | 'medium' | 'high';

/**
 * Contenedor del detalle de cámara. Posee:
 *  - `current`: la cámara mostrada (init desde props; el carril la reemplaza al click).
 *  - El carril horizontal de las OTRAS cámaras (lazy HLS) arriba del panel.
 *  - Los controles de vista (live/histórico, tipo, calidad) en el header — viven acá para
 *    persistir al cambiar de cámara.
 *
 * El panel YOLO/SSE/métricas (`CameraDetailPanel`) va `key`-eado por la cámara actual: al
 * elegir otra cámara en el carril se re-monta limpio (nueva alta on-demand, nuevo SSE,
 * métricas reseteadas), mientras el carril persiste por estar fuera de la `key`.
 */
export const CameraDetailView: React.FC<CameraDetailViewProps> = ({ cameraId, cameraName, streamUrl, onBack }) => {
    const [current, setCurrent] = useState<GridCamera>({ id: cameraId, name: cameraName, stream_url: streamUrl });
    const [viewMode, setViewMode] = useState<ViewMode>('live');
    const [streamType, setStreamType] = useState<StreamType>('processed'); // Default to processed
    const [quality, setQuality] = useState<Quality>('high');

    // Lista completa de cámaras para el carril (mismo origen que el dashboard).
    const [allCameras, setAllCameras] = useState<GridCamera[]>([]);

    useEffect(() => {
        let cancelled = false;
        // Carril liviano: /api/cameras (solo CameraDB, sin agregado Waze). El detalle no usa
        // `status`, así que no debe pagar el query de congestión que sí necesita el dashboard.
        fetch(`${CORE_API_URL}/api/cameras`)
            .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`core ${res.status}`))))
            .then((data: Array<{ id: string; name: string; stream_url: string | null }>) => {
                if (!cancelled && Array.isArray(data)) {
                    setAllCameras(data.map((d) => ({ id: d.id, name: d.name, stream_url: d.stream_url })));
                }
            })
            .catch((err) => {
                // El carril es accesorio: si la lista no carga, el detalle sigue funcionando.
                console.error('No se pudo cargar la lista de cámaras para el carril', err);
            });
        return () => {
            cancelled = true;
        };
    }, []);

    // El carril muestra solo las OTRAS cámaras (excluye la activa).
    const otherCameras = useMemo(
        () => allCameras.filter((c) => c.id !== current.id),
        [allCameras, current.id],
    );

    const handleSelectCamera = (id: string, name: string, url: string | null) => {
        if (id === current.id) return;
        setCurrent({ id, name, stream_url: url });
    };

    return (
        // Detalle EMBEBIDO en el área de contenido (no overlay full-screen): vive dentro del
        // <main>, a la derecha del sidebar y debajo del Header transversal, que quedan visibles.
        // SIN marco/borde/fondo propio: llena el área directamente sobre el slate-950 de la app
        // (como DashboardView en el mismo slot), sin verse como "caja dentro de caja". El detalle
        // CRECE con su contenido (sin h-full/overflow-hidden); el scroll lo toma el wrapper externo
        // de <main> (flex-1 min-h-0 overflow-y-auto) → un solo scroll, el externo. La barra de
        // título se fija con `sticky top-0` (ver header). El back-flow (onBack →
        // setSelectedCamera(null)) no cambia.
        <div className="flex flex-col animate-in fade-in duration-200">
            {/* Barra de título FIJA: sticky top-0 dentro del scroll del wrapper de <main>.
                bg-slate-950 (mismo color del área, no #0A0A0A) la hace opaca para ocluir el
                carril/contenido que scrollea por detrás, sin reintroducir la caja negra. */}
            <div className="h-16 border-b border-white/10 flex items-center justify-between sticky top-0 z-10 bg-slate-950">
                <div className="flex items-center gap-4">
                    <button
                        onClick={onBack}
                        className="p-2 hover:bg-white/10 rounded-full transition-colors text-white mr-2"
                    >
                        <ArrowLeft className="w-6 h-6" />
                    </button>
                    <h2 className="text-xl font-bold text-white tracking-tight">{current.name}</h2>
                    <div className="flex bg-white/5 rounded-lg p-1">
                        <button
                            onClick={() => setViewMode('live')}
                            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${viewMode === 'live'
                                ? 'bg-blue-600 text-white shadow-lg'
                                : 'text-gray-400 hover:text-white'
                                }`}
                        >
                            Analítica en tiempo Real
                        </button>
                        <button
                            onClick={() => setViewMode('history')}
                            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${viewMode === 'history'
                                ? 'bg-blue-600 text-white shadow-lg'
                                : 'text-gray-400 hover:text-white'
                                }`}
                        >
                            Histórico
                        </button>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    {viewMode === 'live' && (
                        <>
                            {/* Stream Type Toggle */}
                            <div className="flex items-center gap-2 bg-black/40 rounded-lg p-1 border border-white/10">
                                <button
                                    onClick={() => setStreamType('processed')}
                                    className={`px-3 py-1 rounded-md text-xs font-bold transition-all flex items-center gap-1 ${streamType === 'processed'
                                        ? 'bg-indigo-600 text-white shadow-sm'
                                        : 'text-gray-500 hover:text-gray-300'
                                        }`}
                                >
                                    <Zap size={12} /> PROCESADO
                                </button>
                                <button
                                    onClick={() => setStreamType('raw')}
                                    className={`px-3 py-1 rounded-md text-xs font-bold transition-all ${streamType === 'raw'
                                        ? 'bg-white/20 text-white shadow-sm ring-1 ring-white/10'
                                        : 'text-gray-500 hover:text-gray-300'
                                        }`}
                                >
                                    RAW
                                </button>
                            </div>

                            {/* Quality Selector */}
                            <div className="flex items-center gap-2 bg-black/40 rounded-lg p-1 border border-white/10">
                                <span className="text-xs text-gray-500 px-2 font-medium">CALIDAD</span>
                                {(['low', 'medium', 'high'] as const).map((q) => (
                                    <button
                                        key={q}
                                        onClick={() => setQuality(q)}
                                        className={`px-3 py-1 rounded-md text-xs font-bold transition-all ${quality === q
                                            ? 'bg-white/20 text-white shadow-sm ring-1 ring-white/10'
                                            : 'text-gray-500 hover:text-gray-300'
                                            }`}
                                    >
                                        {q.toUpperCase()}
                                    </button>
                                ))}
                            </div>
                        </>
                    )}
                </div>
            </div>

            {/* Carril de las otras cámaras: navegación horizontal, lazy HLS. Click → reemplaza
                el panel de detalle por la cámara elegida (y sus métricas). */}
            <div className="border-b border-white/10 pt-2">
                <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 px-1">
                    Otras cámaras en vivo
                </div>
                <CameraStrip cameras={otherCameras} onSelect={handleSelectCamera} />
            </div>

            <CameraDetailPanel
                key={current.id}
                cameraId={current.id}
                streamUrl={current.stream_url}
                viewMode={viewMode}
                streamType={streamType}
                quality={quality}
            />
        </div>
    );
};

/**
 * Panel per-cámara: orquesta el YOLO on-demand, el SSE de métricas y la predicción de la
 * cámara `cameraId`. Se monta `key`-eado por cámara, así su estado (alta, métricas) arranca
 * limpio en cada cambio. La lógica es la misma que tenía `CameraDetailView` antes del split.
 */
const CameraDetailPanel: React.FC<{
    cameraId: string;
    streamUrl: string | null;
    viewMode: ViewMode;
    streamType: StreamType;
    quality: Quality;
}> = ({ cameraId, streamUrl, viewMode, streamType, quality }) => {
    // C1/F1-F2: orquestación on-demand del YOLO en el edge.
    //  - `isActive`: el alta (POST /cameras/{id}) respondió OK → recién ahí montamos MJPEG+SSE.
    //  - `isLoading`: alta en vuelo o esperando el primer frame del MJPEG.
    //  - `error`: alta falló o la cámara de Claro está caída → mensaje claro, no spinner infinito.
    // Init perezoso (el panel se re-monta por cámara vía `key`, así que basta el valor
    // inicial — sin resets síncronos dentro del effect): cargando si hay stream que dar de
    // alta; error directo si la cámara no tiene stream configurado.
    const [isActive, setIsActive] = useState(false);
    const [isLoading, setIsLoading] = useState(() => !!streamUrl);
    const [error, setError] = useState<string | null>(
        () => (streamUrl ? null : 'Esta cámara no tiene un stream configurado.'),
    );

    // Data States
    const [metrics, setMetrics] = useState({
        vehiclesPerHour: 0,
        avgSpeed: 0,
        congestionLevel: 'Bajo',
        density: '0%',
        pedestrians: 0,
        incidents: 0,
        fps: 0,
        bitrate: 0,
        latency: 0,
        flowRate: 0
    });
    const [prediction, setPrediction] = useState<PredictionResult | null>(null);

    // C1/F1: alta on-demand del YOLO al montar. El front SOLO hace POST (idempotente
    // en el edge); la BAJA la posee el edge, no el front:
    //   - single-slot: al entrar a otra cámara, su POST libera la anterior al instante;
    //   - watchdog: al salir, el cierre del SSE + el desmontaje del <img> MJPEG bajan los
    //     consumidores a 0 y el edge libera en ≤45s (E4).
    // Por qué NO un DELETE en el cleanup: bajo <StrictMode> (dev) el ciclo es
    // mount→unmount→remount, y un DELETE en el cleanup de la 1ª pasada aterriza DESPUÉS
    // del POST del remonte → removía la cámara con el detalle abierto ("Cargando…" eterno).
    // Sin DELETE, el ciclo queda POST→POST(no-op) → cámara activa. Robusto por construcción.
    useEffect(() => {
        // Sin stream: el error ya quedó derivado en el init; no se hace alta.
        if (!streamUrl) return;

        let cancelled = false;

        fetch(`${EDGE_API_URL}/cameras/${cameraId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: streamUrl, source_type: 'hls', zones: {} }),
        })
            .then((res) => {
                if (!res.ok) throw new Error(`Edge respondió ${res.status}`);
                if (!cancelled) setIsActive(true);
            })
            .catch((err) => {
                console.error('Alta de cámara en el edge falló', err);
                if (!cancelled) {
                    setError('No se pudo iniciar la detección. La cámara de Claro podría no estar disponible.');
                    setIsLoading(false);
                }
            });

        // El cleanup solo evita setState tras desmontar; NO da de baja (lo hace el edge).
        return () => {
            cancelled = true;
        };
    }, [cameraId, streamUrl]);

    // SSE Effect — shape §6.2 (TTH-08 6d). El backend emite `event: traffic_update`
    // con números crudos; la discretización ES y el `%` se computan client-side.
    // C1/F1: solo se abre una vez que el alta en el edge respondió OK (`isActive`).
    useEffect(() => {
        if (!isActive) return;
        const sseUrl = `${EDGE_API_URL}/stream/${cameraId}`;
        const eventSource = new EventSource(sseUrl);

        eventSource.addEventListener('traffic_update', (event) => {
            try {
                const data = JSON.parse(event.data) as VisionStreamPayload;
                const m = data.metrics;
                setMetrics({
                    vehiclesPerHour: m.unique_vehicles,
                    avgSpeed: m.mean_speed_kmh ?? 0,
                    congestionLevel: congestionLabel(m.mean_occupancy),
                    density: densityPercent(m.mean_occupancy),
                    pedestrians: 0,
                    incidents: 0,
                    fps: 0,
                    bitrate: 0,
                    latency: 0,
                    flowRate: m.flow_vehicles_per_hour / 60,
                });
            } catch (e) {
                console.error("Error parsing SSE data", e);
            }
        });

        return () => {
            eventSource.close();
        };
    }, [cameraId, isActive]);

    // Prediction Polling
    useEffect(() => {
        const fetchPrediction = async () => {
            try {
                const densityVal = parseFloat((metrics.density || '0').toString().replace('%', '')) || 0;
                const result = await predictionService.predictTraffic({
                    camera_id: cameraId,
                    total_vehicles: metrics.vehiclesPerHour,
                    occupancy_rate: densityVal / 100,
                    flow_rate_per_min: metrics.vehiclesPerHour / 60,
                    avg_speed: metrics.avgSpeed,
                    avg_density: densityVal
                });
                setPrediction(result);
            } catch (err) {
                console.error("Prediction fetch failed", err);
            }
        };

        if (metrics.vehiclesPerHour > 0) {
            fetchPrediction();
        }
        const interval = setInterval(fetchPrediction, 10000);
        return () => clearInterval(interval);
    }, [metrics.vehiclesPerHour, metrics.avgSpeed, metrics.density, cameraId]);

    const getQualityParams = () => {
        switch (quality) {
            case 'low': return 'width=480&quality=50';
            case 'medium': return 'width=640&quality=70';
            case 'high': return 'width=1280&quality=85';
            default: return 'width=640&quality=70';
        }
    };

    // Include type param
    const mjpegUrl = `${EDGE_API_URL}/video/${cameraId}?type=${streamType}&${getQualityParams()}`;

    const getCongestionStyles = (level: string) => {
        const l = (level || '').toLowerCase();
        if (l === 'alto' || l === 'heavy') return { width: '100%', color: 'bg-red-500', text: 'text-red-400' };
        if (l === 'moderado' || l === 'high') return { width: '66%', color: 'bg-amber-500', text: 'text-amber-400' };
        return { width: '33%', color: 'bg-emerald-500', text: 'text-emerald-400' };
    };

    const congestionStyle = getCongestionStyles(metrics.congestionLevel);

    return (
        <div className="py-6 grid grid-cols-1 lg:grid-cols-3 gap-6">

            {/* Left Panel: Stream or Chart */}
            <div className="lg:col-span-2 flex flex-col gap-6">
                <Card className="p-0 overflow-hidden border-indigo-500/30 shadow-[0_0_30px_rgba(79,70,229,0.1)] flex-1 min-h-[400px]">
                    {viewMode === 'live' ? (
                        <div className="relative w-full h-full bg-black flex items-center justify-center">
                            {/* C1/F2: el MJPEG se monta solo tras el alta OK en el edge.
                                Mientras: overlay de carga; si falla: mensaje claro. */}
                            {error ? (
                                <div className="flex flex-col items-center justify-center gap-3 text-center px-6">
                                    <AlertTriangle className="w-10 h-10 text-amber-400" />
                                    <p className="text-sm text-slate-300 max-w-xs">{error}</p>
                                </div>
                            ) : isActive ? (
                                <img
                                    src={mjpegUrl}
                                    alt="Live Stream"
                                    className="w-full h-full object-contain"
                                    onLoad={() => setIsLoading(false)}
                                    onError={() => setError('Se perdió la señal de la cámara.')}
                                />
                            ) : null}

                            {isLoading && !error && (
                                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/60">
                                    <div className="w-8 h-8 border-2 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
                                    <span className="text-xs text-slate-300 uppercase tracking-wider">Cargando detección…</span>
                                </div>
                            )}

                            <div className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 bg-red-500/10 border border-red-500/20 backdrop-blur-md rounded-full">
                                <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                                <span className="text-xs font-bold text-red-500 uppercase tracking-wider">En vivo</span>
                            </div>
                            {/* Overlay Date */}
                            <div className="absolute top-4 right-4 bg-black/50 backdrop-blur text-white px-3 py-1 rounded text-xs font-mono border border-white/10">
                                {new Date().toLocaleTimeString()}
                            </div>
                        </div>
                    ) : (
                        <TrafficHistoryWidget cameraId={cameraId} />
                    )}
                </Card>

                {/* AI Insights Card */}
                <Card className={`bg-gradient-to-r ${prediction?.alert ? 'from-red-900/30 to-slate-800/40 border-red-500/30' : 'from-indigo-900/20 to-slate-800/40 border-indigo-500/20'}`}>
                    <h3 className={`text-sm font-bold ${prediction?.alert ? 'text-red-400' : 'text-indigo-400'} mb-3 flex items-center gap-2`}>
                        <Zap size={16} /> Insights de IA (CerebroVial)
                    </h3>
                    {prediction ? (
                        <div className="text-slate-300 text-sm leading-relaxed space-y-2">
                            <p>{prediction.message}</p>
                            <div className="grid grid-cols-3 gap-2 mt-2 text-center text-xs">
                                <div className="bg-slate-800 p-2 rounded">
                                    <div className="text-slate-400">Ahora</div>
                                    <div className="font-bold text-white">{prediction.data.current_congestion_level}</div>
                                </div>
                                <div className="bg-slate-800 p-2 rounded">
                                    <div className="text-slate-400">+15 min</div>
                                    <div className={`font-bold ${prediction.data.predicted_congestion_15min === 'Heavy' ? 'text-red-400' : 'text-white'}`}>
                                        {prediction.data.predicted_congestion_15min}
                                    </div>
                                </div>
                                <div className="bg-slate-800 p-2 rounded">
                                    <div className="text-slate-400">+30 min</div>
                                    <div className="font-bold text-white">{prediction.data.predicted_congestion_30min}</div>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <p className="text-slate-400 text-sm italic">Analizando patrones...</p>
                    )}
                </Card>
            </div>

            {/* Right Sidebar: Metrics */}
            <div className="flex flex-col gap-4">
                <Card>
                    <h3 className="font-bold text-white mb-4 text-sm">Métricas en Tiempo Real</h3>
                    <div className="grid grid-cols-2 gap-3">
                        <div className="p-3 bg-slate-900/50 rounded border border-slate-800">
                            <div className="text-slate-400 text-xs mb-1 flex items-center gap-1"><Car size={12} /> Vehículos detectados</div>
                            <div className="text-xl font-bold text-white">{Math.round(metrics.vehiclesPerHour)}</div>
                        </div>
                        {/* C1/F3: la velocidad es aproximada/experimental (sin calibrar,
                            DEUDA-SPEED-CALIB). El conteo de vehículos sí es real. */}
                        <div
                            className="p-3 bg-slate-900/50 rounded border border-slate-800"
                            title="Velocidad experimental, sin calibrar (DEUDA-SPEED-CALIB)"
                        >
                            <div className="text-slate-400 text-xs mb-1 flex items-center gap-1">
                                <Activity size={12} /> Vel. Promedio
                                <span className="text-amber-400/80 text-[10px] font-semibold uppercase ml-auto">aprox.</span>
                            </div>
                            <div className="text-xl font-bold text-white">~{Math.round(metrics.avgSpeed)} <span className="text-xs font-normal text-slate-500">km/h</span></div>
                        </div>
                        <div className="p-3 bg-slate-900/50 rounded border border-slate-800">
                            <div className="text-slate-400 text-xs mb-1 flex items-center gap-1"><Users size={12} /> Peatones</div>
                            <div className="text-xl font-bold text-white">{metrics.pedestrians}</div>
                        </div>
                        <div className="p-3 bg-slate-900/50 rounded border border-slate-800">
                            <div className="text-slate-400 text-xs mb-1 flex items-center gap-1"><AlertTriangle size={12} /> Incidentes</div>
                            <div className={`text-xl font-bold ${metrics.incidents > 0 ? 'text-amber-400' : 'text-white'}`}>{metrics.incidents}</div>
                        </div>
                    </div>
                </Card>

                <Card>
                    <h3 className="font-bold text-white mb-4 text-sm">Estado del Tráfico</h3>
                    <div className="space-y-4">
                        <div>
                            <div className="flex justify-between text-xs text-slate-400 mb-1">
                                <span>Congestión</span>
                                <span className={`${congestionStyle.text} font-bold`}>{metrics.congestionLevel}</span>
                            </div>
                            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                                <div
                                    className={`${congestionStyle.color} h-full transition-all duration-500`}
                                    style={{ width: congestionStyle.width }}
                                ></div>
                            </div>
                        </div>
                        <div>
                            <div className="flex justify-between text-xs text-slate-400 mb-1">
                                <span>Densidad Vehicular</span>
                                <span className="text-white font-bold">{metrics.density}</span>
                            </div>
                            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                                <div
                                    className="bg-indigo-500 h-full transition-all duration-500"
                                    style={{ width: metrics.density }}
                                ></div>
                            </div>
                        </div>
                    </div>
                </Card>

                <Card className="bg-slate-800/30 mt-auto">
                    <h3 className="font-bold text-white mb-3 text-sm">Control de Dispositivo</h3>
                    <div className="space-y-2">
                        <button className="w-full py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-xs text-slate-300 transition-colors">
                            Reiniciar Nodo Edge
                        </button>
                        <button className="w-full py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded text-xs text-slate-300 transition-colors">
                            Calibrar Cámara
                        </button>
                        <button className="w-full py-2 bg-red-900/20 hover:bg-red-900/40 border border-red-900/50 rounded text-xs text-red-400 transition-colors">
                            Reportar Fallo
                        </button>
                    </div>
                </Card>
            </div>

        </div>
    );
};
