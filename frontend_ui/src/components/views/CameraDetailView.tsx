import React, { useEffect, useState } from 'react';
import {
    X,
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

interface CameraDetailViewProps {
    cameraId: string;
    cameraName: string; // B1: nombre real desde /api/intersections (propagado por App)
    onBack: () => void; // Using onBack to match parent usage
}

export const CameraDetailView: React.FC<CameraDetailViewProps> = ({ cameraId, cameraName, onBack }) => {
    const [viewMode, setViewMode] = useState<'live' | 'history'>('live');
    const [streamType, setStreamType] = useState<'raw' | 'processed'>('processed'); // Default to processed
    const [quality, setQuality] = useState<'low' | 'medium' | 'high'>('high');

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

    // SSE Effect — shape §6.2 (TTH-08 6d). El backend emite `event: traffic_update`
    // con números crudos; la discretización ES y el `%` se computan client-side.
    useEffect(() => {
        const edgeApiUrl = (import.meta.env?.VITE_EDGE_API_URL) || 'http://localhost:8000';
        const sseUrl = `${edgeApiUrl}/stream/${cameraId}`;
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
    }, [cameraId]);

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
    const edgeApiUrl = (import.meta.env?.VITE_EDGE_API_URL) || 'http://localhost:8000';
    const streamUrl = `${edgeApiUrl}/video/${cameraId}?type=${streamType}&${getQualityParams()}`;

    const getCongestionStyles = (level: string) => {
        const l = (level || '').toLowerCase();
        if (l === 'alto' || l === 'heavy') return { width: '100%', color: 'bg-red-500', text: 'text-red-400' };
        if (l === 'moderado' || l === 'high') return { width: '66%', color: 'bg-amber-500', text: 'text-amber-400' };
        return { width: '33%', color: 'bg-emerald-500', text: 'text-emerald-400' };
    };

    const congestionStyle = getCongestionStyles(metrics.congestionLevel);

    return (
        <div className="fixed inset-0 bg-black/95 z-50 flex flex-col animate-in fade-in duration-200">
            {/* Header */}
            <div className="h-16 border-b border-white/10 flex items-center justify-between px-6 bg-[#0A0A0A]">
                <div className="flex items-center gap-4">
                    <button
                        onClick={onBack}
                        className="p-2 hover:bg-white/10 rounded-full transition-colors text-white mr-2"
                    >
                        <ArrowLeft className="w-6 h-6" />
                    </button>
                    <h2 className="text-xl font-bold text-white tracking-tight">{cameraName}</h2>
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
                    <button
                        onClick={onBack}
                        className="p-2 hover:bg-white/10 rounded-full transition-colors text-gray-400 hover:text-white"
                    >
                        <X className="w-6 h-6" />
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Left Panel: Stream or Chart */}
                <div className="lg:col-span-2 flex flex-col gap-6">
                    <Card className="p-0 overflow-hidden border-indigo-500/30 shadow-[0_0_30px_rgba(79,70,229,0.1)] flex-1 min-h-[400px]">
                        {viewMode === 'live' ? (
                            <div className="relative w-full h-full bg-black flex items-center justify-center">
                                <img
                                    src={streamUrl}
                                    alt="Live Stream"
                                    className="w-full h-full object-contain"
                                />
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
                            <div className="p-3 bg-slate-900/50 rounded border border-slate-800">
                                <div className="text-slate-400 text-xs mb-1 flex items-center gap-1"><Activity size={12} /> Vel. Promedio</div>
                                <div className="text-xl font-bold text-white">{Math.round(metrics.avgSpeed)} <span className="text-xs font-normal text-slate-500">km/h</span></div>
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
        </div>
    );
};
