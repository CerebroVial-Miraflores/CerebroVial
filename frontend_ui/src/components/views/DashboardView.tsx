import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Car, TrendingDown, Activity, ShieldCheck, AlertTriangle, Filter, Download } from 'lucide-react';
import { Card } from '../ui/Card';
import { LoadingOverlay } from '../ui/LoadingStates';
import { MapContainer, TileLayer, Marker, Popup, useMap, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import type { VisionStreamPayload } from '../../types/visionStream';
import { CameraGrid } from '../CameraGrid';
import type { PlayerStatus } from '../HlsPlayer';
import { openCongestionStream } from '../../services/congestionSseClient';
import { markerVisual, type MarkerVisual } from '../../utils/markerVisual';

// Fix for default marker icon in React Leaflet
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

const DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;

interface IntersectionData {
    id: string;
    name: string;
    speed: number;
    flow: number;
    status: string;
    lat: number;
    lng: number;
    stream_url: string | null;
}

// Component to handle map movement. El flyTo va en un useEffect y SOLO corre cuando center/zoom
// cambian de verdad (selección de cámara). Si se llamara en el cuerpo del render, se dispararía
// en cada re-render (cada tick del SSE, cada hover) y pisaría el zoom/pan del usuario. El ref
// `first` salta el montaje inicial para no pelear con el encuadre de FitBounds.
function MapUpdater({ center, zoom }: { center: [number, number], zoom: number }) {
    const map = useMap();
    const first = useRef(true);
    useEffect(() => {
        if (first.current) { first.current = false; return; }
        map.flyTo(center, zoom);
    }, [map, center, zoom]);
    return null;
}

// B2.10 — marcador como componente hijo: el icono (color de congestión + pulso/tachado de
// salud) se memoiza y NO depende del hover, así cambiar selectedId (o un tick del SSE que
// re-renderiza el padre) no recrea el divIcon ni reinicia el animate-ping ni huérfana el
// tooltip nativo. El resaltado por hover se aplica como box-shadow sobre el contenedor del
// icono ya montado (sin setIcon), replicando el ring anterior.
function IntersectionMarker({ int, visual, selected, speedFlow, onSelect, onSelectCamera, onHover, onUnhover }: {
    int: IntersectionData;
    visual: MarkerVisual;
    selected: boolean;
    speedFlow: { speed: number; flow: number } | undefined;
    onSelect: () => void;
    onSelectCamera: () => void;
    onHover: () => void;
    onUnhover: () => void;
}) {
    const markerRef = useRef<L.Marker>(null);

    const icon = useMemo(() => {
        const pulseSpan = visual.pulse
            ? `<span class="absolute inline-flex h-full w-full rounded-full ${visual.color} opacity-75 animate-ping"></span>`
            : '';
        // Offline = tachado diagonal sobre el punto (color de congestión conservado).
        const struckSpan = visual.struck
            ? `<span class="absolute inline-flex h-0.5 w-7 bg-white rotate-45 rounded-full shadow"></span>`
            : '';
        return L.divIcon({
            className: 'custom-marker',
            html: `<div class="relative flex items-center justify-center w-6 h-6">
                     ${pulseSpan}
                     <span class="relative inline-flex rounded-full h-4 w-4 ${visual.color} border-2 border-white shadow-lg ${visual.struck ? 'opacity-60' : ''}"></span>
                     ${struckSpan}
                   </div>`,
            iconSize: [24, 24],
            iconAnchor: [12, 12],
        });
    }, [visual.color, visual.pulse, visual.struck]);

    // Resaltado por hover: box-shadow sobre el contenedor ya montado, sin recrear el icono.
    // Replica el ring anterior (1px de offset slate-900 + 2px indigo-400). Re-aplica tras
    // recrear el icono (dep `icon`) porque setIcon devuelve un elemento nuevo.
    useEffect(() => {
        const el = markerRef.current?.getElement();
        if (!el) return;
        el.style.borderRadius = '9999px';
        el.style.boxShadow = selected ? '0 0 0 1px #0f172a, 0 0 0 3px #818cf8' : '';
    }, [selected, icon]);

    return (
        <Marker
            ref={markerRef}
            position={[int.lat, int.lng]}
            icon={icon}
            eventHandlers={{
                click: onSelect,
                mouseover: onHover,
                mouseout: onUnhover,
            }}
        >
            <Tooltip direction="top" offset={[0, -12]} opacity={1} permanent={false}>
                <div className="text-center">
                    <div className="font-bold text-slate-900 text-xs">{int.name}</div>
                    <div className="text-[10px] text-slate-600">
                        {speedFlow?.speed ?? '--'} km/h • {speedFlow?.flow ?? '--'} vpm
                    </div>
                </div>
            </Tooltip>
            <Popup className="custom-popup">
                <div className="p-1 min-w-[120px]">
                    <h4 className="font-bold text-slate-900 text-xs mb-1">{int.name}</h4>
                    <div className="flex flex-col text-[10px] text-slate-600 gap-1">
                        <div className="flex justify-between border-b border-slate-100 pb-1">
                            <span>Velocidad:</span>
                            <span className="font-bold text-indigo-600">{speedFlow?.speed ?? int.speed} km/h</span>
                        </div>
                        <div className="flex justify-between border-b border-slate-100 pb-1">
                            <span>Flujo:</span>
                            <span className="font-bold text-indigo-600">{speedFlow?.flow ?? int.flow} vpm</span>
                        </div>
                        <div className="flex justify-between">
                            <span>Estado:</span>
                            <span className={`font-bold uppercase ${
                                int.status === 'critical' ? 'text-red-500' :
                                int.status === 'moderate' ? 'text-amber-500' : 'text-emerald-500'
                            }`}>
                                {int.status === 'critical' ? 'Crítico' :
                                 int.status === 'moderate' ? 'Moderado' : 'Fluido'}
                            </span>
                        </div>
                    </div>
                    <button
                        onClick={onSelectCamera}
                        className="mt-2 w-full bg-indigo-600 text-white text-[10px] py-1 rounded hover:bg-indigo-700"
                    >
                        Ver Cámara
                    </button>
                </div>
            </Popup>
        </Marker>
    );
}

// B2: encuadra el mapa a TODAS las intersecciones una sola vez (cuando cargan), en vez
// del center/zoom fijo. El guard por ref evita re-encuadrar en cada render y no pelea con
// el flyTo por click de MapUpdater (que corre después, ante una selección del usuario).
function FitBounds({ points }: { points: [number, number][] }) {
    const map = useMap();
    const done = useRef(false);
    useEffect(() => {
        if (done.current || points.length === 0) return;
        done.current = true;
        map.fitBounds(L.latLngBounds(points), { padding: [40, 40] });
    }, [map, points]);
    return null;
}

export const DashboardView = ({ onSelectCamera }: { onSelectCamera: (id: string, name: string, streamUrl: string | null) => void }) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    // B2 D2: el SSE de visión alimenta SOLO speed/flow (señal de cámara). El color del
    // marcador NO sale de acá: lo manda la congestión de red (Waze) en `int.status`.
    const [realData, setRealData] = useState<Record<string, { speed: number, flow: number }>>({});
    // B2: salud de cámara compartida (lazy-aware). Solo contiene estados REPORTADOS por las
    // celdas de la grilla; un id AUSENTE = desconocido → el marcador pulsa normal, NUNCA
    // offline. La grilla vive en otra sub-pestaña; este record persiste entre sub-tabs.
    const [cameraHealth, setCameraHealth] = useState<Record<string, PlayerStatus>>({});
    // B2: cross-selección hover (mapa↔lista↔grilla). Resalta el elemento que matchea.
    const [selectedId, setSelectedId] = useState<string | null>(null);
    // B1: sub-pestañas del tab dashboard. 'map' = el mapa/dashboard que YA existe (sin cambios,
    // B2 lo mejora). 'cameras' = la grilla de previews HLS.
    const [dashTab, setDashTab] = useState<'map' | 'cameras'>('map');

    const [intersections, setIntersections] = useState<IntersectionData[]>([]);

    const [mapCenter, setMapCenter] = useState<[number, number]>([-12.122, -77.028]);
    const [mapZoom, setMapZoom] = useState(14);
    const [viewMode, setViewMode] = useState<'leaflet' | 'waze'>('leaflet');

    // Carga la lista (incluye la congestión REAL por intersección en `status`). Elevada a
    // useCallback: la llaman tanto la carga inicial como el wake del SSE de congestión.
    const fetchIntersections = useCallback(async () => {
        try {
            const apiBaseUrl = (import.meta.env?.VITE_CORE_API_URL) || 'http://localhost:8001';
            const response = await fetch(`${apiBaseUrl}/api/intersections`);
            if (response.ok) {
                const data = await response.json();
                setIntersections(data);
                setError(null);
            } else {
                console.error("Failed to fetch intersections", response.statusText);
                setError("No se pudieron cargar las intersecciones desde la base de datos.");
            }
        } catch (err) {
            console.error("Error fetching intersections:", err);
            setError("Error de red al conectar con el servidor.");
        } finally {
            setLoading(false);
        }
    }, []);

    // Carga inicial.
    useEffect(() => {
        fetchIntersections();
        return () => {
            setViewMode('leaflet');
        };
    }, [fetchIntersections]);

    // B2 D2 — SSE de CONGESTIÓN de red (core 8001, patrón HU-22). Wake sin payload →
    // re-lee /api/intersections y recolorea los marcadores con el nivel Waze más reciente.
    // Es la fuente primaria del color; convive con el SSE de visión de abajo (speed/flow).
    useEffect(() => {
        const controller = openCongestionStream({ onWake: () => void fetchIntersections() });
        return () => controller.abort();
    }, [fetchIntersections]);

    // SSE de VISIÓN (edge 8000): alimenta SOLO speed/flow del tooltip/popup (señal de
    // cámara). Sin YOLO está vacío; el color del marcador NO depende de esto (D2).
    React.useEffect(() => {
        const eventSources: EventSource[] = [];
        const baseUrl = (import.meta.env?.VITE_EDGE_API_URL) || 'http://localhost:8000';

        intersections.forEach(camera => {
            const sseUrl = `${baseUrl}/stream/${camera.id}`;
            const eventSource = new EventSource(sseUrl);

            eventSource.addEventListener('traffic_update', (event) => {
                try {
                    const data = JSON.parse(event.data) as VisionStreamPayload;
                    const m = data.metrics;
                    setRealData(prev => ({
                        ...prev,
                        [camera.id]: {
                            speed: Math.round(m.mean_speed_kmh ?? 0),
                            flow: m.unique_vehicles,
                        }
                    }));
                } catch (err) {
                    console.error(`Error parsing SSE data for ${camera.id}:`, err);
                }
            });

            eventSource.onerror = () => {
                // Silently handle errors to avoid console spam when backend is down
            };

            eventSources.push(eventSource);
        });

        return () => {
            // Cleanup all connections when component unmounts
            eventSources.forEach(es => es.close());
        };
    }, [intersections]); // Run when intersections are loaded

    // B2: reporta estable la salud de una cámara (id ausente = desconocido). No re-set si
    // no cambió, para no re-renderizar de más.
    const reportCameraHealth = useCallback((id: string, status: PlayerStatus) => {
        setCameraHealth(prev => (prev[id] === status ? prev : { ...prev, [id]: status }));
    }, []);

    const handleCameraSelect = (id: string) => {
        const camera = intersections.find(c => c.id === id);
        if (camera) {
            setMapCenter([camera.lat, camera.lng]);
            setMapZoom(16);
            if (viewMode === 'waze') setViewMode('leaflet');
            onSelectCamera(id, camera.name, camera.stream_url);
        }
    };

    return (
        <div className="space-y-6 animate-fade-in">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2">Centro de Control de Tráfico</h1>
                    <p className="text-slate-400">Monitoreo en tiempo real y gestión de incidentes</p>
                </div>
                <div className="flex gap-3">
                    <button className="bg-slate-800 hover:bg-slate-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors border border-slate-700">
                        <Filter size={18} />
                        Filtros
                    </button>
                    <button className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors shadow-lg shadow-indigo-500/20">
                        <Download size={18} />
                        Exportar Reporte
                    </button>
                </div>
            </div>

            {/* B1: toggle Mapa | Cámaras dentro del tab dashboard (operator-only). 'Mapa' es
                el dashboard/mapa que YA existe, sin cambios (B2 lo mejora). 'Cámaras' = la grilla. */}
            <div className="flex gap-2">
                <button
                    onClick={() => setDashTab('map')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${dashTab === 'map' ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'}`}
                >
                    Mapa
                </button>
                <button
                    onClick={() => setDashTab('cameras')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${dashTab === 'cameras' ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'}`}
                >
                    Cámaras
                </button>
            </div>

            {dashTab === 'cameras' ? (
                <CameraGrid
                    cameras={intersections.map(i => ({ id: i.id, name: i.name, stream_url: i.stream_url }))}
                    onSelectCamera={onSelectCamera}
                    onStatusChange={reportCameraHealth}
                    selectedId={selectedId}
                    onHover={setSelectedId}
                />
            ) : (
            <>
            {/* FILA DE KPIs */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="bg-gradient-to-br from-indigo-900/40 to-slate-800/40 border-indigo-500/20">
                    <div className="flex justify-between items-start mb-2">
                        <div className="p-2 bg-indigo-500/20 rounded-lg"><Car className="text-indigo-400" size={20} /></div>
                    </div>
                    <h3 className="text-3xl font-bold text-white">1,245</h3>
                    <p className="text-sm text-slate-400">Vehículos detectados (Hora)</p>
                </Card>

                <Card className="bg-gradient-to-br from-rose-900/40 to-slate-800/40 border-rose-500/20">
                    <div className="flex justify-between items-start mb-2">
                        <div className="p-2 bg-rose-500/20 rounded-lg"><TrendingDown className="text-rose-400" size={20} /></div>
                        <span className="text-xs text-slate-400 font-mono">KPI-02</span>
                    </div>
                    <h3 className="text-3xl font-bold text-white">22 km/h</h3>
                    <p className="text-sm text-slate-400">Velocidad Promedio (Red)</p>
                </Card>

                <Card className="bg-gradient-to-br from-amber-900/40 to-slate-800/40 border-amber-500/20">
                    <div className="flex justify-between items-start mb-2">
                        <div className="p-2 bg-amber-500/20 rounded-lg"><Activity className="text-amber-400" size={20} /></div>
                    </div>
                    <h3 className="text-3xl font-bold text-white">ALTA</h3>
                    <p className="text-sm text-slate-400">Predicción Congestión (15m)</p>
                </Card>

                <Card className="bg-gradient-to-br from-emerald-900/40 to-slate-800/40 border-emerald-500/20">
                    <div className="flex justify-between items-start mb-2">
                        <div className="p-2 bg-emerald-500/20 rounded-lg"><ShieldCheck className="text-emerald-400" size={20} /></div>
                    </div>
                    <h3 className="text-3xl font-bold text-white">34/34</h3>
                    <p className="text-sm text-slate-400">Semáforos Conectados</p>
                </Card>
            </div>

            {/* Main Grid */}
            <div className="grid grid-cols-12 gap-6 h-[600px]">
                {/* Map Section — mapa a todo el ancho (la lista lateral rica con previews es
                    feature futura del dashboard; por ahora el mapa respira sin columna lateral). */}
                <div className="col-span-12 bg-slate-800 rounded-xl border border-slate-700 overflow-hidden relative shadow-2xl flex flex-col">
                    {/* Map Header / Toggle */}
                    <div className="absolute top-4 left-4 z-[400] flex gap-2">
                        <div className="bg-slate-900/90 backdrop-blur px-3 py-1 rounded-full border border-slate-700 text-xs text-white font-medium shadow-lg flex items-center gap-2">
                            <span>Vista:</span>
                            <div className="flex bg-slate-800 rounded p-0.5">
                                <button
                                    onClick={() => setViewMode('leaflet')}
                                    className={`px-2 py-0.5 rounded text-[10px] transition-colors ${viewMode === 'leaflet' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}
                                >
                                    Interactivo
                                </button>
                                <button
                                    onClick={() => setViewMode('waze')}
                                    className={`px-2 py-0.5 rounded text-[10px] transition-colors ${viewMode === 'waze' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'}`}
                                >
                                    Waze / Tráfico
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Map Content */}
                    <div className="w-full h-full bg-slate-900 relative z-0">
                        {loading && <LoadingOverlay message="Cargando mapa interactivo..." />}
                        {error && (
                            <div className="absolute inset-0 z-[500] flex flex-col items-center justify-center bg-slate-900/90 text-white p-6 text-center">
                                <AlertTriangle size={48} className="text-rose-500 mb-4" />
                                <h3 className="text-xl font-bold mb-2">Error de Conexión</h3>
                                <p className="text-slate-400 mb-6 max-w-md">{error}</p>
                                <button 
                                    onClick={() => window.location.reload()}
                                    className="bg-indigo-600 px-6 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
                                >
                                    Reintentar
                                </button>
                            </div>
                        )}
                        {viewMode === 'leaflet' ? (
                            <MapContainer
                                center={[-12.122, -77.028]}
                                zoom={14}
                                style={{ height: '100%', width: '100%' }}
                                zoomControl={false}
                            >
                                <TileLayer
                                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                                />
                                <MapUpdater center={mapCenter} zoom={mapZoom} />
                                <FitBounds points={intersections.map(i => [i.lat, i.lng] as [number, number])} />

                                {intersections.map((int) => (
                                    // B2: dos dimensiones que no se pisan — color = congestión real (int.status,
                                    // Waze); pulso/tachado = salud de la cámara. Desconocido (id ausente del
                                    // record) pulsa normal, NUNCA offline.
                                    <IntersectionMarker
                                        key={int.id}
                                        int={int}
                                        visual={markerVisual(int.status, cameraHealth[int.id])}
                                        selected={int.id === selectedId}
                                        speedFlow={realData[int.id]}
                                        onSelect={() => handleCameraSelect(int.id)}
                                        onSelectCamera={() => onSelectCamera(int.id, int.name, int.stream_url)}
                                        onHover={() => setSelectedId(int.id)}
                                        onUnhover={() => setSelectedId(null)}
                                    />
                                ))}
                            </MapContainer>
                        ) : (
                            <iframe
                                src="https://embed.waze.com/iframe?zoom=14&lat=-12.122&lon=-77.028&ct=livemap"
                                width="100%"
                                height="100%"
                                allowFullScreen
                                className="w-full h-full"
                                style={{ border: 0 }}
                            ></iframe>
                        )}
                    </div>
                </div>
            </div>
            </>
            )}
        </div>
    );
};
