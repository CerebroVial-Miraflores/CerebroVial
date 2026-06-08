import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Car, TrendingDown, Activity, ShieldCheck, AlertTriangle } from 'lucide-react';
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
import { HlsPlayer } from '../HlsPlayer';

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
function IntersectionMarker({ int, visual, selected, speedFlow, onSelect, onHover, onUnhover }: {
    int: IntersectionData;
    visual: MarkerVisual;
    selected: boolean;
    speedFlow: { speed: number; flow: number } | undefined;
    onSelect: () => void;
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
        </Marker>
    );
}

// F3 — miniplayer HLS anclado al marcador. Popup HIJO DEL MAPA (no del Marker): con `position`
// Leaflet lo auto-abre al montar y lo remueve al desmontar. `autoClose=false` deja que varios
// convivan; `closeButton=false` mata el ✕ nativo de Leaflet. La INTERACCIÓN:
//   · cerrar = reclick del marcador (toggle puro en handleMarkerClick) — NO hay ✕ propio.
//   · click sobre el VIDEO = navegar al detalle embebido (onOpenDetail → onSelectCamera).
// El HLS se monta SOLO acá: id fuera del set → este componente no se renderiza → desmonta →
// HlsPlayer corre su cleanup (hls.destroy()). Solo el video flotando (sin nombre, sin marco
// blanco — el card/tip de Leaflet se neutraliza por CSS scopeado a `.miniplayer-popup`).
// `position` MEMOIZADO en lat/lng primitivos: sin esto, un array literal nuevo por render (cada
// tick SSE re-renderiza DashboardView) re-corre el effect del Popup → remove/re-add de la capa
// → flicker (el "temblor", que era esto y NO el pulse del marker, que vive en otro pane).
function MiniPlayerPopup({ int, onOpenDetail }: { int: IntersectionData; onOpenDetail: () => void }) {
    const position = useMemo<[number, number]>(() => [int.lat, int.lng], [int.lat, int.lng]);
    return (
        <Popup
            position={position}
            autoClose={false}
            closeOnClick={false}
            closeButton={false}
            className="miniplayer-popup"
        >
            <div
                onClick={onOpenDetail}
                title={int.name}
                className="w-52 aspect-video bg-black rounded-lg overflow-hidden cursor-pointer"
            >
                {int.stream_url ? (
                    <HlsPlayer src={int.stream_url} controls={false} />
                ) : (
                    <div className="w-full h-full flex items-center justify-center text-rose-400 text-[11px] font-bold tracking-wide">
                        SIN STREAM
                    </div>
                )}
            </div>
        </Popup>
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
    // B1: sub-pestañas del tab dashboard. 'cameras' = la grilla de previews HLS (default,
    // vista primaria de monitoreo). 'map' = el mapa con markers de congestión.
    const [dashTab, setDashTab] = useState<'map' | 'cameras'>('cameras');

    const [intersections, setIntersections] = useState<IntersectionData[]>([]);

    // Centro/zoom inicial del encuadre (FitBounds ajusta al cargar; MapUpdater haría flyTo solo
    // ante un cambio — hoy no cambian, el miniplayer no recentra). Constantes, sin setters.
    const [mapCenter] = useState<[number, number]>([-12.122, -77.028]);
    const [mapZoom] = useState(14);
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

    // F3: ids con miniplayer abierto en el mapa. Vacío al montar (todos cerrados). El HLS se
    // monta SOLO dentro del Popup de un id en este set; quitar el id desmonta el Popup → cleanup.
    const [openPlayers, setOpenPlayers] = useState<Set<string>>(new Set());

    // Click en marcador = TOGGLE puro: si no tiene miniplayer → lo abre; si ya lo tiene → lo
    // cierra (quita del set → desmonta el Popup → hls.destroy()). La navegación al detalle NO
    // vive acá: la dispara el click sobre el video del miniplayer (onOpenDetail → onSelectCamera).
    // Varios miniplayers pueden estar abiertos a la vez (Popup autoClose=false).
    const handleMarkerClick = useCallback((id: string) => {
        setOpenPlayers(prev => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    }, []);

    return (
        <div className="space-y-6 animate-fade-in">
            {/* B1: toggle Cámaras | Mapa dentro del tab dashboard (operator-only). 'Cámaras' =
                la grilla de previews HLS (default). 'Mapa' = el mapa con markers de congestión. */}
            <div className="flex gap-2">
                <button
                    onClick={() => setDashTab('cameras')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${dashTab === 'cameras' ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'}`}
                >
                    Cámaras
                </button>
                <button
                    onClick={() => setDashTab('map')}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors border ${dashTab === 'map' ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'}`}
                >
                    Mapa
                </button>
            </div>

            {/* FILA DE KPIs — transversal a Cámaras y Mapa (valores estáticos por ahora). */}
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
                                        onSelect={() => handleMarkerClick(int.id)}
                                        onHover={() => setSelectedId(int.id)}
                                        onUnhover={() => setSelectedId(null)}
                                    />
                                ))}

                                {/* F3: miniplayers HLS abiertos — uno por id en openPlayers, a nivel de
                                    mapa (no del marcador). Montar auto-abre el Popup; cerrar lo quita del
                                    set → desmonta → hls.destroy(). Lazy: cerrado = NO montado. */}
                                {intersections
                                    .filter((int) => openPlayers.has(int.id))
                                    .map((int) => (
                                        <MiniPlayerPopup
                                            key={`mp-${int.id}`}
                                            int={int}
                                            onOpenDetail={() => onSelectCamera(int.id, int.name, int.stream_url)}
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
