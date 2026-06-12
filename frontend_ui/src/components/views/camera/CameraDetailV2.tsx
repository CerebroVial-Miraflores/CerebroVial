import { useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

import { useCameras } from '../../../hooks/useCameras';
import type { PredictionHistoryInterval } from '../../../types/predictionHistory';
import { SegmentedControl } from '../../ui/SegmentedControl';
import { CameraRail } from './CameraRail';
import { CameraLivePanel } from './CameraLivePanel';
import { CameraHistoryPanel } from './CameraHistoryPanel';
import type { StreamType, Quality } from './cameraControls';

// FASE 4 rediseño UI — detalle de cámara rediseñado (/camara/:id, tab dashboard
// → operator). Reemplaza IN-PLACE el puente CameraDetailRoute + CameraDetailView
// v1. La URL (:id) es la fuente de verdad de la cámara activa: navegar por el
// carril cambia la ruta y re-resuelve; el panel se re-monta por `key={id}` (alta
// on-demand limpia, SSE nuevo, métricas reseteadas), mientras el carril y los
// controles persisten por estar fuera de la `key`.

type ViewMode = 'live' | 'history';

const VIEW_OPTIONS = [
  { value: 'live' as const, label: 'En vivo' },
  { value: 'history' as const, label: 'Histórico' },
];

const STREAM_OPTIONS = [
  { value: 'processed' as const, label: 'Procesado' },
  { value: 'raw' as const, label: 'Raw' },
];

const QUALITY_OPTIONS = [
  { value: 'low' as const, label: 'Baja' },
  { value: 'medium' as const, label: 'Media' },
  { value: 'high' as const, label: 'Alta' },
];

export function CameraDetailV2() {
  const { id = '' } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data: cameras, loading, error, refetch } = useCameras();

  // Controles fuera de la `key` del panel → persisten al navegar entre cámaras.
  const [viewMode, setViewMode] = useState<ViewMode>('live');
  const [streamType, setStreamType] = useState<StreamType>('processed');
  const [quality, setQuality] = useState<Quality>('high');
  const [interval, setInterval] = useState<PredictionHistoryInterval>(5);

  if (loading && cameras === null) {
    return (
      <p className="p-6 text-[12.5px] text-ink-2" role="status">
        Cargando cámara…
      </p>
    );
  }

  if (error !== null && cameras === null) {
    return (
      <div className="flex flex-col items-start gap-3 p-6">
        <p className="text-[12.5px] text-bad">{error}</p>
        <button
          type="button"
          onClick={() => void refetch()}
          className="rounded-btn border border-line bg-panel px-3 py-[7px] text-[11.5px] font-semibold text-ink hover:border-line-2"
        >
          Reintentar
        </button>
      </div>
    );
  }

  const camera = cameras?.find((c) => c.id === id) ?? null;

  if (camera === null) {
    return (
      <div className="flex flex-col items-start gap-3 p-6">
        <p className="text-[12.5px] text-ink-2">
          La cámara <span className="num">{id}</span> no existe en el inventario.
        </p>
        <button
          type="button"
          onClick={() => navigate('/')}
          className="rounded-btn border border-line bg-panel px-3 py-[7px] text-[11.5px] font-semibold text-ink hover:border-line-2"
        >
          Volver al comando
        </button>
      </div>
    );
  }

  return (
    <div className="flex min-h-full animate-fade-in flex-col gap-4">
      {/* Header: back + nombre + toggle vista; controles de stream sólo en vivo */}
      <header className="flex flex-wrap items-center gap-x-4 gap-y-3">
        <button
          type="button"
          onClick={() => navigate('/')}
          aria-label="Volver al comando"
          className="grid h-9 w-9 place-items-center rounded-full text-ink transition-colors hover:bg-panel-2"
        >
          <ArrowLeft size={20} aria-hidden="true" />
        </button>
        <h2 className="text-lg font-bold tracking-tight">{camera.name}</h2>
        <SegmentedControl
          ariaLabel="Modo de vista"
          options={VIEW_OPTIONS}
          value={viewMode}
          onChange={setViewMode}
        />
        {viewMode === 'live' && (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 sm:ml-auto">
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-bold uppercase tracking-[0.1em] text-ink-3">
                Stream
              </span>
              <SegmentedControl
                ariaLabel="Tipo de stream"
                options={STREAM_OPTIONS}
                value={streamType}
                onChange={setStreamType}
              />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-bold uppercase tracking-[0.1em] text-ink-3">
                Calidad
              </span>
              <SegmentedControl
                ariaLabel="Calidad del stream"
                options={QUALITY_OPTIONS}
                value={quality}
                onChange={setQuality}
              />
            </div>
          </div>
        )}
      </header>

      {/* Carril de otras cámaras (lazy HLS), navegable por la URL */}
      <div className="border-b border-line pb-1">
        <div className="mb-1 px-0.5 text-[10px] font-bold uppercase tracking-[0.12em] text-ink-3">
          Otras cámaras en vivo
        </div>
        <CameraRail cameras={cameras ?? []} activeId={camera.id} onSelect={(cid) => navigate(`/camara/${cid}`)} />
      </div>

      {viewMode === 'live' ? (
        <CameraLivePanel
          key={camera.id}
          cameraId={camera.id}
          streamUrl={camera.stream_url}
          streamType={streamType}
          quality={quality}
        />
      ) : (
        <CameraHistoryPanel
          cameraId={camera.id}
          interval={interval}
          onIntervalChange={setInterval}
        />
      )}
    </div>
  );
}
