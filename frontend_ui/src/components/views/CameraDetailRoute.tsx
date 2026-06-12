// FASE 3 rediseño UI — PUENTE TEMPORAL /camara/:id.
//
// Monta CameraDetailView (v1) TAL CUAL para no perder el flujo de visión
// on-demand entre fases: en v1 la abría DashboardRoute con estado local; el
// comando nuevo enlaza acá ("Detalle de cámara" del drawer / click en nodo en
// la costura 3A). ESTE PUENTE MUERE EN FASE 4 (las vistas de cámara nuevas lo
// reemplazan junto con CameraDetailView).
import { useNavigate, useParams } from 'react-router-dom';

import { useIntersections } from '../../hooks/useIntersections';
import { Button } from '../ui/Button';
import { CameraDetailView } from './CameraDetailView';

export function CameraDetailRoute() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data, loading, error, refetch } = useIntersections();

  const camera = data?.find((i) => i.id === id) ?? null;

  if (loading) {
    return (
      <p className="p-6 text-[12.5px] text-ink-2" role="status">
        Cargando cámara…
      </p>
    );
  }

  if (error !== null) {
    return (
      <div className="flex flex-col items-start gap-3 p-6">
        <p className="text-[12.5px] text-bad">{error}</p>
        <Button onClick={() => void refetch()}>Reintentar</Button>
      </div>
    );
  }

  if (camera === null) {
    return (
      <div className="flex flex-col items-start gap-3 p-6">
        <p className="text-[12.5px] text-ink-2">
          La cámara <span className="num">{id}</span> no existe en el inventario.
        </p>
        <Button onClick={() => navigate('/')}>Volver al comando</Button>
      </div>
    );
  }

  return (
    <CameraDetailView
      cameraId={camera.id}
      cameraName={camera.name}
      streamUrl={camera.stream_url}
      onBack={() => navigate('/')}
    />
  );
}
