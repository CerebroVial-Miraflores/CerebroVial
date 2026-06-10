import { useState } from 'react';

import { DashboardView } from './DashboardView';
import { CameraDetailView } from './CameraDetailView';

// FASE 0 rediseño UI: wrapper a nivel ruta del conmutador grilla ↔ detalle que
// vivía en App.tsx. B1: la selección lleva id + nombre real (de /api/intersections)
// para que el detalle no hardcodee el nombre. C1/F1: lleva también la stream_url
// de Claro para que el detalle orqueste el alta on-demand del YOLO en el edge
// (POST /cameras/{id}). Delta consciente de FASE 0: al navegar fuera de '/' este
// componente se desmonta y la selección se resetea (antes persistía en App).
export function DashboardRoute() {
  const [selectedCamera, setSelectedCamera] = useState<{
    id: string;
    name: string;
    streamUrl: string | null;
  } | null>(null);

  if (selectedCamera) {
    return (
      <CameraDetailView
        cameraId={selectedCamera.id}
        cameraName={selectedCamera.name}
        streamUrl={selectedCamera.streamUrl}
        onBack={() => setSelectedCamera(null)}
      />
    );
  }

  return (
    <DashboardView
      onSelectCamera={(id, name, streamUrl) => setSelectedCamera({ id, name, streamUrl })}
    />
  );
}
