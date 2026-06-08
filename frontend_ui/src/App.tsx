import React, { useCallback, useEffect, useState } from 'react';

import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { DashboardView } from './components/views/DashboardView';
import { CameraDetailView } from './components/views/CameraDetailView';
import { AnalyticsView } from './components/views/AnalyticsView';
import { AlertsView } from './components/views/AlertsView';
import { AdminView } from './components/views/AdminView';
import { ControlView } from './components/views/control/ControlView';
import { CongestionMapView } from './components/views/CongestionMapView';
import { TomTomView } from './tomtom/TomTomView';
import { ThesisModal } from './components/modals/ThesisModal';

import { useSession } from './auth/SessionContext';
import { RoleGate } from './auth/RoleGate';
import { TABS_BY_ROLE, defaultTabForRole, type Tab } from './auth/roles';

const CerebroVialApp = () => {
  const { role } = useSession();

  // HU-01: el tab por defecto depende del rol. Inicialización perezosa
  // para no recalcular en cada render.
  const [activeTab, setActiveTabInternal] = useState<string>(
    () => defaultTabForRole(role) ?? 'dashboard',
  );
  const [showThesis, setShowThesis] = useState(false);
  // B1: la selección lleva id + nombre real (de /api/intersections) para que el detalle
  // no hardcodee el nombre. C1/F1: lleva también la stream_url de Claro para que el
  // detalle orqueste el alta on-demand del YOLO en el edge (POST /cameras/{id}).
  const [selectedCamera, setSelectedCamera] = useState<{ id: string; name: string; streamUrl: string | null } | null>(null);
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // HU-01: cambiar a un tab fuera del rol redirige al default del rol,
  // nunca a un estado intermedio. Defensa en profundidad junto a RoleGate
  // (RNF-INT-07 + validación dual lado cliente).
  const setActiveTab = useCallback(
    (tab: string) => {
      if (!role) return;
      const allowed = TABS_BY_ROLE[role] as readonly string[];
      if (allowed.includes(tab)) {
        setActiveTabInternal(tab);
      } else {
        const fallback = defaultTabForRole(role);
        if (fallback) setActiveTabInternal(fallback);
      }
    },
    [role],
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500 selection:text-white">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        setShowThesis={setShowThesis}
      />

      {/* <main> acotado al viewport como flex column: el Header es hermano de alto automático
          y el wrapper de abajo (flex-1 min-h-0 overflow-y-auto) toma el alto restante y es el
          ÚNICO scroll del área de contenido — el body no scrollea (overflow-hidden). Robusto a
          cambios de alto del Header (sin número mágico): el wrapper se reajusta solo. */}
      <main className="ml-20 md:ml-64 py-4 pl-4 md:py-8 md:pl-8 relative h-screen flex flex-col overflow-hidden">
        <Header activeTab={activeTab} currentTime={currentTime} />

        {/* Wrapper de scroll SIN padding derecho → llega al borde derecho del viewport, así la
            scrollbar (.scrollbar-thin-grey) queda pegada al borde. El gap derecho del contenido
            (32px) lo repone el wrapper interno de abajo (pr-4 md:pr-8). El left 288 (ml-64+pl-8)
            no se toca. */}
        <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin-grey">
        <div className="pr-4 md:pr-8">
        <RoleGate allowed={['operator']}>
          {activeTab === ('dashboard' as Tab) && (
            selectedCamera ? (
              <CameraDetailView
                cameraId={selectedCamera.id}
                cameraName={selectedCamera.name}
                streamUrl={selectedCamera.streamUrl}
                onBack={() => setSelectedCamera(null)}
              />
            ) : (
              <DashboardView
                onSelectCamera={(id, name, streamUrl) => setSelectedCamera({ id, name, streamUrl })}
              />
            )
          )}
          {activeTab === ('alerts' as Tab) && <AlertsView />}
        </RoleGate>

        {/* HU-05 / DHU-020: 'control' lo ven operator (vista pasiva) y admin
            (playground). El switch interno de ControlView decide qué render. */}
        <RoleGate allowed={['operator', 'admin']}>
          {activeTab === ('control' as Tab) && <ControlView />}
        </RoleGate>

        <RoleGate allowed={['manager']}>
          {activeTab === ('analytics' as Tab) && <AnalyticsView />}
        </RoleGate>

        <RoleGate allowed={['admin']}>
          {activeTab === ('admin' as Tab) && <AdminView />}
        </RoleGate>

        {/* HU-22: el mapa de congestión en tiempo real es operator-only. */}
        <RoleGate allowed={['operator']}>
          {activeTab === 'congestion' && <CongestionMapView />}
        </RoleGate>

        {/* Track feature/tomtom (EXPERIMENTAL, Fase A): tráfico en vivo de TomTom,
            operator-only (misma familia que 'congestion'). */}
        <RoleGate allowed={['operator']}>
          {activeTab === 'tomtom' && <TomTomView />}
        </RoleGate>
        </div>
        </div>
      </main>

      {showThesis && <ThesisModal onClose={() => setShowThesis(false)} />}
    </div>
  );
};

export default CerebroVialApp;
