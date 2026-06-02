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
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
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

      <main className="ml-20 md:ml-64 p-4 md:p-8 relative min-h-screen">
        <Header activeTab={activeTab} currentTime={currentTime} />

        <RoleGate allowed={['operator']}>
          {activeTab === ('dashboard' as Tab) && (
            selectedCameraId ? (
              <CameraDetailView
                cameraId={selectedCameraId}
                onBack={() => setSelectedCameraId(null)}
              />
            ) : (
              <DashboardView onSelectCamera={setSelectedCameraId} />
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
      </main>

      {showThesis && <ThesisModal onClose={() => setShowThesis(false)} />}
    </div>
  );
};

export default CerebroVialApp;
