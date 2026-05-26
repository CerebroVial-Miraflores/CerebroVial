import React, { useState, useEffect } from 'react';

import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { DashboardView } from './components/views/DashboardView';
import { CameraDetailView } from './components/views/CameraDetailView';
import { AnalyticsView } from './components/views/AnalyticsView';
import { AlertsView } from './components/views/AlertsView';
import { AdminView } from './components/views/AdminView';
import { ControlView } from './components/views/control/ControlView';
import { ThesisModal } from './components/modals/ThesisModal';

const CerebroVialApp = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [showThesis, setShowThesis] = useState(false);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500 selection:text-white">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        setShowThesis={setShowThesis}
      />

      <main className="ml-20 md:ml-64 p-4 md:p-8 relative min-h-screen">
        <Header activeTab={activeTab} currentTime={currentTime} />

        {activeTab === 'dashboard' && (
          selectedCameraId ? (
            <CameraDetailView
              cameraId={selectedCameraId}
              onBack={() => setSelectedCameraId(null)}
            />
          ) : (
            <DashboardView onSelectCamera={setSelectedCameraId} />
          )
        )}
        {activeTab === 'analytics' && <AnalyticsView />}
        {activeTab === 'alerts' && <AlertsView />}
        {activeTab === 'admin' && <AdminView />}
        {activeTab === 'control' && <ControlView />}
      </main>

      {showThesis && <ThesisModal onClose={() => setShowThesis(false)} />}
    </div>
  );
};

export default CerebroVialApp;