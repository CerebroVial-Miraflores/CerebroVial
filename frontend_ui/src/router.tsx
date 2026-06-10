import { Outlet, createBrowserRouter, type RouteObject } from 'react-router-dom';

import { AppShell } from './components/layout/AppShell';
import { LoginView } from './auth/LoginView';
import { ProtectedRoute } from './auth/ProtectedRoute';
import { DefaultTabRedirect, RoleRoute } from './auth/RoleRoute';
import { SessionProvider } from './auth/SessionContext';
import { DashboardRoute } from './components/views/DashboardRoute';
import { AnalyticsView } from './components/views/AnalyticsView';
import { AlertsView } from './components/views/AlertsView';
import { AdminView } from './components/views/AdminView';
import { ControlView } from './components/views/control/ControlView';
import { CongestionMapView } from './components/views/CongestionMapView';
import { TomTomView } from './tomtom/TomTomView';

// FASE 0 rediseño UI: navegación por rutas reales bajo el AppShell (layout con
// Outlet). El shell deriva el tab activo de la URL; cada ruta protege su tab con
// RoleRoute (ruta no permitida → redirect al default del rol, lo que cubre
// también el aterrizaje post-login de LoginView en '/'). Las vistas existentes
// se montan tal cual (su reemplazo es de fases posteriores).
// Exportado como RouteObject[] para que los tests monten createMemoryRouter.
export const appRoutes: RouteObject[] = [
  {
    element: (
      <SessionProvider>
        <Outlet />
      </SessionProvider>
    ),
    children: [
      { path: '/login', element: <LoginView /> },
      {
        element: <ProtectedRoute />,
        children: [
          {
            path: '/',
            element: <AppShell />,
            children: [
              {
                index: true,
                element: (
                  <RoleRoute tab="dashboard">
                    <DashboardRoute />
                  </RoleRoute>
                ),
              },
              {
                path: 'control',
                element: (
                  <RoleRoute tab="control">
                    <ControlView />
                  </RoleRoute>
                ),
              },
              {
                path: 'alertas',
                element: (
                  <RoleRoute tab="alerts">
                    <AlertsView />
                  </RoleRoute>
                ),
              },
              {
                path: 'congestion',
                element: (
                  <RoleRoute tab="congestion">
                    <CongestionMapView />
                  </RoleRoute>
                ),
              },
              {
                path: 'trafico',
                element: (
                  <RoleRoute tab="tomtom">
                    <TomTomView />
                  </RoleRoute>
                ),
              },
              {
                path: 'analitica',
                element: (
                  <RoleRoute tab="analytics">
                    <AnalyticsView />
                  </RoleRoute>
                ),
              },
              {
                path: 'admin',
                element: (
                  <RoleRoute tab="admin">
                    <AdminView />
                  </RoleRoute>
                ),
              },
              { path: '*', element: <DefaultTabRedirect /> },
            ],
          },
        ],
      },
    ],
  },
];

export const router = createBrowserRouter(appRoutes);
