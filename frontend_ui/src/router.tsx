import { Navigate, Outlet, createBrowserRouter, type RouteObject } from 'react-router-dom';

import { AppShell } from './components/layout/AppShell';
import { LoginView } from './auth/LoginView';
import { ProtectedRoute } from './auth/ProtectedRoute';
import { DefaultTabRedirect, RoleRoute } from './auth/RoleRoute';
import { SessionProvider } from './auth/SessionContext';
import { CommandView } from './components/views/command/CommandView';
import { CameraDetailRoute } from './components/views/CameraDetailRoute';
import { AnalyticsView } from './components/views/AnalyticsView';
import { AdminView } from './components/views/AdminView';
import { ControlView } from './components/views/control/ControlView';
import { TomTomView } from './tomtom/TomTomView';

// FASE 0 rediseño UI: navegación por rutas reales bajo el AppShell (layout con
// Outlet). El shell deriva el tab activo de la URL; cada ruta protege su tab con
// RoleRoute (ruta no permitida → redirect al default del rol, lo que cubre
// también el aterrizaje post-login de LoginView en '/').
// FASE 3: el index es CommandView (Centro de Comando, fusiona dashboard +
// congestión + alertas); /alertas y /congestion son redirects legacy (D3.1a);
// /camara/:id es el puente temporal al detalle v1 (muere en Fase 4).
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
                    <CommandView />
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
              // D3.1(a) — redirects legacy SIN RoleRoute: el destino ("/") ya
              // protege con su propio guard (manager rebota a /analitica).
              { path: 'alertas', element: <Navigate to="/?panel=alertas" replace /> },
              { path: 'congestion', element: <Navigate to="/" replace /> },
              // PUENTE TEMPORAL Fase 3 → muere en Fase 4 (flujo de visión
              // on-demand nuevo). Guard del tab dashboard: es flujo del comando.
              {
                path: 'camara/:id',
                element: (
                  <RoleRoute tab="dashboard">
                    <CameraDetailRoute />
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
              // Galería del design system — SOLO DEV. El ternario se elimina
              // estáticamente en build (import.meta.env.DEV → false): el chunk
              // no se emite y /ui-lab cae al catch-all → default del rol.
              // Sin RoleRoute: no es un Tab (cualquier rol autenticado, solo DEV).
              ...(import.meta.env.DEV
                ? [
                    {
                      path: 'ui-lab',
                      lazy: () =>
                        import('./components/views/UiLabView').then((m) => ({
                          Component: m.default,
                        })),
                    },
                  ]
                : []),
              { path: '*', element: <DefaultTabRedirect /> },
            ],
          },
        ],
      },
    ],
  },
];

export const router = createBrowserRouter(appRoutes);
