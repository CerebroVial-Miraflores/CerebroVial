import { Outlet, createBrowserRouter } from 'react-router-dom';

import App from './App';
import { LoginView } from './auth/LoginView';
import { ProtectedRoute } from './auth/ProtectedRoute';
import { SessionProvider } from './auth/SessionContext';

export const router = createBrowserRouter([
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
        children: [{ path: '*', element: <App /> }],
      },
    ],
  },
]);
