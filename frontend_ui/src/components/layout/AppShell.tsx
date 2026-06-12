import { useEffect, useState } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { FileText, LogOut } from 'lucide-react';

import { ThesisModal } from '../modals/ThesisModal';
import { ToastProvider } from '../ui/Toast';
import { useSession } from '../../auth/SessionContext';
import { roleAllowsTab, roleLabel } from '../../auth/roles';
import { NAV_DESCRIPTORS, tabForPath } from './navigation';

// FASE 0 rediseño UI — shell único de la app (reemplaza Sidebar + Header).
// Estética calcada del topbar/rail de design/cerebrovial-ui-concept.html.
// Mobile-first: en <md el rail no existe y la navegación vive en una bottom-nav
// fija; el rail de iconos con tooltips aparece recién en ≥md. Alturas por
// flex/min-h-0 + vars --h-topbar/--w-rail/--h-bottomnav (sin calc(100vh-*)).

// Logo semáforo del prototipo (header del concepto, calcado 1:1).
function BrandLogo() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="4" y="3" width="16" height="18" rx="4" />
      <circle cx="12" cy="8.4" r="1.7" fill="currentColor" stroke="none" />
      <circle cx="12" cy="15.6" r="1.7" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function AppShell() {
  const { role, logout } = useSession();
  const navigate = useNavigate();
  const location = useLocation();

  const [showThesis, setShowThesis] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // El tab activo se deriva de la URL: navegar = navigate(), nunca setState.
  const activeTab = tabForPath(location.pathname);
  const navItems = NAV_DESCRIPTORS.filter((item) => roleAllowsTab(role, item.tab));

  // FASE 3: ToastProvider envuelve TODO el shell (topbar incluida) — cualquier
  // vista o control de la app puede disparar toasts. Cierra el pendiente de
  // Fase 1 (el provider solo envolvía /ui-lab).
  return (
    <ToastProvider>
    <div className="flex h-dvh flex-col selection:bg-brand selection:text-white">
      {/* Topbar 60px — marca, pill de sesión/rol, reloj es-PE, logout */}
      <header className="z-40 flex h-(--h-topbar) shrink-0 items-center gap-3 border-b border-line bg-canvas/75 px-3 backdrop-blur-xl sm:px-4">
        <div className="flex items-center gap-2.5">
          <div className="grid h-[31px] w-[31px] place-items-center rounded-[9px] bg-linear-to-br from-brand to-accent text-white shadow-lg shadow-brand/45">
            <BrandLogo />
          </div>
          <div className="leading-tight">
            <span className="block text-[14.5px] font-bold tracking-[0.2px]">CerebroVial</span>
            <span className="hidden text-[9.5px] font-semibold uppercase tracking-[0.16em] text-ink-2 sm:block">
              Centro de comando
            </span>
          </div>
        </div>

        <div className="flex-1" />

        {/* Acceso a ThesisModal en <md (en ≥md vive al pie del rail). Zona protegida. */}
        <button
          type="button"
          onClick={() => setShowThesis(true)}
          aria-label="Ver Ficha de Tesis"
          className="grid h-9 w-9 place-items-center rounded-[10px] border border-line bg-panel text-ink-2 transition-colors hover:border-line-2 hover:text-ink md:hidden"
        >
          <FileText size={16} />
        </button>

        {/* Pill de estado de sesión/rol (estilo .pill del prototipo) */}
        <div className="flex items-center gap-2 rounded-full border border-ok/35 bg-ok/10 px-3 py-1.5 text-[11.5px] font-bold tracking-[0.04em] text-ok">
          <span className="h-[7px] w-[7px] animate-pulse rounded-full bg-current" />
          {roleLabel(role)}
        </div>

        {/* Reloj es-PE (oculto en viewport angosto, como el prototipo) */}
        <span className="num hidden min-w-[74px] text-right text-[14.5px] font-semibold sm:block">
          {currentTime.toLocaleTimeString('es-PE', { hour12: false })}
        </span>

        <button
          type="button"
          onClick={() => logout()}
          className="flex items-center gap-2 rounded-[10px] border border-line bg-panel px-3 py-2 text-[12.5px] text-ink-2 transition-colors hover:border-line-2 hover:text-ink"
        >
          <LogOut size={15} />
          <span className="hidden sm:inline">Cerrar sesión</span>
        </button>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* Rail de iconos 66px — solo ≥md, tooltips a la derecha (estilo .rail del prototipo) */}
        <nav
          aria-label="Navegación principal"
          className="hidden w-(--w-rail) shrink-0 flex-col items-center gap-1.5 border-r border-line bg-canvas/50 py-3.5 backdrop-blur-md md:flex"
        >
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.tab;
            return (
              <button
                key={item.tab}
                type="button"
                onClick={() => navigate(item.path)}
                aria-current={isActive ? 'page' : undefined}
                className={`group relative grid h-[42px] w-[42px] place-items-center rounded-xl transition-colors duration-200 ${
                  isActive
                    ? 'bg-linear-to-br from-brand to-accent text-white shadow-lg shadow-brand/45'
                    : 'text-ink-2 hover:bg-panel-2 hover:text-ink'
                }`}
              >
                <Icon size={19} />
                <span className="pointer-events-none absolute left-[54px] top-1/2 z-10 -translate-x-1.5 -translate-y-1/2 whitespace-nowrap rounded-lg border border-line-2 bg-canvas-2 px-2.5 py-[5px] text-[11.5px] text-ink opacity-0 transition-all duration-200 ease-fluid group-hover:translate-x-0 group-hover:opacity-100">
                  {item.label}
                </span>
              </button>
            );
          })}

          {/* Acceso a ThesisModal en ≥md. Zona protegida: migra, nunca se elimina. */}
          <div className="mt-auto flex flex-col items-center gap-1.5">
            <span className="h-px w-[26px] bg-line" />
            <button
              type="button"
              onClick={() => setShowThesis(true)}
              aria-label="Ver Ficha de Tesis"
              className="group relative grid h-[42px] w-[42px] place-items-center rounded-xl text-ink-2 transition-colors duration-200 hover:bg-panel-2 hover:text-ink"
            >
              <FileText size={19} />
              <span className="pointer-events-none absolute left-[54px] top-1/2 z-10 -translate-x-1.5 -translate-y-1/2 whitespace-nowrap rounded-lg border border-line-2 bg-canvas-2 px-2.5 py-[5px] text-[11.5px] text-ink opacity-0 transition-all duration-200 ease-fluid group-hover:translate-x-0 group-hover:opacity-100">
                Ficha de Tesis
              </span>
            </button>
          </div>
        </nav>

        {/* Área de contenido: único scroll; las vistas llenan alto vía flex/min-h-0.
            El padding-bottom en <md reserva el alto de la bottom-nav + safe-area iOS. */}
        <main className="min-w-0 flex-1 overflow-y-auto scrollbar-thin-grey">
          <div className="p-4 pb-[calc(var(--h-bottomnav)+env(safe-area-inset-bottom)+16px)] md:p-6 md:pb-6">
            <Outlet />
          </div>
        </main>
      </div>

      {/* Bottom-nav — solo <md, mismos tabs del rol */}
      <nav
        aria-label="Navegación inferior"
        className="fixed inset-x-0 bottom-0 z-40 border-t border-line bg-canvas/85 pb-[env(safe-area-inset-bottom)] backdrop-blur-xl md:hidden"
      >
        <div className="flex h-(--h-bottomnav) items-stretch">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.tab;
            return (
              <button
                key={item.tab}
                type="button"
                onClick={() => navigate(item.path)}
                aria-current={isActive ? 'page' : undefined}
                className={`flex min-w-0 flex-1 flex-col items-center justify-center gap-1 px-1 transition-colors duration-200 ${
                  isActive ? 'text-brand-2' : 'text-ink-2'
                }`}
              >
                <Icon size={20} />
                <span className="max-w-full truncate text-[10px] font-medium">{item.label}</span>
              </button>
            );
          })}
        </div>
      </nav>

      {showThesis && <ThesisModal onClose={() => setShowThesis(false)} />}
    </div>
    </ToastProvider>
  );
}
