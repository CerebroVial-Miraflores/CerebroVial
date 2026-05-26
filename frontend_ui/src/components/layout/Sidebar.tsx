import React from 'react';
import {
    Map as MapIcon,
    BarChart3,
    AlertTriangle,
    Settings,
    SlidersHorizontal,
    Cpu,
    FileText,
    LogOut
} from 'lucide-react';

import { useSession } from '../../auth/SessionContext';

interface SidebarProps {
    activeTab: string;
    setActiveTab: (tab: string) => void;
    setShowThesis: (show: boolean) => void;
}

export const Sidebar = ({ activeTab, setActiveTab, setShowThesis }: SidebarProps) => {
    const { logout } = useSession();
    return (
        <aside className="fixed left-0 top-0 h-full w-20 md:w-64 bg-slate-900 border-r border-slate-800 flex flex-col z-20 transition-all duration-300">
            <div className="p-6 flex items-center gap-3 border-b border-slate-800">
                <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center shadow-[0_0_15px_rgba(79,70,229,0.5)]">
                    <Cpu className="text-white w-5 h-5" />
                </div>
                <span className="font-bold text-lg text-white hidden md:block tracking-tight">CerebroVial</span>
            </div>

            <nav className="flex-1 p-4 space-y-2">
                <button
                    onClick={() => setActiveTab('dashboard')}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${activeTab === 'dashboard' ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/50' : 'hover:bg-slate-800 text-slate-400'}`}
                >
                    <MapIcon size={20} /> <span className="hidden md:block">Monitoreo (C4)</span>
                </button>

                <button
                    onClick={() => setActiveTab('analytics')}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${activeTab === 'analytics' ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/50' : 'hover:bg-slate-800 text-slate-400'}`}
                >
                    <BarChart3 size={20} /> <span className="hidden md:block">Analítica e IA</span>
                </button>

                <button
                    onClick={() => setActiveTab('alerts')}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${activeTab === 'alerts' ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/50' : 'hover:bg-slate-800 text-slate-400'}`}
                >
                    <AlertTriangle size={20} />
                    <span className="hidden md:block">Alertas</span>
                    <span className="ml-auto bg-red-500 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full hidden md:block">3</span>
                </button>

                <button
                    onClick={() => setActiveTab('admin')}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${activeTab === 'admin' ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/50' : 'hover:bg-slate-800 text-slate-400'}`}
                >
                    <Settings size={20} /> <span className="hidden md:block">Administración</span>
                </button>

                <button
                    onClick={() => setActiveTab('control')}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${activeTab === 'control' ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/50' : 'hover:bg-slate-800 text-slate-400'}`}
                >
                    <SlidersHorizontal size={20} /> <span className="hidden md:block">Motor Adaptativo</span>
                </button>
            </nav>

            <div className="p-4 border-t border-slate-800">
                <button
                    onClick={() => setShowThesis(true)}
                    className="w-full flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 text-indigo-400 border border-indigo-500/30 py-2 rounded-lg text-sm transition-colors"
                >
                    <FileText size={16} /> <span className="hidden md:block">Ver Ficha de Tesis</span>
                </button>
                <div className="mt-4 flex items-center gap-3 px-2">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-500 to-indigo-600 flex items-center justify-center text-xs font-bold text-white">
                        OP
                    </div>
                    <div className="hidden md:block">
                        <p className="text-sm font-medium text-white">Operador C4</p>
                        <p className="text-xs text-slate-500">En línea</p>
                    </div>
                </div>
                <button
                    type="button"
                    onClick={() => logout()}
                    className="mt-4 w-full flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 py-2 rounded-lg text-sm transition-colors"
                >
                    <LogOut size={16} /> <span className="hidden md:block">Cerrar sesión</span>
                </button>
            </div>
        </aside>
    );
};
