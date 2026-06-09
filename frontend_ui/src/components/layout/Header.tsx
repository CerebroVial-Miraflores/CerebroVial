import React from 'react';
import { Map as MapIcon } from 'lucide-react';

interface HeaderProps {
    activeTab: string;
    currentTime: Date;
}

export const Header = ({ activeTab, currentTime }: HeaderProps) => {
    return (
        <header className="flex justify-between items-center mb-8 pr-4 md:pr-8">
            <div>
                <h1 className="text-2xl font-bold text-white mb-1">
                    {activeTab === 'dashboard' && 'Centro de Control de Tráfico'}
                    {activeTab === 'analytics' && 'Predicción y Análisis de Datos'}
                    {activeTab === 'alerts' && 'Gestión de Incidentes'}
                    {activeTab === 'admin' && 'Configuración del Sistema'}
                    {activeTab === 'control' && 'Motor Adaptativo de Semáforos'}
                    {activeTab === 'tomtom' && 'Tráfico en vivo (TomTom)'}
                </h1>
                <p className="text-slate-400 text-sm flex items-center gap-2">
                    <MapIcon size={14} /> Distrito de Miraflores, Lima - {currentTime.toLocaleDateString()}
                </p>
            </div>
            <div className="flex items-center gap-4">
                <div className="bg-slate-900 border border-slate-700 px-4 py-2 rounded-lg flex items-center gap-2 text-sm">
                    <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                    <span className="text-slate-300">Sistema IA: </span>
                    <span className="text-emerald-400 font-semibold">ACTIVO</span>
                </div>
                <div className="text-right hidden sm:block">
                    <p className="text-2xl font-mono text-white font-light">
                        {currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                </div>
            </div>
        </header>
    );
};
