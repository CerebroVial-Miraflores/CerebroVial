import React from 'react';
import { User, Cpu } from 'lucide-react';
import { Card } from '../ui/Card';

export const AdminView = () => {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
                <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                    <User size={18} className="text-indigo-400" /> Gestión de Usuarios
                </h3>
                <p className="text-slate-400 text-sm">
                    La gestión de usuarios se entrega en una historia futura.
                </p>
            </Card>
            <Card>
                <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                    <Cpu size={18} className="text-indigo-400" /> Estado del Sistema
                </h3>
                <div className="space-y-4 text-sm">
                    <div className="flex justify-between">
                        <span className="text-slate-400">API Gateway (Azure)</span>
                        <span className="text-emerald-400 font-mono">ONLINE (34ms)</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-slate-400">Database (PostgreSQL + Timescale)</span>
                        <span className="text-emerald-400 font-mono">ONLINE</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-slate-400">Servicio Predicción (PyTorch)</span>
                        <span className="text-emerald-400 font-mono">IDLE</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="text-slate-400">Nodos Edge (Raspberry Pi)</span>
                        <span className="text-yellow-400 font-mono">33/34 ONLINE</span>
                    </div>
                </div>
            </Card>
        </div>
    );
};
