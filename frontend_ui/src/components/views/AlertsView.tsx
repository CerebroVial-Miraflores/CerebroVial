import React from 'react';

import { AlertTriangle } from 'lucide-react';
import { Card } from '../ui/Card';

// --- DATOS MOCK ---
const alerts = [
    { id: 1, type: 'Congestión', msg: 'Nivel crítico detectado en Av. Larco', time: 'Hace 2 min', details: 'Velocidad flujo < 10km/h por más de 5 minutos.' },
    { id: 2, type: 'IA', msg: 'Predicción: Bloqueo en 10 min (Pardo)', time: 'Hace 5 min', details: 'Modelo GRU detecta patrón de saturación inminente.' },
    { id: 3, type: 'Hardware', msg: 'Latencia alta en Nodo Edge #4', time: 'Hace 12 min', details: 'Ping > 500ms en dispositivo Raspberry Pi intersección 4.' },
];

export const AlertsView = () => {
    return (
        <div className="grid grid-cols-1 gap-4">
            {alerts.map((alert, idx) => (
                <Card key={idx} className="flex flex-col md:flex-row md:items-center gap-4 hover:bg-slate-800 transition-colors border-l-4 border-l-red-500">
                    <div className="flex items-center gap-4 flex-1">
                        <div className="p-3 bg-red-500/10 rounded-full">
                            <AlertTriangle className="text-red-500" />
                        </div>
                        <div>
                            <h4 className="font-bold text-white">{alert.type}: {alert.msg}</h4>
                            <p className="text-sm text-slate-400">Detectado {alert.time} • Nivel de confianza IA: 94%</p>
                        </div>
                    </div>
                    <div className="flex gap-2 justify-end">
                        <button className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded text-sm transition-colors">
                            Ver Cámara
                        </button>
                    </div>
                </Card>
            ))}
        </div>
    );
};
