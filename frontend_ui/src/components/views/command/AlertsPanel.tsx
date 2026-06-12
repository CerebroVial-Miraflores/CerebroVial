// FASE 3 rediseño UI — panel de alertas priorizadas (spec: .side .panel del
// prototipo). TODO el feed es MOCK señalizado (DemoBadge en el header): no
// existe fuente real de alertas. Comportamiento calcado del prototipo:
// · entrada dinámica a los ~9 s (toast pred + prepend con animación new +
//   CountBadge del header que sube),
// · acciones → toasts demo; "Ver intersección" abre el detalle REAL del nodo;
//   "Ver predicción" cambia el mapa a modo predicción (+flash);
//   "Reiniciar nodo" encadena info → ok a los 2.8 s,
// · "Aplicar plan IA" pasa el botón a done (estado local de demo).
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { AlertTriangle, CheckCheck, ScanLine, TrendingUp } from 'lucide-react';

import { AlertAccordion, type AlertSeverity } from '../../ui/AlertAccordion';
import { Button } from '../../ui/Button';
import { ConfBar } from '../../ui/ConfBar';
import { DemoBadge } from '../../ui/DemoBadge';
import { Panel } from '../../ui/Panel';
import { useToast } from '../../ui/Toast';
import {
  INCOMING_ALERT,
  INCOMING_ALERT_DELAY_MS,
  INITIAL_ALERTS,
  type AlertAction,
  type CommandAlert,
} from './mockData';

const SEVERITY_ICON: Record<AlertSeverity, ReactNode> = {
  crit: <AlertTriangle size={15} />,
  pred: <TrendingUp size={15} />,
  hw: <ScanLine size={15} />,
  info: <CheckCheck size={15} />,
};

interface AlertsPanelProps {
  /** "Ver intersección" → detalle real del nodo (3A: /camara/:id; 3B: drawer). */
  onOpenNode: (cameraId: string) => void;
  /** "Ver predicción" → modo predicción del mapa + flash. */
  onShowPrediction: () => void;
}

export function AlertsPanel({ onOpenNode, onShowPrediction }: AlertsPanelProps) {
  const { push } = useToast();
  const [alerts, setAlerts] = useState<CommandAlert[]>(INITIAL_ALERTS);
  const [openIds, setOpenIds] = useState<ReadonlySet<string>>(new Set());
  const [appliedIds, setAppliedIds] = useState<ReadonlySet<string>>(new Set());
  const [newIds, setNewIds] = useState<ReadonlySet<string>>(new Set());
  const timersRef = useRef<number[]>([]);

  // Entrada dinámica simulada (~9 s), como el prototipo. Cleanup total: el
  // unmount antes del timer no debe disparar toasts ni setState.
  useEffect(() => {
    const id = window.setTimeout(() => {
      push({
        kind: 'pred',
        title: 'Nueva alerta IA',
        body: 'Cola creciente · Pardo × Libertadores · conf 88%',
      });
      setAlerts((prev) => [INCOMING_ALERT, ...prev]);
      setNewIds((prev) => new Set(prev).add(INCOMING_ALERT.id));
    }, INCOMING_ALERT_DELAY_MS);
    timersRef.current.push(id);
    const timers = timersRef.current;
    return () => {
      for (const t of timers) window.clearTimeout(t);
    };
  }, [push]);

  const toggle = (id: string) => {
    setOpenIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const dispatch = (alert: CommandAlert, action: AlertAction) => {
    switch (action.kind) {
      case 'see':
        if (alert.cameraId) onOpenNode(alert.cameraId);
        break;
      case 'predview':
        onShowPrediction();
        break;
      case 'apply':
        setAppliedIds((prev) => new Set(prev).add(alert.id));
        push({
          kind: 'ok',
          title: 'Plan IA aplicado',
          body: `Extensión N–S +12 s · ${alert.place} · demo`,
        });
        break;
      case 'restart': {
        push({ kind: 'info', title: 'Comando enviado', body: `Reinicio de ${alert.place} en curso… · demo` });
        const id = window.setTimeout(() => {
          push({ kind: 'ok', title: `${alert.place} en línea`, body: 'Latencia 38 ms · operación normal · demo' });
        }, 2800);
        timersRef.current.push(id);
        break;
      }
      default:
        push({
          kind: 'info',
          title: 'Acción de demo',
          body: `«${action.label}» · sin efecto`,
        });
    }
  };

  return (
    <Panel
      title="Alertas priorizadas por IA"
      count={alerts.length}
      mini="orden: impacto × confianza"
      actions={<DemoBadge />}
    >
      <div className="flex max-h-[620px] flex-col gap-2 overflow-y-auto p-[9px]">
        {alerts.map((alert) => (
          <AlertAccordion
            key={alert.id}
            severity={alert.severity}
            icon={SEVERITY_ICON[alert.severity]}
            title={alert.title}
            meta={
              <>
                <span>{alert.place}</span>
                <span>{alert.timeAgo}</span>
                {alert.conf !== undefined && <span className="num">conf. {alert.conf}%</span>}
              </>
            }
            open={openIds.has(alert.id)}
            onToggle={() => toggle(alert.id)}
            isNew={newIds.has(alert.id)}
          >
            {alert.mstats && (
              <div className="mb-2.5 flex gap-4">
                {alert.mstats.map((stat) => (
                  <div key={stat.label}>
                    <span className="block text-[9.5px] font-bold uppercase tracking-[0.08em] text-ink-3">
                      {stat.label}
                    </span>
                    <b className="num text-[13.5px] text-ink">{stat.value}</b>
                  </div>
                ))}
              </div>
            )}
            {alert.conf !== undefined && <ConfBar percent={alert.conf} className="mb-2.5" />}
            {alert.body !== undefined && <p>{alert.body}</p>}
            {alert.actions.length > 0 && (
              <div className="flex flex-wrap gap-[7px]">
                {alert.actions.map((action) => {
                  const isApplied = action.kind === 'apply' && appliedIds.has(alert.id);
                  return (
                    <Button
                      key={action.label}
                      variant={isApplied ? 'done' : action.primary ? 'pri' : 'default'}
                      onClick={() => dispatch(alert, action)}
                    >
                      {isApplied ? '✓ Aplicado' : action.label}
                    </Button>
                  );
                })}
              </div>
            )}
          </AlertAccordion>
        ))}
      </div>
    </Panel>
  );
}
