import { useState } from 'react';
import { GeoJSON } from 'react-leaflet';

import { Button } from '../ui/Button';
import { Chip, HChip } from '../ui/Chip';
import { SegmentedControl } from '../ui/SegmentedControl';
import { StatusChip } from '../ui/StatusChip';
import { Pill } from '../ui/Pill';
import { CountBadge } from '../ui/CountBadge';
import { ConfBar } from '../ui/ConfBar';
import { Panel } from '../ui/Panel';
import { Modal } from '../ui/Modal';
import { Drawer } from '../ui/Drawer';
import { DemoBadge } from '../ui/DemoBadge';
import { useToast, type ToastKind } from '../ui/Toast';
import { Sparkline } from '../ui/Sparkline';
import { KpiCard } from '../ui/KpiCard';
import { MapCanvas } from '../map/MapCanvas';
import { NodeMarker } from '../map/NodeMarker';
import { MapLegend } from '../map/MapLegend';
import { MapModeBadge, type MapMode } from '../map/MapModeBadge';
import { LayerChips } from '../map/LayerChips';
import { jamLevelPathOptions } from '../map/edgeStyle';
import { MIRAFLORES_CENTER, MOCK_EDGES } from '../map/mockGeo';
import { LiveDataSection } from './uilab/LiveDataSection';

// FASE 1 rediseño UI — galería del design system (/ui-lab). SOLO existe en DEV
// (router: import.meta.env.DEV + route.lazy → fuera del bundle de producción).
// Valida visualmente cada componente con mocks contra el prototipo, en desktop
// y 390px. Default export: lo exige el route.lazy.

const SECTIONS = [
  { id: 'tokens', label: 'Tokens' },
  { id: 'button', label: 'Button' },
  { id: 'chips', label: 'Chips' },
  { id: 'estados', label: 'Estados' },
  { id: 'confbar', label: 'ConfBar' },
  { id: 'panel', label: 'Panel' },
  { id: 'sparkline', label: 'Sparkline' },
  { id: 'kpi', label: 'KpiCard' },
  { id: 'toasts', label: 'Toasts' },
  { id: 'modal', label: 'Modal' },
  { id: 'drawer', label: 'Drawer' },
  { id: 'mapa', label: 'Mapa' },
  { id: 'live', label: 'Datos en vivo' },
] as const;

const COLOR_SWATCHES = [
  { name: 'brand', className: 'bg-brand' },
  { name: 'brand-2', className: 'bg-brand-2' },
  { name: 'accent', className: 'bg-accent' },
  { name: 'ok', className: 'bg-ok' },
  { name: 'ok-road', className: 'bg-ok-road' },
  { name: 'warn', className: 'bg-warn' },
  { name: 'bad', className: 'bg-bad' },
  { name: 'sev', className: 'bg-sev' },
  { name: 'info', className: 'bg-info' },
  { name: 'ink', className: 'bg-ink' },
  { name: 'ink-2', className: 'bg-ink-2' },
  { name: 'ink-3', className: 'bg-ink-3' },
] as const;

const RADIUS_SWATCHES = [
  { name: 'btn · 9px', className: 'rounded-btn' },
  { name: 'ctl · 12px', className: 'rounded-ctl' },
  { name: 'card · 14px', className: 'rounded-card' },
  { name: 'panel · 16px', className: 'rounded-panel' },
] as const;

const SPARK_IDX = [38, 40, 44, 47, 52, 55, 58, 60, 61, 62];
const SPARK_VEL = [31, 30, 29, 27.5, 26, 25, 24, 23.2, 22.8, 22.4];
const SPARK_FLU = [760, 820, 900, 980, 1040, 1100, 1160, 1200, 1230, 1245];

function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-16 space-y-4">
      <h2 className="border-b border-line pb-2 text-[13.5px] font-bold uppercase tracking-[0.08em] text-ink-2">
        {title}
      </h2>
      {children}
    </section>
  );
}

function LabContent() {
  const { push } = useToast();

  const [chipOn, setChipOn] = useState(true);
  const [hchipOn, setHchipOn] = useState(false);
  const [segValue, setSegValue] = useState<'h' | 'n' | 'p'>('n');
  const [modalOpen, setModalOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [mapMode, setMapMode] = useState<MapMode>('live');
  const [layers, setLayers] = useState([
    { id: 'traffic', label: 'Tráfico', on: true },
    { id: 'nodes', label: 'Semáforos', on: true },
    { id: 'inc', label: 'Incidentes', on: false },
  ]);

  const trafficOn = layers.find((l) => l.id === 'traffic')?.on ?? true;
  const nodesOn = layers.find((l) => l.id === 'nodes')?.on ?? true;

  const pushKind = (kind: ToastKind, title: string, body: string) => () => push({ kind, title, body });

  return (
    <div className="mx-auto max-w-[1100px]">
      <div className="mb-2 flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-bold">UI Lab</h1>
        <DemoBadge />
      </div>
      <p className="mb-4 text-[12px] text-ink-2">
        Galería del design system del rediseño — solo DEV, no existe en producción.
      </p>

      <nav
        aria-label="Secciones"
        className="sticky top-0 z-10 -mx-4 mb-8 flex gap-3 overflow-x-auto border-b border-line bg-canvas/85 px-4 py-2 backdrop-blur-md"
      >
        {SECTIONS.map((s) => (
          <a key={s.id} href={`#${s.id}`} className="whitespace-nowrap text-[11px] font-semibold text-ink-2 transition-colors hover:text-ink">
            {s.label}
          </a>
        ))}
      </nav>

      <div className="space-y-12 pb-12">
        <Section id="tokens" title="Tokens">
          <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6">
            {COLOR_SWATCHES.map((swatch) => (
              <div key={swatch.name} className="space-y-1.5">
                <div className={`h-10 rounded-ctl border border-line ${swatch.className}`} />
                <p className="num text-[10px] text-ink-2">{swatch.name}</p>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap gap-3">
            {RADIUS_SWATCHES.map((swatch) => (
              <div key={swatch.name} className={`grid h-16 w-28 place-items-center border border-line-2 bg-panel-2 text-[10px] text-ink-2 ${swatch.className}`}>
                {swatch.name}
              </div>
            ))}
          </div>
          <p className="text-sm">
            Tipografía: sans para texto, <span className="num">0123456789 .num tabular</span> para cifras.
          </p>
        </Section>

        <Section id="button" title="Button">
          <div className="flex flex-wrap items-center gap-3">
            <Button>Simular</Button>
            <Button variant="pri">Aplicar plan IA</Button>
            <Button variant="done">✓ Aplicado · ciclo 1/3</Button>
          </div>
        </Section>

        <Section id="chips" title="Chip · HChip · SegmentedControl">
          <div className="flex flex-wrap items-center gap-3">
            <Chip on={chipOn} onToggle={setChipOn}>Tráfico</Chip>
            <Chip on={!chipOn} onToggle={(next) => setChipOn(!next)}>Cámaras</Chip>
            <HChip on={hchipOn} onToggle={setHchipOn}>+15 min</HChip>
            <HChip on={!hchipOn} onToggle={(next) => setHchipOn(!next)}>+60</HChip>
          </div>
          <SegmentedControl
            ariaLabel="Modo temporal"
            options={[
              { value: 'h', label: 'Histórico' },
              { value: 'n', label: '● Ahora' },
              { value: 'p', label: 'Predicción' },
            ]}
            value={segValue}
            onChange={setSegValue}
          />
        </Section>

        <Section id="estados" title="StatusChip · Pill · CountBadge · DemoBadge">
          <div className="flex flex-wrap items-center gap-3">
            <StatusChip status="ok">ADAPTATIVO · NORMAL</StatusChip>
            <StatusChip status="warn" />
            <StatusChip status="bad">CRÍTICO · ADAPTATIVO EN PAUSA</StatusChip>
            <StatusChip status="recovery" />
          </div>
          <div className="flex flex-wrap items-center gap-4">
            <Pill>IA ACTIVA</Pill>
            <span className="flex items-center gap-2 text-[11px] text-ink-2">
              rail: <CountBadge count={3} variant="rail" />
            </span>
            <span className="flex items-center gap-2 text-[11px] text-ink-2">
              panel: <CountBadge count={4} variant="panel" />
            </span>
            <span className="flex items-center gap-2 text-[11px] text-ink-2">
              cero (oculto): <CountBadge count={0} />
            </span>
            <DemoBadge />
          </div>
        </Section>

        <Section id="confbar" title="ConfBar">
          <div className="max-w-sm space-y-3">
            <ConfBar percent={94} />
            <ConfBar percent={61} />
            <ConfBar percent={28} />
          </div>
        </Section>

        <Section id="panel" title="Panel">
          <Panel title="Alertas priorizadas por IA" count={3} mini="orden: impacto × confianza" className="max-w-md">
            <div className="space-y-2 p-3 text-[12px] text-ink-2">
              <p>Contenido de ejemplo del panel.</p>
              <p>Las listas reales (alertas, corredores) llegan con sus vistas en Fase 3.</p>
            </div>
          </Panel>
        </Section>

        <Section id="sparkline" title="Sparkline">
          <div className="flex flex-wrap items-end gap-6">
            <Sparkline data={SPARK_IDX} className="h-9 w-32 text-warn" />
            <Sparkline data={SPARK_VEL} className="h-9 w-32 text-bad" />
            <Sparkline data={SPARK_FLU} className="h-9 w-32 text-brand" />
          </div>
        </Section>

        <Section id="kpi" title="KpiCard">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <KpiCard
              label="Índice de congestión · red"
              value={62}
              unit="/100"
              delta={{ text: '▲ +9', tone: 'neg' }}
              spark={SPARK_IDX}
              sparkClassName="text-warn"
            />
            <KpiCard
              label="Velocidad media"
              value={22.4}
              decimals={1}
              unit="km/h"
              delta={{ text: '▼ −3.2', tone: 'neg' }}
              spark={SPARK_VEL}
              sparkClassName="text-bad"
            />
            <KpiCard
              label="Flujo total"
              value={1245}
              unit="veh/h"
              delta={{ text: '＋4% vs ayer', tone: 'neu' }}
              spark={SPARK_FLU}
              sparkClassName="text-brand"
              onClick={pushKind('info', 'Acción de demo', 'KpiCard clickeable · sin efecto en /ui-lab')}
            />
            <KpiCard
              label="Predicción 15 min · red"
              value={94}
              unit="% conf."
              delta={{ text: 'GRU', tone: 'pos' }}
              warn
              footer={<ConfBar percent={94} className="mt-3" />}
            />
          </div>
        </Section>

        <Section id="toasts" title="Toasts">
          <div className="flex flex-wrap gap-3">
            <Button onClick={pushKind('ok', 'Plan aplicado', 'Extender N–S +12 s · monitoreando efecto…')}>ok</Button>
            <Button onClick={pushKind('warn', 'Latencia alta', 'Nodo Edge #4 · 312 ms')}>warn</Button>
            <Button onClick={pushKind('pred', 'Nueva alerta IA', 'Cola creciente · Pardo × Libertadores · conf 88%')}>pred</Button>
            <Button onClick={pushKind('info', 'Comando enviado', 'Reinicio de Nodo Edge #4 en curso…')}>info</Button>
          </div>
        </Section>

        <Section id="modal" title="Modal">
          <Button variant="pri" onClick={() => setModalOpen(true)}>Abrir modal</Button>
          <Modal
            open={modalOpen}
            onClose={() => setModalOpen(false)}
            title="Índice de congestión · red"
            subtitle="Últimas 3 h · línea punteada = predicción del modelo"
          >
            <Sparkline data={SPARK_IDX} className="h-40 w-full text-warn" />
            <div className="mt-4 flex gap-3 border-t border-line pt-3">
              <Button variant="pri" onClick={() => setModalOpen(false)}>Cerrar</Button>
              <Button onClick={pushKind('info', 'Acción de demo', 'Sin efecto en /ui-lab')}>Exportar</Button>
            </div>
          </Modal>
        </Section>

        <Section id="drawer" title="Drawer">
          <Button variant="pri" onClick={() => setDrawerOpen(true)}>Abrir drawer</Button>
          <Drawer
            open={drawerOpen}
            onClose={() => setDrawerOpen(false)}
            title="Av. Larco × Av. Benavides"
            status="bad"
            statusLabel="CRÍTICO"
          >
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-2">
                {[
                  { k: 'Cola', v: '142 m' },
                  { k: 'Demora', v: '96 s' },
                  { k: 'Flujo', v: '820 veh/h' },
                  { k: 'Ocupación', v: '87 %' },
                ].map((stat) => (
                  <div key={stat.k} className="rounded-[11px] border border-line bg-white/2 px-3 py-2.5">
                    <p className="text-[9px] font-bold uppercase tracking-[0.1em] text-ink-2">{stat.k}</p>
                    <p className="num mt-1 text-lg font-bold">{stat.v}</p>
                  </div>
                ))}
              </div>
              <ConfBar percent={94} />
              <div className="flex flex-wrap gap-2">
                <Button variant="pri" onClick={pushKind('ok', 'Plan aplicado', 'Extender N–S +12 s')}>Aplicar</Button>
                <Button>Simular</Button>
                <Button variant="done">✓ Aplicado</Button>
              </div>
              <p className="text-[12px] leading-relaxed text-ink-2">
                Contenido de relleno para validar el scroll interno del drawer en pantallas bajas.
                {' '}El detalle real de intersección llega en Fase 3.
              </p>
            </div>
          </Drawer>
        </Section>

        <Section id="mapa" title="Mapa (MapCanvas + mockGeo)">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <LayerChips
              layers={layers}
              onToggle={(id) => setLayers((curr) => curr.map((l) => (l.id === id ? { ...l, on: !l.on } : l)))}
            />
            <SegmentedControl
              ariaLabel="Modo del mapa"
              options={[
                { value: 'historic', label: 'Histórico' },
                { value: 'live', label: '● Ahora' },
                { value: 'prediction', label: 'Predicción' },
              ]}
              value={mapMode}
              onChange={setMapMode}
            />
          </div>
          <div className="flex h-[400px]">
            <MapCanvas
              center={MIRAFLORES_CENTER}
              zoom={14}
              legend={<MapLegend />}
              modeBadge={<MapModeBadge mode={mapMode} offsetMin={15} timestampLabel="hoy 16:00" />}
              coachTip={
                <span className="whitespace-nowrap rounded-full border border-brand/50 bg-brand/15 px-4 py-2 text-[11.5px] text-brand-2 backdrop-blur-md">
                  ✨ Probá: clic en el nodo rojo
                </span>
              }
            >
              {trafficOn && (
                <GeoJSON
                  data={MOCK_EDGES}
                  style={(feature) => jamLevelPathOptions(feature?.properties?.jam_level)}
                />
              )}
              {nodesOn && (
                <>
                  <NodeMarker position={[-12.1198, -77.0314]} status="ok" />
                  <NodeMarker position={[-12.1211, -77.0299]} status="warn" />
                  <NodeMarker
                    position={[-12.1284, -77.029]}
                    status="bad"
                    critical
                    onClick={() => setDrawerOpen(true)}
                  />
                </>
              )}
            </MapCanvas>
          </div>
          <p className="text-[11px] text-ink-2">
            Tramos por jam_level (mapping provisional de edgeStyle) y nodos ok/warn/bad — el rojo
            es crítico (halo) y abre el Drawer.
          </p>
        </Section>

        {/* FASE 2 — datos REALES del backend local (sin DemoBadge: ver intro
            de la sección). Verificación manual de la capa de hooks. */}
        <Section id="live" title="Datos en vivo (backend local)">
          <LiveDataSection />
        </Section>
      </div>
    </div>
  );
}

// FASE 3: el ToastProvider global vive en AppShell (el ui-lab se monta dentro
// del shell) — el wrapper local de Fase 1 quedó redundante y se eliminó.
export default function UiLabView() {
  return <LabContent />;
}
