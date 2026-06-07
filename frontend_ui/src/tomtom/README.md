# Track `tomtom/` — capa de tráfico en vivo (TomTom Traffic Flow)

> **EXPERIMENTAL · Fase A — solo capa visual raster.**
> Vista nueva y aislada que muestra el tráfico en vivo de TomTom Traffic Flow
> (tiles raster) sobre un basemap OSM propio. Es **validación visual: solo se mira.**
> No mide, no consulta `flowSegmentData`, no calcula KPIs. Front puro: **cero
> backend, cero persistencia.**

## Alcance de Fase A
- `TomTomView.tsx` — vista autónoma con su propio `MapContainer` (OSM + Flow + atribución).
- `TomTomFlowLayer.tsx` — `<TileLayer>` raster del Traffic Flow (API v4, estilo `relative0-dark`).
- `TomTomAttribution.tsx` — atribución "© TomTom" visible y no removible.
- `types.ts` — tipos locales del track.

Registro (5 toques fuera de la carpeta, reversibles):
`auth/roles.ts` (tab + rol operator), `Sidebar.tsx` (ítem + icono Navigation),
`App.tsx` (render bajo `RoleGate` operator), `Header.tsx` (título), `.env.example`
(`VITE_TOMTOM_KEY`).

## Invariantes legales NO NEGOCIABLES (ToS de TomTom)
- **No cachear/reenviar tiles desde servidor** (cláusula 11.4): los tiles son
  *Results*. Los pide el **navegador directo** a `api.tomtom.com`; el backend NO
  interviene en absoluto en este track (prohíbe el fan-out servidor→N clientes).
- **No extraer datos de los tiles** (11.6.1): nada de parsear PNG ni leer píxeles.
  Los tiles son **solo para pintar**.
- **Atribución TomTom obligatoria y visible** (cláusula 17.3): no es opcional ni
  cosmético. Vive en `TomTomAttribution.tsx` y no debe ocultarse ni condicionarse.
- **TomTom NO alimenta ningún training** (STGNN/GRU) ni dataset persistido
  (11.6.3 / 11.6.4).
- **`VITE_TOMTOM_KEY` es una display key**, protegida por domain-whitelist + QPS
  en el dashboard de TomTom — **NO un secret de servidor**. Va en el bundle a
  propósito; su protección es la whitelist + límite de QPS, no el ocultamiento.
  Nunca se hardcodea: si falta, la capa degrada con gracia (solo OSM).

## Fase B (NO implementada — futura)
Corredores definidos por el operador + KPIs (índice de congestión, demora, nivel
categórico, agregación `max()` por corredor multi-segmento) vía `flowSegmentData`
(no-tile, on-demand). Eso SÍ requeriría un proxy backend que oculte la key del
endpoint de segmentos y cálculo de KPIs en memoria (sin persistir valores de
TomTom). La Copyrights API de TomTom (texto de atribución exacto por área) también
se difiere a Fase B. **Nada de eso existe en Fase A.**
