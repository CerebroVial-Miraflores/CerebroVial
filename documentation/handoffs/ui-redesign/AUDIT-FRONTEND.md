# AUDITORÍA READ-ONLY DEL FRONTEND — CerebroVial

> Insumo para diseñar una migración incremental del frontend a un nuevo sistema de diseño
> (design tokens → componentes base mobile-first → vistas en ruta paralela `/v2`).
> Cada afirmación no trivial se etiqueta `[VERIFICADO]` (leído del repo) o `[INFERIDO]` (deducido).
> Read-only: no se modificó código. Generado 2026-06-09.

---

## 1. IDENTIFICACIÓN

- **Rama actual:** `feature/refundacion-vision` `[VERIFICADO]`
- **Último commit:** `c7b81c7e` — *feat(vision): detector + executor de inferencia compartidos en el scheduler (B1 2-A2 paso 2)* `[VERIFICADO]`
- **Working tree:** NO limpio — 1 archivo modificado: `documentation/handoffs/refundacion-vision/b1-2a-scheduler-handoff.md` (doc backend, NO frontend). `src/` del frontend está intacto. `[VERIFICADO]`
- **Paquete frontend:** `frontend_ui/` en la raíz del repo (`/Users/rasec/Tesis/CerebroVial/frontend_ui/`). Único `package.json` no-node_modules del repo. `name: "front"`. `[VERIFICADO]`

## 2. STACK (versiones del `package.json`; rangos `^`/`~`, no resueltos del lockfile)

| Dominio | Paquete | Versión declarada |
|---|---|---|
| Core | react / react-dom | `^19.2.0` |
| Build | vite | `^7.2.4` |
| Lenguaje | typescript | `~5.9.3` |
| **Estilos** | **tailwindcss** | **`^4.1.17` (v4)** |
| Estilos | @tailwindcss/vite | `^4.1.17` |
| Mapas | leaflet / react-leaflet | `^1.9.4` / `^5.0.0` |
| Mapas | @types/leaflet | `^1.9.21` |
| Router | react-router-dom | `^7.15.1` |
| HTTP | axios | `^1.13.2` |
| SSE | @microsoft/fetch-event-source | `^2.0.1` |
| Video | hls.js | `^1.6.16` |
| Iconos | lucide-react | `^0.556.0` |
| Charts | recharts | `^3.5.1` |
| Test | vitest / @vitest/coverage-v8 | `^4.0.15` |
| Test | @testing-library/react / jest-dom | `^16.3.0` / `^6.9.1` |
| Test env | jsdom | `^27.2.0` |

- **CRÍTICO Tailwind v4 CSS-first:** se usa el plugin `@tailwindcss/vite` en [vite.config.ts](frontend_ui/vite.config.ts) y `@import "tailwindcss";` en [src/index.css](frontend_ui/src/index.css). **NO existe `tailwind.config.js`/`.ts`** en `frontend_ui/`. **NO hay bloque `@theme`** en ningún `.css`. Config de Tailwind = defaults puros, sin tokens custom. `[VERIFICADO]`
- **Estado global:** NO hay Redux/Zustand/Jotai. Estado de sesión vía React Context (`SessionContext`); el resto es `useState` local por vista. `[VERIFICADO]`
- **Scripts npm:** `dev` (vite), `build` (vite build), `lint` (eslint .), `preview`, `test` (vitest run), `test:coverage`. `[VERIFICADO]`
- **Sin librería de fechas** (date-fns/dayjs/luxon): se usa `Date` nativo + `toLocaleString`/`toLocaleTimeString`. `[VERIFICADO]`

## 3. ESTRUCTURA (`src/`, profundidad 3, archivos por carpeta)

```
src                                  6 archivos (App.tsx, App.css, index.css, main.tsx, router.tsx, setupTests.ts)
├── __tests__                        1   (App.test.tsx)
├── assets                           1   (react.svg)
├── auth                            10  + __tests__ 8
├── components                       3   (CameraGrid, CameraStrip, HlsPlayer) + __tests__ 3
│   ├── layout                       2   (Sidebar, Header) + __tests__ 1
│   ├── modals                       1   (ThesisModal)
│   ├── ui                           3   (Card, Badge, LoadingStates)
│   ├── views                        6  + __tests__ 5
│   │   └── control                 13  + __tests__ 2
│   └── widgets                      1   (TrafficHistoryWidget) + __tests__ 1
├── services                         7  + __tests__ 4
├── tomtom                           5  (+README.md) + __tests__ 1
├── types                            3   (index, congestion, visionStream)
└── utils                            3   (congestion, markerVisual, trafficLabels) + __tests__ 3
```

- **LOC `src` EXCLUYENDO tests:** **6.521** líneas `[VERIFICADO]`
- **LOC de tests** (`__tests__` + `*.test.*` + `setupTests.ts`): **4.441** líneas `[VERIFICADO]`
- 29 archivos de test en total. `[VERIFICADO]`

## 4. VISTAS Y NAVEGACIÓN

- **Definición de rutas:** [src/router.tsx](frontend_ui/src/router.tsx) — `createBrowserRouter`. Solo 2 rutas reales: `/login` (público) y `*` (protegido). Toda la app navega por **estado interno** (`activeTab`), NO por URL. `[VERIFICADO]`
- **Árbol de router:** `SessionProvider` (raíz) → `[ /login=LoginView | ProtectedRoute → "*"=App ]`. `ProtectedRoute` ([src/auth/ProtectedRoute.tsx](frontend_ui/src/auth/ProtectedRoute.tsx)) redirige a `/login` si `!isAuthenticated`. `[VERIFICADO]`
- **El shell** ([src/App.tsx](frontend_ui/src/App.tsx)) monta `Sidebar` + `Header` + `<main>` y hace **render condicional por `activeTab`** envuelto en `RoleGate`. NO hay rutas anidadas por vista. `[VERIFICADO]`

**Tabla vista → tab → rol(es) → componentes principales** `[VERIFICADO]`

| Vista (tab) | `activeTab` | Rol(es) | Componente raíz |
|---|---|---|---|
| Monitoreo | `dashboard` | operator | `DashboardView` → (si hay cámara sel.) `CameraDetailView` |
| Alertas | `alerts` | operator | `AlertsView` (mock) |
| Motor Adaptativo | `control` | operator, admin | `ControlView` → `ActiveStrategyView` (operator) \| `ControlPlayground` (admin) |
| Mapa de congestión | `congestion` | operator | `CongestionMapView` |
| Tráfico en vivo | `tomtom` | operator | `TomTomView` (EXPERIMENTAL) |
| Analítica e IA | `analytics` | manager | `AnalyticsView` (mock, recharts) |
| Administración | `admin` | admin | `AdminView` (mock) |

- **RoleGate:** [src/auth/RoleGate.tsx](frontend_ui/src/auth/RoleGate.tsx) (23 LOC). Mecánica: lee `role` de `useSession()`; si `role && allowed.includes(role)` renderiza `children`, si no `fallback` (default `null`). Es defensa de **presentación**; el enforcement real es backend (`require_role`). `[VERIFICADO]`
- **Tabs por rol y default** — [src/auth/roles.ts](frontend_ui/src/auth/roles.ts): `TABS_BY_ROLE = { operator:[dashboard,control,alerts,congestion,tomtom], manager:[analytics], admin:[admin,control] }`. `DEFAULT_TAB_BY_ROLE = { operator:'dashboard', manager:'analytics', admin:'admin' }`. `ROLE_LABEL_ES` = Operador/Gerente/Administrador. `[VERIFICADO]`
- **Defensa en profundidad:** `App.setActiveTab` valida el tab contra `TABS_BY_ROLE[role]` y cae al default si no aplica. `[VERIFICADO]`

## 5. INVENTARIO DE COMPONENTES (LOC excluye tests)

**10 más grandes por LOC** `[VERIFICADO]`

| # | LOC | Archivo | ¿Reutilizable? |
|---|---|---|---|
| 1 | **950** | `views/CongestionMapView.tsx` | acoplado (vista monolítica) |
| 2 | **505** | `views/CameraDetailView.tsx` | acoplado |
| 3 | **455** | `views/DashboardView.tsx` | acoplado |
| 4 | **355** | `views/control/ControlPlayground.tsx` | acoplado |
| 5 | **312** | `widgets/TrafficHistoryWidget.tsx` | semi (recibe `cameraId`) |
| 6 | **245** | `views/control/ActiveStrategyView.tsx` | acoplado |
| 7 | **236** | `utils/congestion.ts` (no-componente: lógica de estilo/merge) | reutilizable |
| 8 | **228** | `views/control/TrafficLightCycle.tsx` | reutilizable (anim. semáforo) |
| 9 | **209** | `views/control/RecommendationPanel.tsx` | acoplado a control |
| 10 | **158** | `services/sseClient.ts` (no-componente) | reutilizable |

**Componentes "dios" (>300 LOC, múltiples responsabilidades):** `[VERIFICADO]`
- **`CongestionMapView` (950)** — 3 modos (`live`/`historic`/`prediction`) en un union inline; **25 `useState` + 11 `useEffect`**; maneja geometría, SSE, serie histórica (slider), predicción GRU (slider de horizonte), leyendas, tooltips/popups Leaflet. Es la deuda registrada **DEUDA-SWITCH-MODE** (ver §12).
- **`CameraDetailView` (505)** — contenedor (estado de cámara actual) + `CameraStrip` (carril lazy) + sub-componente `CameraDetailPanel` que orquesta alta on-demand del YOLO en el edge (`POST /cameras/{id}`), abre SSE de visión, calcula métricas y monta `TrafficHistoryWidget` + `predictionService`.
- **`DashboardView` (455)** — mapa Leaflet + markers (sub-comp `IntersectionMarker`, `MapUpdater`, `FitBounds`) + grilla de cámaras (`CameraGrid`) + **2 SSE simultáneos** (congestión de red + visión por cámara) + fila de KPIs hardcodeados.
- **`ControlPlayground` (355)** — editor de demanda por fase + presets + dispara `/control/recommend` y `/control/__internal/activate`.

**Componentes base reutilizables existentes (pocos):** `ui/Card` (7 LOC), `ui/Badge` (16), `ui/LoadingStates` (`LoadingOverlay`+`SkeletonCard`, 34), `HlsPlayer` (130), `CameraGrid`/`CameraStrip` (lazy HLS por viewport), `control/Slider` (36), `control/TimingBar` (51). `[VERIFICADO]`

- **Iconos:** `lucide-react` en casi todo. **Charts:** `recharts` solo en `AnalyticsView` y `TrafficHistoryWidget`. **HLS:** `hls.js` en `HlsPlayer` (consumido por `CameraGrid`/`CameraStrip`/`DashboardView`/`CameraDetailView`). `[VERIFICADO]`

## 6. SISTEMA DE ESTILOS ACTUAL

- **Dónde viven colores/espaciados:** **NO centralizados.** Sin `@theme`, sin `tailwind.config`, sin archivo de tokens. Todo es **utilidades Tailwind por defecto inline** (`bg-slate-950`, `text-indigo-400`, etc.) repetidas en cada componente. `[VERIFICADO]`
- **NO existe archivo de design tokens hoy.** `[VERIFICADO]`
- **CSS global:** solo 2 archivos.
  - [src/index.css](frontend_ui/src/index.css) (8 LOC): `@import "tailwindcss";` + regla `html,body { @apply bg-slate-950; margin/padding/height }`. `[VERIFICADO]`
  - [src/App.css](frontend_ui/src/App.css) (42 LOC): **residuo del scaffold de Vite** (`#root max-width:1280px`, `.logo`, `@keyframes logo-spin`, `.card`, `.read-the-docs`). **NO se importa en ningún lado** (`main.tsx` solo importa `index.css`) → CSS muerto. `[VERIFICADO]`
- **Clases utilitarias custom:** **NINGUNA** definida (no hay `@layer`/`@utility`/`.scrollbar-*` propias). El único `@apply` está en `index.css` (`bg-slate-950`). `[VERIFICADO]`
- **Clases de animación usadas pero NO definidas:** `animate-fade-in` y `animate-in` se usan en `LoadingStates`, `AnalyticsView`, `CameraDetailView`, `DashboardView` pero **no hay `@keyframes`/`@theme`/plugin que las defina** (no está `tailwindcss-animate`). Son **clases inertes (no-ops)**. `animate-spin`/`animate-pulse`/`animate-ping` sí son nativas de Tailwind y funcionan. `[VERIFICADO]`
- **Patrones repetidos frágiles:**
  - `h-[calc(100vh-7rem)]` en [TomTomView.tsx:40](frontend_ui/src/tomtom/TomTomView.tsx#L40) y [CongestionMapView.tsx:644](frontend_ui/src/components/views/CongestionMapView.tsx#L644) — alto de mapa atado a una altura de header fija (7rem). `[VERIFICADO]`
  - `z-[400]`/`z-[500]` (z-index arbitrarios sobre el mapa Leaflet) dispersos en `DashboardView`, `CongestionMapView`, `TomTomView`, `LoadingStates`. `[VERIFICADO]`
  - Patrón sidebar `w-20 md:w-64` + `ml-20 md:ml-64` (ancho del aside duplicado en el margen del `<main>`, acoplados a mano). `[VERIFICADO]`
  - **~17 colores hex hardcodeados** en `.tsx` (vía `style=`/recharts/box-shadow), p.ej. `#4f46e5`, `#10b981`, `#0A0A0A`, `#0f172a`, `#818cf8`, escala TomTom `#2ECC71/#F4C20D/#E24B4A/#8E1B1B`. No salen de ninguna paleta central. `[VERIFICADO]`
- **Dark theme:** **FIJO, sin toggle.** El shell aplica `bg-slate-950 text-slate-200` en `App.tsx` y `LoginView`; toda la paleta es slate/indigo oscura cableada. NO hay `ThemeProvider`, `dark:` variants, ni clase `dark` en `<html>`. `[VERIFICADO]`

## 7. CAPA DE DATOS

- **`httpClient`** ([src/services/httpClient.ts](frontend_ui/src/services/httpClient.ts), 42 LOC): instancia axios con `baseURL = VITE_CORE_API_URL ?? 'http://localhost:8001'`. **JWT:** interceptor de request inyecta `Authorization: Bearer <token>` leyendo `authBridge.getToken()`. **Errores:** interceptor de response — si `401` Y la request llevaba `Authorization`, dispara `authBridge.onUnauthorized()` (auto-logout). El 401 de `/auth/login` no dispara (no lleva header). `[VERIFICADO]`
- **`authBridge`** ([src/auth/authBridge.ts](frontend_ui/src/auth/authBridge.ts), 37 LOC): singleton fuera del árbol React; `SessionProvider` registra `getToken`/`onUnauthorized` al montar. Seam para axios y los SSE (que no pueden usar hooks). `[VERIFICADO]`
- **SSE — dos clientes basados en `@microsoft/fetch-event-source`** (pasan Bearer manual; `EventSource` nativo no admite headers):
  - [sseClient.ts](frontend_ui/src/services/sseClient.ts) (158) → `openControlActiveStateStream(nodeId)` → `GET /control/active-state/{nodeId}/stream`. Consumidor: **`ActiveStrategyView`** (HU-05). Filtra evento `active-state-changed`; backoff exp. 1→16s; 401 → `onUnauthorized` + corte. `[VERIFICADO]`
  - [congestionSseClient.ts](frontend_ui/src/services/congestionSseClient.ts) (132) → `openCongestionStream()` → `GET /congestion/state/stream`. Consumidores: **`CongestionMapView`** (HU-22) **y `DashboardView`** (recolorea markers). Wake `congestion-updated` sin payload → re-fetch. Duplicación deliberada de `sseClient` (declarada en cabecera). `[VERIFICADO]`
- **Tercer SSE — `EventSource` nativo** en `DashboardView` (visión por cámara, edge): `new EventSource(${VITE_EDGE_API_URL}/stream/{id})`, evento `traffic_update`. **Sin JWT** (EventSource nativo). `[VERIFICADO]`

**Servicios REST (un renglón c/u)** `[VERIFICADO]`

| Servicio | Endpoints (método + ruta) | Vía |
|---|---|---|
| `authService` | `POST /auth/login` | httpClient |
| `congestionService` | `GET /congestion/{geometry,state,series?day,prediction?t}` | httpClient |
| `controlService` | `POST /control/recommend`, `POST /control/__internal/activate` | httpClient |
| `controlActiveStateService` | `GET /control/active-state/{node_id}` | httpClient |
| `predictionService` | `POST /predictions/predict` | httpClient |

- **Llamadas `fetch()` crudas SIN httpClient (sin JWT):** `GET /api/intersections` (en `DashboardView` y `CameraDetailView`), `GET /predictions/history/{cameraId}?interval=` (en `TrafficHistoryWidget`), `POST /cameras/{id}` al edge (alta YOLO en `CameraDetailPanel`), y el `GET /stream/{id}` del edge. `[VERIFICADO]`

**Mapa endpoint backend → consumidor frontend** `[VERIFICADO]` (edge = puerto 8000; resto core = 8001)

| Endpoint | Consumidor | JWT |
|---|---|---|
| `POST /auth/login` | `authService` ← `SessionContext.login` ← `LoginView` | n/a |
| `GET /api/intersections` | `DashboardView`, `CameraDetailView` (fetch crudo) | NO |
| `GET /stream/{id}` (edge) | `DashboardView` (EventSource visión) | NO |
| `POST /cameras/{id}` (edge) | `CameraDetailPanel` (alta YOLO on-demand) | NO |
| `GET /predictions/history/{id}` | `TrafficHistoryWidget` (fetch crudo) | NO |
| `POST /predictions/predict` | `predictionService` ← `CameraDetailPanel` | sí |
| `GET /control/active-state/{node}` | `controlActiveStateService` ← `ActiveStrategyView` | sí |
| `GET /control/active-state/{node}/stream` | `sseClient` ← `ActiveStrategyView` | sí (Bearer manual) |
| `POST /control/recommend` | `controlService` ← `ControlPlayground` | sí |
| `POST /control/__internal/activate` | `controlService` ← `ControlPlayground` | sí |
| `GET /congestion/geometry,state,series,prediction` | `congestionService` ← `CongestionMapView` | sí |
| `GET /congestion/state/stream` | `congestionSseClient` ← `CongestionMapView`, `DashboardView` | sí (Bearer manual) |

## 8. MAPA LEAFLET

- **Componentes que montan `MapContainer`:** `DashboardView`, `CongestionMapView`, `TomTomView`. Cada uno tiene su **propio** `MapContainer` (no comparten instancia; declarado explícito en `TomTomView`). `[VERIFICADO]`
- **`TomTomFlowLayer`** monta un `TileLayer` extra (raster TomTom) dentro del map de `TomTomView`. `[VERIFICADO]`
- **Capas/primitivas usadas:** `[VERIFICADO]`
  - `DashboardView`: `TileLayer` (OSM) + `Marker` (divIcon custom) + `Tooltip` + `Popup`; helpers `useMap` (`MapUpdater` con `flyTo`, `FitBounds` con `fitBounds`). Tiene además un modo `waze` que es un `<iframe>` a `embed.waze.com` (no Leaflet).
  - `CongestionMapView`: `TileLayer` (OSM) + **`GeoJSON`** (1660 tramos coloreados por `style` callback; decisión firme `GeoJSON` sobre `Polyline` porque el backend da `[lon,lat]`). Recolorización vía `key` que remonta la capa. `bindTooltip`/`bindPopup` por feature.
  - `TomTomView`: `TileLayer` OSM base + `TileLayer` TomTom raster.
- **Markers custom:** `DashboardView` usa `L.divIcon` con HTML+clases Tailwind: punto coloreado por congestión (Waze) + `animate-ping` (pulso = cámara transmite/desconocido) + tachado diagonal (offline). Lógica en [utils/markerVisual.ts](frontend_ui/src/utils/markerVisual.ts): `markerVisual(status, health)` separa **color (congestión)** de **pulso/tachado (salud cámara)**. También fija `L.Marker.prototype.options.icon` con los PNG default de Leaflet importados. `[VERIFICADO]`
- **"Miniplayer HLS sobre markers":** **NO existe un miniplayer de video embebido en el marker.** El popup del marker (`IntersectionMarker`) muestra solo texto (velocidad/flujo/estado + botón "Ver Cámara"). El video HLS vive en componentes separados: `CameraGrid` (grilla, sub-tab "Cámaras" del dashboard) y `CameraStrip` (carril horizontal en `CameraDetailView`), ambos con `HlsPlayer` montado **lazy por `IntersectionObserver`** (solo reproduce en viewport; al salir desmonta y hace `hls.destroy()`). El click en el marker/celda navega al `CameraDetailView`. `[VERIFICADO]`
- **Mock de `react-leaflet` en tests:** `vi.mock('react-leaflet')` hoisted devolviendo `<div>` con `data-testid` (NO renderiza Leaflet real). Presente en tests de `CongestionMapView` (×3: base, historic, prediction), `DashboardView.sse`, `TomTomView`. `[VERIFICADO]`

## 9. ESTADO EN MEMORIA

- **`SessionContext`** ([src/auth/SessionContext.tsx](frontend_ui/src/auth/SessionContext.tsx), 129 LOC): **único Context global.** Guarda `{ token, role, userId, isAuthenticated }`; expone `login`/`logout`. Hidrata desde `tokenStorage` (localStorage), decodifica el JWT (`decodeJwtPayload`), valida `exp`. Registra los callbacks del `authBridge`. Consumidores: `App`, `Sidebar`, `RoleGate`, `ControlView`, `ProtectedRoute`, `LoginView`. `[VERIFICADO]`
- **NO hay otros stores/contexts.** Todo el demás estado es `useState` local de cada vista (no compartido entre vistas, salvo `selectedCamera`/`activeTab` que viven en `App.tsx` y se pasan por props). `[VERIFICADO]`
- **`tokenStorage`** (23 LOC): wrapper de `localStorage` para el token. `[VERIFICADO]`

## 10. TESTS

- **Setup global vitest:** [vite.config.ts](frontend_ui/src/../vite.config.ts) → `test: { globals:true, environment:'jsdom', setupFiles:'./src/setupTests.ts' }`. [src/setupTests.ts](frontend_ui/src/setupTests.ts) = solo `import '@testing-library/jest-dom'`. `[VERIFICADO]`
- **Archivos de test por carpeta (29 total):** auth 8, services 4, views 5, views/control 2, components 3, layout 1, widgets 1, tomtom 1, utils 3, src raíz 1. `[VERIFICADO]`
- **Stubs/mocks relevantes:** `vi.mock('react-leaflet')` (5 archivos); `IntersectionObserver` stub en tests de `CameraGrid`/`CameraStrip`/`CameraDetailView`; `EventSource` mock en `DashboardView.sse`, `CameraDetailView`, `ActiveStrategyView.transport_pings`, `sseClient`; `vi.mock('hls.js')` en `HlsPlayer`/`CameraDetailView`. `[VERIFICADO]`
- **¿Pasa la suite hoy?** `npx vitest run` → **228/229 tests pasan; 1 falla** (28/29 archivos). El fallo es `tomtom/__tests__/TomTomView.test.tsx` "degrada sin VITE_TOMTOM_KEY": el entorno local **tiene `VITE_TOMTOM_KEY` no-vacía en `frontend_ui/.env`**, así que la vista NO muestra el aviso esperado. **Fallo dependiente del entorno, no del código** (con key vacía pasaría). `[VERIFICADO]`

## 11. RESPONSIVE HOY

- **Conteo de prefijos en `.tsx` (excl. tests):** `sm:` 2, `md:` 16, `lg:` 8, `xl:` 0, `2xl:` 0. `[VERIFICADO]`
- **Layout para móvil:** el sidebar colapsa a `w-20` (solo iconos) en <md y se expande a `w-64` en `md:` — único patrón responsive deliberado. Las grillas usan `grid-cols-1 md:grid-cols-2/4` y `sm:grid-cols-2 lg:grid-cols-3` (CameraGrid). NO hay menú hamburguesa, drawer móvil, ni breakpoints `xl:`. El sidebar es **fijo** (`fixed`), nunca se oculta. `[VERIFICADO]`
- **Media queries custom:** solo 2 en `App.css` (CSS muerto del scaffold: `prefers-reduced-motion` y nada de la app real). `[VERIFICADO]`
- **Veredicto (1 línea):** la app es **desktop-first con responsive mínimo y reactivo** (26 prefijos totales, 0 en `xl/2xl`, sin drawer móvil, alturas de mapa atadas a `calc(100vh-7rem)`); **no hay diseño mobile-first**. `[INFERIDO]` (sobre el conteo verificado)

## 12. DEUDAS UI EN CÓDIGO

`grep DEUDA|TODO|FIXME|HACK` en `src` (excl. tests): `[VERIFICADO]`

- `views/CameraDetailView.tsx:435` y `:438` — **DEUDA-SPEED-CALIB**: velocidad experimental sin calibrar (badge/tooltip); el conteo de vehículos sí es real.
- `views/control/controlTypes.ts:63` — `TODO(HU-05+)`: reemplazar `KNOWN_NODE_IDS` (lista estática de 5 nodos) por un `GET` a endpoint dedicado de `graph_nodes`.
- `views/control/ControlPlayground.tsx:293` y `services/controlService.ts:80` — `TODO(HU-07)`: el disparador `/control/__internal/activate` es TEMPORAL (demo DHU-020), gateado por `ENABLE_TEST_ACTIVATOR`.
- **DEUDA-SWITCH-MODE** — NO está en `src`; vive en [documentation/docs/TODO.md:178](documentation/docs/TODO.md#L178): el switch de modos de `CongestionMapView.tsx` (`live`/`historic`/`prediction`) es un union inline en `useState` + patrón disperso (7 `useEffect` con guard `if (mode !== X) return`); candidato a extraer a `type Mode` + componente de control reutilizable. `[VERIFICADO]`

## 13. ASSETS Y BRANDING

- **Logo:** no hay archivo de logo propio. La marca "CerebroVial" se renderiza como texto + icono `Cpu` de lucide en `Sidebar` y `LoginView`. `[VERIFICADO]`
- **Favicon:** `index.html` referencia `/vite.svg` — **pero `public/` solo contiene `vite.svg`** (el favicon default de Vite, no uno de marca). `src/assets/react.svg` (logo React del scaffold) no se usa. `[VERIFICADO]`
- **Fuentes:** NO se cargan fuentes custom (sin `<link>` a Google Fonts, sin `@font-face`). Se usa `font-sans`/`font-mono` (stack default del SO vía Tailwind). `[VERIFICADO]`
- **Paleta:** NO centralizada (ver §6). De facto: fondo `slate-950/900`, acento `indigo-600/400`, semáforo de estado `emerald`/`amber`/`red`. `[VERIFICADO]`
- **`<title>` del documento:** `"front"` (placeholder del scaffold). `[VERIFICADO]`
- **`index.html`** inyecta `window.process = { env: {} }` (shim para libs que leen `process.env` en browser). `[VERIFICADO]`

## 14. RIESGOS PARA MIGRACIÓN INCREMENTAL (solo hechos)

1. **Sin tokens ni `@theme`:** Tailwind v4 con config default puro. No hay capa de tokens donde un design system pueda engancharse; los colores/espaciados están inline y duplicados en ~25 archivos. `[VERIFICADO]`
2. **Navegación por estado, no por URL:** una sola ruta `*` → `App` que hace render condicional por `activeTab`. Una ruta paralela `/v2` requiere agregar entradas en `router.tsx`, pero las vistas v1 no son direccionables por URL (no hay `/dashboard`, `/congestion`…). `[VERIFICADO]`
3. **Componentes "dios":** `CongestionMapView` (950, 25 `useState`/11 `useEffect`/3 modos), `CameraDetailView` (505), `DashboardView` (455), `ControlPlayground` (355) — mezclan data-fetching, SSE, estado de UI y presentación; reescritura no es extracción trivial. `[VERIFICADO]`
4. **CSS global que afecta todo:** `index.css` aplica `bg-slate-950` a `html,body` con `@apply` — cualquier `/v2` con otro fondo hereda esto salvo override. `App.css` es CSS muerto (no importado) pero presente. `[VERIFICADO]`
5. **Dark fijo cableado:** sin `ThemeProvider` ni variantes `dark:`; el tema es literal en cada className. Soportar light/temas en `/v2` no tiene punto de inyección hoy. `[VERIFICADO]`
6. **Heterogeneidad de la capa de datos:** conviven `httpClient` (axios+JWT), 2 SSE `fetch-event-source` (Bearer manual), `EventSource` nativo (edge, sin JWT) y **4 `fetch()` crudos sin JWT** (`/api/intersections`, `/predictions/history`, `/cameras/{id}`, `/stream/{id}`). Un rediseño de capa de datos debe contemplar las 4 vías. `[VERIFICADO]`
7. **Tres `MapContainer` independientes** (Dashboard, Congestion, TomTom) + un `<iframe>` Waze: no hay componente de mapa compartido para reusar en `/v2`. `[VERIFICADO]`
8. **Clases de animación inertes** (`animate-fade-in`/`animate-in` sin definición): migrar a un sistema que SÍ las defina cambiará apariencia (hoy no animan). `[VERIFICADO]`
9. **Acoplamiento de altura a header:** `h-[calc(100vh-7rem)]` asume header de 7rem; cambiar el chrome en `/v2` rompe el alto de los mapas. `[VERIFICADO]`
10. **Imports circulares:** no se detectaron en la lectura (los `import` de `views/control/*` y `services/*` son unidireccionales: views→services→authBridge; types hoja). `[INFERIDO]` (no se corrió un detector de ciclos como `madge`).
11. **`ThesisModal` + su botón en `Sidebar`** están marcados como zona protegida por `CLAUDE.md` (no remover en migración). `[VERIFICADO]`

## 15. PREGUNTAS ABIERTAS

1. **Resolución exacta de versiones:** se reportan los rangos `^`/`~` del `package.json`; no se leyó `package-lock.json` (230 KB) para fijar las versiones instaladas exactas. `[NO VERIFICADO]`
2. **Imports circulares:** descartados por lectura manual, no por herramienta (`madge`/`eslint-plugin-import`). Confianza media. `[NO VERIFICADO]`
3. **Cobertura real de los endpoints:** el mapa endpoint→consumidor se construyó leyendo el frontend; no se cruzó contra el router del backend (`core_management_api`) para detectar endpoints expuestos sin consumidor UI o viceversa. `[NO VERIFICADO]`
4. **`GET /api/health` con `require_role(ADMIN)`:** `CLAUDE.md` lo cita como único endpoint con enforcement de muestra; **no encontré consumidor en `src`** (consistente con "la UI no lo consume en su flujo natural"). `[INFERIDO]`
5. **Comportamiento del lint (`npm run lint`):** no se ejecutó ESLint; solo se corrió la suite de vitest. Estado de lint desconocido. `[NO VERIFICADO]`
6. **Contenido de `dist/` y `coverage/`:** existen en `frontend_ui/` (builds/reportes previos); no se auditaron por ser artefactos generados. `[NO VERIFICADO]`
7. **`B1`/`B2`/`C1`/`F1` en comentarios:** son fases de un sprint de visión/dashboard previo; no se rastreó su documentación para confirmar alcance. `[NO VERIFICADO]`

---
*Fin del reporte. Read-only: sin commits, sin cambios al código.*
