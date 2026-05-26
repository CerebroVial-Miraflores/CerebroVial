# Quickstart — `001-cerebrovial-mvp`

> **Adopción brownfield (DHU-021).** Este quickstart **mapea** el flujo de
> arranque ya documentado en `README.md` y `CLAUDE.md` al artefacto Spec Kit
> correspondiente; no introduce un flujo nuevo. La fuente operativa canónica
> sigue siendo `invoke` (gestor de tareas), `docker compose` y los manifiestos
> del repo.

## Propósito

Permitir a un desarrollador (o asesor) levantar localmente el sistema
correspondiente a esta feature y validar manualmente el estado de los CAs
que ya están construidos. **No es un test automatizado** — para cobertura
automatizada ver `invoke test`.

## Prerrequisitos

Verificados contra `README.md` raíz y `CLAUDE.md` de la feature:

- **Docker Desktop / Docker Engine** con Docker Compose v2 (`docker compose`).
- **Python 3.11** (para `invoke` y el venv local de desarrollo).
- **Git LFS** instalado y activo (`git lfs install`) — sin LFS los modelos
  binarios (`.pt`, `.ckpt`, `.h5`, `.joblib`) llegan como punteros y el
  servicio core falla al cargarlos.
- **Node 20+** *solo* si vas a tocar `frontend_ui/` fuera de Docker (Vite dev
  server).

## Setup primer-uso

Replica de `README.md` §"TL;DR":

```bash
cp .env.example .env
# Editar .env y reemplazar todos los "changeme"
pip install invoke
invoke setup-dev     # crea venv local con dependencias de dev
invoke up            # las tablas se crean solas (alembic upgrade head en el entrypoint)
invoke seed          # carga datos iniciales de Miraflores
```

Endpoints expuestos:

- Frontend SPA: <http://localhost:5173>
- API docs (OpenAPI): <http://localhost:8001/docs>
- DB (PostgreSQL + TimescaleDB + PostGIS): localhost:5432 (credenciales en `.env`)

## Día a día

Comandos de `invoke` relevantes (la lista completa con `invoke --list`):

| Comando | Cuándo |
|---|---|
| `invoke up` / `invoke down` | Levantar / bajar el stack |
| `invoke up-dev` | Hot-reload del core (usa `docker-compose.dev.yml`; alembic NO corre automático) |
| `invoke migrate` | Tras `up-dev` o tras un `git pull` con migraciones nuevas |
| `invoke up-build [--service=<nombre>]` | Rebuild de imágenes (frontend o cambios de Dockerfile/requirements) |
| `invoke logs` / `invoke ps` | Diagnóstico operativo |
| `invoke test` | Suite de tests (pytest backend + Vitest frontend) |
| `invoke shell-api` / `invoke shell-db` | Debugging interactivo |
| `invoke db-reset` | Resetear schema cuando hay cambios incompatibles (destructivo) |

> **NO usar `docker compose` directo.** `invoke` agrega validaciones (chequeo de
> LFS, `.env` presente) que evitan errores crípticos. Regla heredada de `CLAUDE.md`.

## Estado vivo de la feature (paseo manual)

El **~25%** del backlog está construido al 2026-05-20 (ver `tasks.md` §"Tabla
maestra"). Después de `invoke up && invoke seed`, lo que se puede tocar hoy:

1. **TTH-02 ✓ Docker** — el simple hecho de que `invoke up` haya levantado los
   cuatro servicios (`core`, `edge`, `frontend`, `db`) demuestra TTH-02. Verificar
   con `invoke ps`.
2. **TTH-08 parcial · Visión** — `edge_device/` corre YOLO11n sobre el video
   sintético; la salida se escribe a CSV (no a `vision_aggregates`; ver Delta-05 /
   SAN-03 en `tasks.md`). Para tocarlo: `invoke logs --service=edge`.
3. **TTH-10 parcial · Motor adaptativo** — endpoint `POST /control/recommend`
   está vivo (Webster + MaxPressure + capa MTC). Probarlo desde `/docs`. El
   motor NO persiste decisiones aún (Delta-10, pendiente del Sprint 4 item #3).
4. **HU-05 parcial · ControlView (vista pasiva)** — `frontend_ui/` muestra hoy
   el playground interactivo descrito en Delta-08. **DHU-020** dejó claro que
   prevalece la semántica pasiva: el refactor a vista pasiva es el item #4 del
   Sprint 4 (3 SP). El playground se preservará como herramienta de
   Administrador, no se elimina.

Lo demás (HU-01 RBAC, HU-02 monitoreo, HU-03 predicción GRU, HU-10 alerta
transversal, HU-13/14/15 admin, HU-16/17 reportería gerente) está en Trabajos
Futuros — ver `REPORTE_PLANIFICACION_SPRINT_4.md` §7.

## Checklist de validación

Ejecutar en orden tras `invoke up && invoke seed`:

- [ ] `invoke ps` muestra `core`, `edge`, `frontend`, `db` con estado healthy.
- [ ] `curl http://localhost:8001/health` responde `200 OK`.
- [ ] `GET http://localhost:8001/docs` abre la consola OpenAPI con los endpoints
      de `predictions/`, `control/` y `intersections/` listados.
- [ ] `POST /control/recommend` con un payload de fases válido devuelve una
      estrategia (`webster` o `max_pressure`) y un `reasoning` no vacío.
- [ ] `http://localhost:5173` carga el SPA y permite seleccionar una intersección.
- [ ] `invoke test` corre la suite de tests sin fallos. *(El job CI completo
      tiene gaps documentados en Delta-03. SAN-01 cerrada el 2026-05-26 en `san-06`.)*

## Solución de problemas

| Síntoma | Causa común | Fix |
|---|---|---|
| `core` falla con `RuntimeError: pickle ...` | LFS no instalado / punteros sin descargar | `git lfs install && git lfs pull` |
| `alembic` no aplica migraciones en dev | `invoke up-dev` salta el entrypoint | `invoke migrate` a mano |
| Schema rompe tras `git pull` | Migraciones incompatibles | `invoke db-reset` (destructivo) → `invoke up` → `invoke seed` |
| Frontend muestra "Asistente CerebroVial" / Reporte IA | Componentes huérfanos Delta-13 (SAN-02) | Saneamiento diferido; ver Art. 21 de constitution |
| `core` no levanta por dependencia `torch` | Si aparece tras un `git pull` de una rama antigua: torch fue removido de `core_management_api/requirements.txt` en `san-06` (2026-05-26, cierre SAN-01). | Rebuild con `invoke up-build --service=core_management_api` |

## Referencias canónicas

- `README.md` raíz — TL;DR operativo (fuente de este quickstart).
- `CLAUDE.md` — reglas de oro (NO refactor visión, NO `torch` en core,
  Alembic obligatorio).
- `.specify/memory/constitution.md` — los 22 artículos vinculantes del
  proyecto.
- `documentation/sdd/SDD_CEREBROVIAL.md` — arquitectura objetivo verificada.
- `documentation/lean-inception/planificacion/REPORTE_PLANIFICACION_SPRINT_4.md`
  — alcance ejecutable comprometido y Trabajos Futuros.
- `specs/001-cerebrovial-mvp/{spec,plan,tasks,data-model}.md` — artefactos
  Spec Kit hermanos de este quickstart.
