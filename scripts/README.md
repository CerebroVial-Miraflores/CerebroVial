# scripts/

Utilidades operativas. Hay dos formas de correrlas contra la BD del compose:

**Desde host** (recomendado para scripts nuevos en branches sin rebuild de
imagen): con el venv del repo activo y `PYTHONPATH=core_management_api` cuando
el script importe módulos de `src/`:

```bash
PYTHONPATH=core_management_api .venv/bin/python scripts/<x>.py
```

Los scripts leen `DATABASE_URL` del `.env` y conmutan `@db:` a `@localhost:`
cuando detectan que NO están dentro del contenedor (vía la ausencia de
`/.dockerenv`).

**Desde el contenedor** (`docker compose exec` o `invoke shell-api`): el
binding-mount del compose monta solo `/app` con el contenido de
`core_management_api/`; el directorio `scripts/` del repo raíz **no está
montado**. Para que un script de esta carpeta esté visible dentro del
contenedor hay que rebuildear la imagen (`invoke up-build --service=core_management_api`)
o copiarlo a mano. Mientras no se haga, la opción "host" de arriba es la que
funciona.

## seed.py — datos iniciales de Miraflores

Idempotente (usa `session.merge()`). Inserta 5 nodos, 6 edges, 4 cámaras y 1
admin. Reaplicar tras cualquier `invoke down --volumes` que recree el volumen
de la DB:

```bash
invoke seed
```

## seed_rbac_smoke.py — usuarios de smoke RBAC (HU-01)

Inserta tres usuarios con roles distintos (`operator`, `manager`, `admin`)
para ejercitar a mano los flujos de HU-01.

## activate_decision.py — activador de estrategia vigente (HU-05)

Activa una decisión existente como estrategia vigente de un nodo, escribiendo
a `engine_active_state`. **No dispara SSE**.

```bash
docker compose exec core_management_api python scripts/activate_decision.py \
    --node-id larco_schell \
    --decision-id <uuid de motor_decisions> \
    --activated-by cli   # opcional, default "cli"
```

Validaciones (fail-fast con exit code distinto de cero):

- `--node-id` debe existir en `graph_nodes`.
- `--decision-id` debe existir en `motor_decisions`.
- El `node_id` de la decisión debe coincidir con `--node-id` (no se permite
  activar una decisión de un nodo como vigente de otro; rompería el audit
  trail).

### Por qué este CLI NO publica el evento SSE

El broadcaster del módulo `control` (`src/control/infrastructure/broadcaster.py`)
es un `dict[node_id, set[asyncio.Queue]]` **por proceso**. Los clientes SSE
suscritos viven en el proceso del `uvicorn`; este CLI corre en un proceso
Python distinto, incluso cuando se invoca con `docker compose exec` (que
lanza un proceso nuevo dentro del contenedor). Llamar a `broadcaster.publish`
desde acá escribiría a un broadcaster sin subscribers, mientras los SSE
reales seguirían esperando en el otro proceso.

El CLI cumple su rol: dejar la fila apropiada en BD para que el GET
`/control/active-state/{node_id}` la devuelva. Si la UI estaba abierta, una
recarga del panel mostrará el estado actualizado. **No es esperable que el
panel se actualice "en vivo" tras este CLI.**

### Para verificar SSE en vivo (CA-05.3)

Usar el endpoint dev-only `POST /control/__internal/activate`, que sí corre
intra-uvicorn y propaga el evento al broadcaster:

```bash
# El endpoint solo existe si el compose dev seteó ENABLE_TEST_ACTIVATOR=true.
curl -X POST http://localhost:8000/control/__internal/activate \
    -H "Authorization: Bearer <jwt de admin>" \
    -H "Content-Type: application/json" \
    -d '{"node_id":"larco_schell","decision_id":"<uuid>","activated_by":"manual"}'
```

El activador productivo (operador o motor disparándolo desde dentro de
uvicorn, sin gate de env var) es alcance de HU-07/futuro.

## cleanup_jira.py

Utilidad ad-hoc para limpiar issues en Jira de pruebas pasadas. No es parte
del flujo regular.
