# CerebroVial

Sistema predictivo de tráfico para Miraflores (Lima, Perú). Proyecto de tesis.

Detección por visión computacional + predicción + control adaptativo de
semáforos. Para el alcance funcional ver [`documentation/tesis/`](documentation/tesis/)
y [`documentation/docs/PLAN.md`](documentation/docs/PLAN.md).

---

## TL;DR

```bash
# 1. Clonar (con git-lfs ya instalado)
git clone https://github.com/AndresBR2003/CerebroVial.git && cd CerebroVial

# 2. Configurar variables
cp .env.example .env
# (editá .env y reemplazá "changeme")

# 3. Instalar invoke + crear venv local
pip install invoke
invoke setup-dev

# 4. Levantar sistema y cargar datos iniciales
invoke up
invoke seed
```

Después: http://localhost:5173 (frontend), http://localhost:8001/docs (API).

---

## Arquitectura

Monolito modular dockerizado, **4 servicios**:

| Servicio | Build context | Puerto host | Depende de |
|---|---|---|---|
| `db` | TimescaleDB image | 5432 | — |
| `core_management_api` | [`core_management_api/Dockerfile`](core_management_api/Dockerfile) | 8001 | `db` (healthy) |
| `edge_device` | [`edge_device/Dockerfile`](edge_device/Dockerfile) | 8000 | `db` (healthy) |
| `frontend` | [`frontend_ui/Dockerfile`](frontend_ui/Dockerfile) | 5173 | `core_management_api` |

Las migraciones de schema viven en [`core_management_api/alembic/versions/`](core_management_api/alembic/versions/)
y se aplican automáticamente cuando arranca el contenedor del core (su entrypoint
ejecuta `alembic upgrade head` antes de levantar uvicorn).

---

## Prerequisitos del sistema

| Herramienta | Versión | Para qué |
|---|---|---|
| Docker Desktop | 4.x | Levantar todos los servicios |
| Git LFS | 3.x | Descargar binarios (modelos, datasets) |
| Python | 3.11 | Venv local para tests, alembic, seed |
| Node.js | 18+ | Frontend (`npm run dev` opcional) |

### Instalación

**macOS:**
```bash
brew install --cask docker
brew install git-lfs python@3.11 node
git lfs install
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install docker.io docker-compose-plugin git-lfs python3.11 nodejs npm
git lfs install
```

**Windows:**
- Docker Desktop: https://docker.com/products/docker-desktop
- Git LFS: https://git-lfs.com (descargar instalador)
- Python 3.11: https://python.org/downloads
- Node.js: https://nodejs.org
- Después de instalar git-lfs: `git lfs install`

### Verificación rápida

```bash
docker info              # debe responder sin error
git lfs --version        # >= 3.x
python3.11 --version     # 3.11.x
node --version           # >= v18
```

---

## Setup primer-uso (paso a paso)

### 1. Clonar el repo

```bash
git clone https://github.com/AndresBR2003/CerebroVial.git
cd CerebroVial
```

**Importante con Git LFS:** el repo guarda los binarios pesados (modelos
`.joblib`/`.pt`, datasets `.h5`/`.npy`, checkpoints `.ckpt`) en Git LFS. Si
hiciste `git lfs install` antes de clonar, los binarios reales bajan
automáticamente. Si NO, los archivos quedan como **pointers de texto** (~130
bytes que dicen "este archivo está en LFS"), y el sistema crashea al cargar
modelos con `UnpicklingError`.

#### Verificar si LFS bajó los binarios

```bash
git lfs ls-files | head           # debe listar varios .joblib / .pt / .h5
file core_management_api/models/traffic_rf_class_current.joblib
# OK:    "data" (binario real, varios MB)
# Roto:  "ASCII text" (es un pointer)
```

#### ¿Cloné sin git-lfs? Arreglalo así (no hace falta re-clonar)

```bash
# 1. Instalar git-lfs si todavía no está (ver "Instalación" arriba)
brew install git-lfs                # macOS
# o:  apt install git-lfs           # Linux
# o:  https://git-lfs.com           # Windows

# 2. Activar los hooks de LFS para tu usuario (una sola vez por máquina)
git lfs install

# 3. Bajar los binarios reales del repo ya clonado
git lfs pull
```

Después de `git lfs pull`, verificá con `file ...joblib` que ya no diga
"ASCII text". `invoke up` chequea esto automáticamente con
[`check_lfs`](tasks.py) y aborta con un mensaje claro si faltan binarios.

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

Editá `.env` y reemplazá `changeme` por una contraseña real. Las variables más
importantes:

- `POSTGRES_PASSWORD` y `DATABASE_URL`: contraseña de la DB local. Cualquier
  string sirve para desarrollo, pero ambas tienen que coincidir.
- `VITE_API_BASE_URL`: URL del core_management_api desde el navegador.
  El default `http://localhost:8001` está bien para desarrollo local.

Ver comentarios dentro de [`.env.example`](.env.example) para el detalle.

### 3. Instalar `invoke` (gestor de tareas)

`invoke` es el wrapper de comandos del proyecto. Mejor instalarlo en el venv
del proyecto, pero si lo querés global también funciona:

```bash
pip install invoke
```

### 4. Crear venv local con dependencias de dev

```bash
invoke setup-dev
```

Esto crea `.venv/` (core+visión, **numpy≥2**) con Python 3.11 e instala el paquete
`cerebrovial_shared` en modo editable + las deps de los módulos backend/visión +
pytest. Necesario para: correr tests, correr `seed`, generar migraciones Alembic locales.

```bash
source .venv/bin/activate
```

> **Dos venvs separados — cuál para qué.** El entrenamiento del GRU (`ia_prediction_service`)
> usa **tsl**, que ancla `numpy<2` y choca con opencv/visión (que va a numpy≥2). Por eso
> hay dos entornos aislados:
> - **`.venv`** (raíz, `invoke setup-dev`) — core + visión, numpy≥2. Para correr el sistema,
>   tests, seed, migraciones.
> - **`ia_prediction_service/.venv`** (`invoke setup-train`) — entrenamiento, numpy 1.x + tsl.
>   Para dataset builders y scripts de train. Ejecutar con
>   `ia_prediction_service/.venv/bin/python ia_prediction_service/scripts/…`.
>
> Detalle de la deuda y su resolución: `documentation/ESTADO_Y_PROXIMOS_PASOS.md`.

### 5. Levantar el sistema

```bash
invoke up
```

Esto arranca los 4 servicios. El core_management_api corre `alembic upgrade head`
en su entrypoint, así que las tablas se crean solas la primera vez. Tarda
~30-60s la primera vez (build de imágenes).

### 6. Cargar datos iniciales

```bash
invoke seed
```

Crea los 5 nodos de Miraflores, 6 aristas, 4 cámaras y 1 usuario admin.
Es **idempotente** — podés correrlo varias veces sin duplicar filas.

### 7. Verificar

- http://localhost:5173 → frontend
- http://localhost:8001/api/health → core (debe devolver 200)
- http://localhost:8001/docs → Swagger del core
- http://localhost:8000/docs → Swagger del edge_device

---

## Día a día

Todos los comandos se invocan con `invoke <comando>`. Lista completa:
`invoke --list`.

### Comandos básicos

| Comando | Qué hace |
|---|---|
| `invoke up` | Levantar el sistema completo |
| `invoke up --service=core_management_api` | Levantar un solo servicio |
| `invoke down` | Apagar (conserva datos de DB) |
| `invoke logs` | Ver logs de todos los servicios |
| `invoke logs --service=core_management_api` | Filtrar logs por servicio |
| `invoke ps` | Estado de los containers |
| `invoke health` | curl al endpoint de health del core |

### Migraciones y datos

| Comando | Qué hace |
|---|---|
| `invoke migrate` | Aplicar migraciones nuevas (sin rebuild) |
| `invoke migrate-create -m "mensaje"` | Generar migración con autogenerate |
| `invoke seed` | Cargar/refrescar datos de Miraflores |
| `invoke db-reset` | ⚠ Borrar DB y volver a migrar+sembrar |

### Acceso interactivo

| Comando | Qué hace |
|---|---|
| `invoke shell-api` | Bash dentro del container del core |
| `invoke shell-db` | Abrir `psql` en la DB |

### Builds

| Comando | Qué hace |
|---|---|
| `invoke up-build` | Rebuild de **todas** las imágenes (con cache de capas) y levantar |
| `invoke up-build --service=<nombre>` | Rebuild de **un solo** servicio (con cache) y levantar |
| `invoke rebuild` | Build sin cache de **todos** los servicios, conserva volúmenes |
| `invoke rebuild-clean` | ⚠ Build nuclear, borra DB |

`up-build` usa cache de capas, así que `npm install` / `pip install` no
re-descargan dependencias mientras no cambies `package*.json` /
`requirements.txt`.

Rebuild por servicio:

```bash
invoke up-build --service=frontend              # SPA Vite + nginx (build estático)
invoke up-build --service=core_management_api   # FastAPI + alembic
invoke up-build --service=edge_device           # visión (YOLO + tracking + SSE)
invoke up-build --service=db                    # imagen de TimescaleDB (rara vez hace falta)
invoke up-build                                 # rebuildea todos
```

Casos típicos:

- Tocaste código del **frontend** (que se sirve como build estático con nginx,
  no tiene HMR en el container): `invoke up-build --service=frontend`.
- Cambiaste el `Dockerfile` o `requirements.txt` de un servicio:
  `invoke up-build --service=<ese>`.
- Cambiaste algo que afecta a varios servicios: `invoke up-build` (todos).

> Para `core_management_api` y `edge_device`, si solo cambiaste `.py` (no
> Dockerfile ni requirements), conviene `invoke up-dev` (hot-reload) en vez
> de rebuildear. Para iterar el frontend, `npm run dev` con HMR es más rápido
> que rebuildear la imagen. Ver [Modo desarrollo](#modo-desarrollo-hot-reload).

### Tests

```bash
source .venv/bin/activate
invoke test
```

> **Convención del proyecto:** no usar `docker compose ...` directamente —
> `invoke` agrega validaciones (LFS, .env presente) que evitan errores crípticos.

---

## Después de un `git pull` (modo rápido)

Si alguien commiteó **migraciones nuevas** pero no cambió Dockerfiles ni
requirements, no hace falta rebuildear nada:

```bash
git pull
invoke migrate    # aplica las migraciones nuevas en el container existente
```

Tarda segundos. El core sigue corriendo, sólo se aplica el delta de schema.

Si el pull trae **cambios en Dockerfiles o requirements.txt**:

```bash
git pull
invoke up-build                            # rebuildea todos
# o, si solo cambió un servicio:
invoke up-build --service=core_management_api
```

Si el pull trae un **cambio de schema incompatible** (raro, pero pasa cuando
se rehace el modelo de datos):

```bash
git pull
invoke db-reset   # ⚠ borra la DB, la rehace desde cero y siembra
```

> **Nota histórica:** la versión vieja del repo creaba las tablas con
> `Base.metadata.create_all()`. Si tu volumen local tiene tablas de antes de
> que existieran las migraciones, `invoke db-reset` es la forma correcta de
> migrar a la nueva fuente de verdad (Alembic).

---

## Levantar módulos individualmente

Cada servicio se puede levantar suelto, respetando dependencias:

```bash
invoke up --service=db                       # solo la DB
invoke up --service=core_management_api      # core (levanta db automáticamente por depends_on)
invoke up --service=edge_device              # edge (levanta db)
invoke up --service=frontend                 # frontend (levanta core y db)
```

Reglas de dependencia (de [`docker-compose.yml`](docker-compose.yml)):

- `core_management_api` → necesita `db` (espera healthcheck OK)
- `edge_device` → necesita `db` (espera healthcheck OK)
- `frontend` → necesita `core_management_api`
- `core_management_api` y `edge_device` son independientes entre sí

Casos típicos:

- **Sólo backend para correr tests de API:** `invoke up --service=core_management_api`
- **Trabajar en visión sin levantar core:** `invoke up --service=edge_device`
- **Sólo DB para ejecutar `seed.py` o `alembic` desde el venv local:**
  `invoke up --service=db`

---

## Modo desarrollo (hot-reload)

Para iterar rápido **sin rebuildear contenedores cada vez que cambiás un .py**:

```bash
invoke up-dev
```

Qué hace:

- Usa [`docker-compose.dev.yml`](docker-compose.dev.yml) como override.
- Monta `core_management_api/` y `shared/` como **volúmenes** dentro del container.
- Reemplaza el ENTRYPOINT por `uvicorn --reload`. Editás un `.py` y se recarga
  en ~1s, sin rebuild.

Caveats importantes:

1. **En modo dev, `alembic upgrade head` NO corre automáticamente**
   (porque se override el entrypoint). Si agregás una migración nueva mientras
   estás en up-dev, ejecutá `invoke migrate` a mano.
2. **edge_device no tiene hot-reload en Docker.** Su entrypoint usa Hydra para
   inicializar cámaras antes de uvicorn, así que un `--reload` simple rompería
   esa inicialización. Si necesitás iterar en código de visión, lo más práctico
   es correrlo local: `cd edge_device && python scripts/run_server.py` (con el
   venv activo).
3. **Frontend hot-reload va por separado** (ver siguiente sección).

Para volver al modo normal: `invoke down && invoke up`.

### Frontend con HMR de Vite

El frontend en Docker usa nginx sirviendo build estático — los cambios requieren
rebuild, lento. Para iterar el frontend con hot module replacement de Vite:

```bash
docker compose stop frontend       # parar el contenedor del frontend
cd frontend_ui
npm install                        # primera vez
npm run dev                        # arranca en http://localhost:5173 con HMR
```

`npm run dev` apunta al core en `http://localhost:8001` por default
(controlado por `VITE_API_BASE_URL` en `.env`).

---

## Estructura del repo

```
CerebroVial/
├── core_management_api/        # Backend FastAPI (predictor + endpoints + alembic)
│   ├── alembic/                # Migraciones de schema (3 actuales)
│   ├── src/                    # Código del módulo
│   ├── entrypoint.sh           # Corre alembic upgrade head + uvicorn
│   ├── Dockerfile
│   └── requirements.txt
├── edge_device/                # Módulo de visión (YOLO + tracking + SSE)
├── ia_prediction_service/      # Pipeline de entrenamiento offline (no servicio)
├── frontend_ui/                # Dashboard Vite + React + Tailwind
├── shared/                     # Paquete pip local "cerebrovial_shared"
│   └── cerebrovial_shared/
│       ├── database/           # SQLAlchemy models, engine
│       ├── schemas/            # Pydantic schemas compartidos
│       └── ...
├── scripts/
│   ├── seed.py                 # Carga inicial de Miraflores
│   └── cleanup_jira.py
├── infra/
│   └── docker/initdb/          # Script de inicialización PostGIS
├── documentation/
│   ├── docs/                   # PLAN, TODO, DECISIONS, DATA_MODEL
│   ├── legacy/                 # Artefactos pre-refactor
│   └── tesis/                  # Documento académico
├── docker-compose.yml          # Compose principal (producción local)
├── docker-compose.dev.yml      # Override para hot-reload (opt-in)
├── tasks.py                    # Comandos de invoke
├── .env.example                # Plantilla de variables de entorno
├── CLAUDE.md                   # Reglas para asistentes IA
└── README.md                   # Este archivo
```

---

## Troubleshooting

### "ERROR: archivos LFS son pointers" al correr `invoke up`

Git-lfs no descargó los binarios. Solución:
```bash
git lfs install
git lfs pull
invoke up
```

### "ERROR: falta el archivo .env"

Falta crear el archivo desde la plantilla:
```bash
cp .env.example .env
# editar valores
```

### `invoke up` arranca pero el core crashea con `UnpicklingError`

Síntoma de LFS missing — los modelos `.joblib` están como pointers. Ver el
primer caso.

### `core_management_api` no levanta y los logs dicen `relation "..." does not exist`

Las migraciones no se aplicaron. Causas posibles:

1. Estás en `invoke up-dev` (no corre alembic en arranque): ejecutá `invoke migrate`.
2. La DB tiene tablas viejas del sistema pre-Alembic que entran en conflicto:
   `invoke db-reset`.

### "Bind for 0.0.0.0:5432 failed: port is already allocated"

Hay un container viejo o un Postgres del host ocupando el puerto:
```bash
docker compose down --remove-orphans
invoke up
```

### "could not translate host name 'db'" en logs locales

Estás corriendo algo del venv local (alembic, seed) sin estar dentro del
container. El código reemplaza automáticamente `@db:` por `@localhost:`,
pero verificá que en `.env` el `DATABASE_URL` apunte a `@db:5432` (no a un
hostname distinto), y que el servicio `db` esté arriba.

### `npm install` reporta vulnerabilidades

Deuda conocida (ver `documentation/docs/TODO.md` C10.2). No bloquea funcionalidad.

### Iterar es lento porque cada cambio rebuildea

Estás en modo prod. Cambiate a `invoke up-dev` para hot-reload del backend, y
usá `npm run dev` local para el frontend. Ver sección **Modo desarrollo**.

### Cambié código y `invoke up` sigue mostrando el estado anterior

`invoke up` solo arranca las imágenes ya construidas — no las rebuildea. Si
tocaste código que se compila a la imagen (típicamente el **frontend**, que se
sirve como build estático de Vite con nginx), necesitás rebuildear esa imagen.
Usá el flag `--service` para rebuildear solo el que tocaste:

```bash
invoke up-build --service=frontend              # cambios en frontend_ui/
invoke up-build --service=core_management_api   # cambios en core (sin up-dev) o requirements
invoke up-build --service=edge_device           # cambios en edge_device/ o requirements
invoke up-build --service=db                    # rara vez hace falta
invoke up-build                                 # cambios que afectan a varios
```

Reusa la cache de capas (no re-descarga `node_modules` ni `pip install`
mientras no cambien `package*.json` / `requirements.txt`) y solo recrea
el container del servicio indicado. Para `core_management_api` el código se
monta como volumen en `invoke up-dev` y los cambios se reflejan sin rebuild
(ver sección **Modo desarrollo**). `edge_device` no tiene hot-reload en
Docker — para iterar ahí lo más práctico es correrlo local con el venv.

---

## Para asistentes de IA (Claude Code, etc.)

Ver [`CLAUDE.md`](CLAUDE.md) en la raíz para reglas, contexto y convenciones del
proyecto. Los asistentes deben leerlo antes de tocar código.

---

## Estado del proyecto

- **Fase 1 (Estabilización del repo):** ✓ Cerrada el 2026-05-03.
- **Fase 2 (Cimientos: Alembic, JWT, frontend conectado, visión→BD):** en curso.
- **Fase 3 (Predictor GRU, control adaptativo, métricas):** después de Fase 2.

Detalle en [`documentation/docs/PLAN.md`](documentation/docs/PLAN.md).
