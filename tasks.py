"""
Tareas de invoke para CerebroVial.
Ejecutar con: invoke <target>
Ver lista completa: invoke --list
"""
from invoke import task
import sys
import platform
import os
import time


# === Helpers ===

def _print_box(title, lines):
    """Imprime un mensaje destacado, multilínea, con bordes."""
    width = max(len(title), max(len(l) for l in lines)) + 4
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    for l in lines:
        print(f"  {l}")
    print("=" * width)
    print()


def _venv_python():
    """Path al python del venv local. Lanza error si no existe."""
    is_windows = platform.system() == "Windows"
    path = ".venv\\Scripts\\python.exe" if is_windows else ".venv/bin/python"
    if not os.path.exists(path):
        _print_box("ERROR: no hay venv local", [
            "Crealo primero: invoke setup-dev",
        ])
        sys.exit(1)
    return path


# El override de dev se compone con el compose principal cuando se invocan
# los targets *-dev. NO se carga automáticamente en `invoke up`.
_DEV_COMPOSE = "-f docker-compose.yml -f docker-compose.dev.yml"


# === Validación de entorno ===

@task
def check_lfs(c):
    """Verifica que los binarios LFS sean reales y no pointers de texto."""
    # Un sentinela por familia de modelo: el RandomForest baseline (.joblib) y el
    # GRU servido por el core (.pt, TTH-09 Fase 3c). Si cualquiera está ausente o
    # quedó como puntero LFS sin materializar, abortamos antes de levantar.
    sentinels = [
        "core_management_api/models/traffic_rf_class_current.joblib",
        "core_management_api/models/gru/gru_N.pt",
    ]
    for sentinel in sentinels:
        if not os.path.exists(sentinel):
            _print_box("ERROR: archivo de modelo no encontrado", [
                f"Path esperado: {sentinel}",
                "",
                "Probablemente el clone fue parcial o git-lfs no está activo.",
                "Solución:",
                "  1. Instalá git-lfs:",
                "       macOS:  brew install git-lfs",
                "       Linux:  apt install git-lfs",
                "       Windows: https://git-lfs.com (descargar instalador)",
                "  2. git lfs install",
                "  3. git lfs pull",
            ])
            sys.exit(1)

        with open(sentinel, "rb") as f:
            head = f.read(64)
        if head.startswith(b"version https://git-lfs"):
            _print_box("ERROR: archivos LFS son pointers, no binarios reales", [
                f"El modelo {sentinel} está como puntero de texto.",
                "Esto significa que git-lfs no descargó los binarios.",
                "",
                "Solución:",
                "  1. Instalá git-lfs (ver instrucciones en README.md).",
                "  2. git lfs install",
                "  3. git lfs pull",
                "",
                "Después de pullear, volvé a correr este comando.",
            ])
            sys.exit(1)

    print("✓ LFS OK — binarios presentes y reales")


@task
def check_env(c):
    """Verifica que .env exista en la raíz del repo."""
    if not os.path.exists(".env"):
        _print_box("ERROR: falta el archivo .env", [
            "Copialo desde la plantilla y completá los valores:",
            "",
            "  cp .env.example .env",
            "",
            "Después editá .env y reemplazá 'changeme' por valores reales.",
        ])
        sys.exit(1)


# === Comandos del día a día ===

@task(pre=[check_lfs, check_env], help={
    "service": "Servicio único a levantar (db, core_management_api, edge_device, frontend). "
               "Si se omite, levanta todos.",
})
def up(c, service=None):
    """Levantar el sistema (todos los servicios o uno solo).

    El core_management_api corre `alembic upgrade head` en su entrypoint,
    así que las migraciones se aplican automáticamente al arrancar.
    Para hot-reload del backend, usá `invoke up-dev`.
    """
    target = service or ""
    c.run(f"docker compose up -d --remove-orphans {target}", pty=False)
    print()
    if service:
        print(f"✓ Servicio '{service}' levantado.")
    else:
        print("✓ Sistema levantado.")
        print("  - core_management_api: http://localhost:8001  (Swagger en /docs)")
        print("  - edge_device:         http://localhost:8000  (Swagger en /docs)")
        print("  - frontend:            http://localhost:5173")
        print("  - db (Postgres):       localhost:5432")
    print()
    print("Las tablas se crean solas (alembic corre en el entrypoint del core).")
    print("Para cargar datos iniciales de Miraflores: invoke seed")
    print()
    print("Otros: invoke logs · invoke ps · invoke down")


@task(pre=[check_lfs, check_env], help={
    "service": "Servicio único a rebuildear (db, core_management_api, edge_device, frontend). "
               "Si se omite, rebuildea todos.",
})
def up_build(c, service=None):
    """Levantar con rebuild de imágenes (usa cache de capas; no re-descarga deps).

    Útil después de tocar Dockerfile, requirements.txt, package.json o el código
    de un servicio servido como build estático (p.ej. el frontend, que sirve el
    output de `npm run build` con nginx).

    Ejemplos:
      invoke up-build                       # rebuildea todos
      invoke up-build --service=frontend    # solo frontend
      invoke up-build --service=core_management_api
    """
    target = service or ""
    c.run(f"docker compose up -d --build --remove-orphans {target}", pty=False)
    print()
    if service:
        print(f"✓ Servicio '{service}' rebuildeado y levantado.")
    else:
        print("✓ Sistema levantado (con rebuild).")


@task(pre=[check_lfs, check_env])
def up_dev(c):
    """Levantar en modo desarrollo: hot-reload del core_management_api.

    Usa docker-compose.dev.yml como override. Monta el código como volumen y
    corre uvicorn con --reload. Las migraciones NO corren automáticamente en
    este modo — ejecutá `invoke migrate` cuando agregues una migración nueva.

    Frontend y edge_device se levantan sin override (ver docker-compose.dev.yml
    para el porqué). Para iterar frontend: `cd frontend_ui && npm run dev`.
    """
    c.run(f"docker compose {_DEV_COMPOSE} up -d --remove-orphans", pty=False)
    print()
    print("✓ Sistema levantado en modo DEV (hot-reload de core_management_api).")
    print("  Editá archivos en core_management_api/ o shared/ y se recargan solos.")
    print()
    print("⚠ IMPORTANTE: en modo dev, alembic NO corre automáticamente.")
    print("  Si agregás una migración nueva: invoke migrate")
    print()
    print("  Para HMR de frontend:")
    print("    docker compose stop frontend && cd frontend_ui && npm run dev")


@task
def down(c):
    """Apagar el sistema (conserva volúmenes y datos de DB)."""
    c.run("docker compose down", pty=False)


@task
def logs(c, service=None):
    """Ver logs en vivo. Pasá --service=core_management_api para filtrar."""
    target = service or ""
    c.run(f"docker compose logs -f {target}", pty=False)


@task
def ps(c):
    """Estado actual de los containers."""
    c.run("docker compose ps", pty=False)


@task
def health(c):
    """Validar que core_management_api responde.

    Desde HU-01 (RBAC), GET /api/health está protegido con require_role(admin).
    El smoke de proceso vivo usa GET / (root, abierto, retorna name+version).
    Para chequear /api/health con credenciales, exportá ADMIN_TOKEN y corré
    `curl -i -H "Authorization: Bearer $ADMIN_TOKEN" http://localhost:8001/api/health`.
    """
    c.run("curl -i http://localhost:8001/", pty=False, warn=True)


# === Migraciones y datos ===

@task
def migrate(c):
    """Aplicar migraciones Alembic (sin rebuild de imagen).

    Útil después de hacer git pull cuando hay migraciones nuevas en
    core_management_api/alembic/versions/. Mucho más rápido que `invoke up-build`.
    Requiere que el container core_management_api esté arriba.
    """
    c.run("docker compose exec core_management_api alembic upgrade head", pty=False)
    print("\n✓ Migraciones aplicadas.")


@task(help={"message": "Descripción corta de la migración"})
def migrate_create(c, message):
    """Generar una migración nueva con autogenerate.

    Uso: invoke migrate-create -m "add foo column"
    """
    c.run(
        f'docker compose exec core_management_api '
        f'alembic revision --autogenerate -m "{message}"',
        pty=False,
    )
    print("\n✓ Migración creada en core_management_api/alembic/versions/.")
    print("  Revisala antes de commitear — autogenerate no es perfecto.")


@task(pre=[check_env])
def seed(c):
    """Cargar datos iniciales de Miraflores (5 nodos, 6 edges, 4 cámaras, 1 admin).

    Idempotente: usa session.merge(), se puede correr múltiples veces sin duplicar.
    Corre desde el venv local (scripts/ no se copia al container).
    Requiere `invoke setup-dev` previo y que la DB esté arriba.
    """
    py = _venv_python()
    c.run(f"{py} scripts/seed.py", pty=False)
    print("\n✓ Seed aplicado.")


@task(pre=[check_env])
def seed_rbac_smoke(c):
    """Sembrar 3 usuarios de smoke (operator, manager, admin) para HU-01.

    Credenciales de desarrollo (no usar fuera de smoke local):
      operator@cv.pe / Smoke1234  → rol operator
      manager@cv.pe  / Smoke1234  → rol manager
      admin@cv.pe    / Smoke1234  → rol admin

    Idempotente: salta usuarios cuyo email ya existe.
    Hashing vía hash_password() de TTH-01 (bcrypt cost=12).
    Útil para el smoke manual de CA-01.6 (auto-logout por token expirado) y
    para probar visibilidad de pestañas por rol en el frontend.
    """
    py = _venv_python()
    c.run(f"{py} scripts/seed_rbac_smoke.py", pty=False)


@task
def db_reset(c):
    """⚠ Borra la DB, la levanta de cero, aplica migraciones y siembra.

    Útil cuando el schema cambió de forma incompatible y querés arrancar limpio.
    También resuelve "tengo el volumen viejo del sistema anterior" después de
    un git pull con cambios estructurales.
    """
    confirm = input("⚠️  Esto BORRA todos los datos de la DB. ¿Continuar? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Cancelado.")
        return
    c.run("docker compose down -v", pty=False)
    c.run("docker compose up -d db", pty=False)

    print("Esperando que la DB esté lista...")
    for _ in range(30):
        result = c.run(
            "docker compose exec -T db pg_isready -U cerebrovial -d cerebrovial",
            pty=False, warn=True, hide=True,
        )
        if result.ok:
            break
        time.sleep(1)
    else:
        print("✗ La DB no respondió a tiempo.")
        sys.exit(1)

    # Levantar core: su entrypoint correrá alembic upgrade head al arrancar.
    c.run("docker compose up -d core_management_api", pty=False)
    print("Esperando que core_management_api termine las migraciones...")
    time.sleep(5)
    seed(c)
    print("\n✓ DB reseteada, migrada y sembrada.")
    print("  Para levantar el resto del sistema: invoke up")


# === Acceso interactivo a los containers ===

@task
def shell_api(c):
    """Abrir shell en el container core_management_api."""
    c.run("docker compose exec core_management_api /bin/bash", pty=True)


@task
def shell_db(c):
    """Abrir psql en el container db."""
    c.run("docker compose exec db psql -U cerebrovial -d cerebrovial", pty=True)


# === Builds limpios ===

@task(pre=[check_lfs, check_env])
def rebuild(c):
    """Build limpio de imágenes (sin cache de capas), conservando volúmenes."""
    c.run("docker compose build --no-cache", pty=False)
    c.run("docker compose up -d --remove-orphans", pty=False)
    print("\n✓ Rebuild completado, sistema arriba.")


@task(pre=[check_lfs, check_env])
def rebuild_clean(c):
    """Build nuclear: borra volúmenes (incluida DB), pulls fresh, rebuilds, levanta. ⚠ borra datos."""
    confirm = input("⚠️  Esto BORRA los datos de la DB. ¿Continuar? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Cancelado.")
        return
    c.run("docker compose down -v", pty=False)
    c.run("docker compose build --no-cache --pull", pty=False)
    c.run("docker compose up -d --remove-orphans", pty=False)
    print("\n✓ Rebuild-clean completado.")


# === Setup y tests ===

@task
def setup_dev(c):
    """Crear venv local CORE+VISIÓN (.venv) con deps de dev (numpy>=2, sin tsl).

    El entrenamiento del GRU (tsl, numpy 1.x) vive en un venv APARTE: ver
    `invoke setup-train`. Los dos venvs están separados a propósito — tsl ancla
    numpy<2 y choca con opencv/visión (ver documentation/ESTADO_Y_PROXIMOS_PASOS.md).
    """
    is_windows = platform.system() == "Windows"
    venv_python = ".venv\\Scripts\\python.exe" if is_windows else ".venv/bin/python"

    print("Creando venv core+visión en .venv/ ...")
    c.run("python -m venv .venv" if is_windows else "python3.11 -m venv .venv", pty=False)
    c.run(f"{venv_python} -m pip install --upgrade pip", pty=False)
    # Instalación COMBINADA en UNA sola resolución de pip: así el pin torch==2.9.1 del
    # core (excepción D-010) compone con los deps de visión (edge) en vez de ser pisado
    # por una invocación separada (ultralytics/torchvision arrastrarían un torch más nuevo).
    # Se corre desde core_management_api/ para que los `-e ../shared` de los requirements
    # resuelvan a ./shared (pip resuelve los paths relativos de un requirements contra el CWD).
    abs_python = os.path.abspath(venv_python)
    with c.cd("core_management_api"):
        c.run(
            f'"{abs_python}" -m pip install -e ../shared '
            f"-r requirements.txt -r ../edge_device/requirements.txt -r ../requirements-dev.txt",
            pty=False,
        )

    activate = ".venv\\Scripts\\activate" if is_windows else "source .venv/bin/activate"
    _print_box("✓ Venv core+visión creado en .venv/", [
        f"Activalo con: {activate}",
        "Después podés correr: invoke test, invoke seed, alembic, etc.",
        "Para entrenar el GRU: invoke setup-train (venv aparte).",
    ])


@task
def setup_train(c):
    """Crear venv de ENTRENAMIENTO en ia_prediction_service/.venv (tsl + numpy 1.x).

    Aislado del .venv core+visión: tsl exige numpy<2 (Requires-Dist: numpy<2,>1.20.3)
    y este venv NO debe tener opencv/cv2. Usalo para los dataset builders y scripts de
    train de ia_prediction_service.
    """
    is_windows = platform.system() == "Windows"
    venv_python = (
        "ia_prediction_service\\.venv\\Scripts\\python.exe"
        if is_windows
        else "ia_prediction_service/.venv/bin/python"
    )

    print("Creando venv de training en ia_prediction_service/.venv/ ...")
    c.run(
        "python -m venv ia_prediction_service/.venv"
        if is_windows
        else "python3.11 -m venv ia_prediction_service/.venv",
        pty=False,
    )
    c.run(f"{venv_python} -m pip install --upgrade pip", pty=False)
    # Correr desde ia_prediction_service/ para que el `-e ../shared` del requirements
    # resuelva a ./shared (pip resuelve los paths relativos contra el CWD).
    abs_python = os.path.abspath(venv_python)
    with c.cd("ia_prediction_service"):
        c.run(f'"{abs_python}" -m pip install -r requirements.txt', pty=False)

    activate = (
        "ia_prediction_service\\.venv\\Scripts\\activate"
        if is_windows
        else "source ia_prediction_service/.venv/bin/activate"
    )
    _print_box("✓ Venv de training creado en ia_prediction_service/.venv/", [
        f"Activalo con: {activate}",
        "Corré los scripts de train/dataset con este python, p. ej.:",
        "  ia_prediction_service/.venv/bin/python \\",
        "    ia_prediction_service/scripts/train_miraflores_baseline.py --quick",
    ])


@task
def test(c):
    """Correr tests de core_management_api y edge_device.

    Asume que el venv está activo y tiene pytest instalado.
    Si falla con 'No module named pytest', correr 'invoke setup-dev' primero
    y activar el venv.
    """
    print("=== core_management_api ===")
    c.run("cd core_management_api && python -m pytest tests/ -v", pty=False, warn=True)
    print("\n=== edge_device ===")
    c.run("cd edge_device && python -m pytest tests/ -v", pty=False, warn=True)


# === Help ===

@task(default=True)
def help(c):
    """Muestra los comandos disponibles."""
    c.run("invoke --list", pty=False)
