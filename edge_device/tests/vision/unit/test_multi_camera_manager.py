import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from omegaconf import DictConfig
from src.vision.application.services.multi_camera import MultiCameraManager, CameraInstance


@pytest.fixture(autouse=True)
def stub_create_detector():
    """B1 Paso 1a: CameraInstance ahora construye el detector vía `create_detector`
    e inyecta el resultado en `build_pipeline(detector=...)`. Estos tests mockean
    el builder entero y usan cfgs sin `model` key, así que stubeamos la factory
    para no cargar YOLO real. Es scaffolding: refleja la dependencia nueva del
    caller, no cambia la sustancia de ninguna aserción."""
    with patch(
        'src.vision.application.services.multi_camera.create_detector',
        return_value=MagicMock(),
    ):
        yield


@pytest.fixture(autouse=True)
def stub_scheduler():
    """B1 Paso 4: `start_camera` arma SIEMPRE un `CameraScheduler`. Los tests de
    nivel manager (registro, flags de render, lifecycle) NO ejercen el loop del
    scheduler — eso vive en test_camera_scheduler.py. Stubeamos la clase en el
    boundary del manager para no spinear threads reales (`ThreadedCapture` sobre un
    `pipeline.source` mockeado). Devuelve el mock de la clase para inspección;
    `.run` es awaitable (el task se crea con él) y `.stop` observable."""
    with patch('src.vision.application.services.multi_camera.CameraScheduler') as MockSched:
        instance = MockSched.return_value
        instance.run = AsyncMock()
        instance.stop = MagicMock()
        yield MockSched


@pytest.fixture
def mock_broadcaster():
    """Mock del broadcaster Protocol-conforme (§6.10/§6.11): publish/subscriber_count/is_subscribed."""
    broadcaster = MagicMock()
    broadcaster.publish = AsyncMock()
    return broadcaster

@pytest.fixture
def mock_builder():
    builder = MagicMock()
    pipeline = MagicMock()
    pipeline.run.return_value = [] # Empty generator by default
    builder.build_pipeline.return_value = pipeline
    return builder

@pytest.fixture
def manager(mock_broadcaster):
    return MultiCameraManager(mock_broadcaster)

def test_add_camera(manager):
    """C1.6 migrado: CameraInstance expone su id vía state.camera_id (no atributo directo)."""
    config = DictConfig({'vision': {'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        MockBuilder.return_value.build_pipeline.return_value = MagicMock()

        camera = manager.add_camera("cam1", config)

        assert "cam1" in manager.cameras
        assert isinstance(camera, CameraInstance)
        assert camera.state.camera_id == "cam1"
        # Caso B roto: renderer no se instancia adentro; default None.
        assert camera.state.renderer is None

def test_add_duplicate_camera(manager):
    config = DictConfig({'vision': {'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder'):
        manager.add_camera("cam1", config)
        with pytest.raises(ValueError):
            manager.add_camera("cam1", config)

@pytest.mark.asyncio
async def test_start_stop_camera(manager, stub_scheduler):
    """B1 Paso 4: start arma el scheduler y registra el task; stop lo para vía
    `scheduler.stop()` (ya NO `pipeline.stop()` — el path viejo se retiró) y limpia
    el task. El loop del scheduler está stubeado: este test cubre el lifecycle del
    manager, no la conducta del scheduler (esa vive en test_camera_scheduler.py)."""
    config = DictConfig({'vision': {'zones': {}}})

    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        MockBuilder.return_value.build_pipeline.return_value = MagicMock()

        manager.add_camera("cam1", config)

        # Start: scheduler creado, task registrado.
        await manager.start_camera("cam1")
        assert manager.cameras["cam1"].state.is_running
        assert "cam1" in manager._tasks
        assert manager.cameras["cam1"].state.scheduler is stub_scheduler.return_value

        # Stop: el scheduler se para y el task se limpia.
        await manager.stop_camera("cam1")
        assert not manager.cameras["cam1"].state.is_running
        assert "cam1" not in manager._tasks
        stub_scheduler.return_value.stop.assert_called_once()
        assert manager.cameras["cam1"].state.scheduler is None

@pytest.mark.asyncio
async def test_start_camera_not_found(manager):
    with pytest.raises(ValueError):
        await manager.start_camera("non_existent")

def test_get_status(manager):
    config = DictConfig({'vision': {'source': 'test', 'zones': {'z1': {}}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder'):
        manager.add_camera("cam1", config)
        
        status = manager.get_status()
        assert "cam1" in status
        assert status["cam1"]["source"] == "test"
        assert "z1" in status["cam1"]["zones"]


# ---- C1: baja dinámica, single-slot y auto-liberación ------------------

class _FakeDetector:
    """Detector con release() observable (C1/E1)."""

    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def _patch_builder(MockBuilder, detector=None):
    """Configura el builder patcheado: pipeline vacío + get_components con
    detector/aggregator controlados (aggregator None para no iterar flush)."""
    pipeline = MagicMock()
    pipeline.run.return_value = iter([])  # generador vacío: el task termina solo
    MockBuilder.return_value.build_pipeline.return_value = pipeline
    MockBuilder.return_value.get_components.return_value = {
        'detector': detector if detector is not None else _FakeDetector(),
        'aggregator': None,
    }


@pytest.mark.asyncio
async def test_remove_no_libera_shared_detector_shutdown_si(manager):
    """B1 Paso 4 (invariante nuevo): `remove_camera` NO libera el detector — es el
    singleton compartido, liberarlo mataría el modelo de las demás (F2). El
    compartido se libera UNA sola vez en `shutdown()`. Este par de asserts protege
    exactamente la falla que el guard `_owns_detector` prevenía."""
    shared_det = _FakeDetector()
    cfg = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder, \
         patch('src.vision.application.services.multi_camera.create_detector', return_value=shared_det):
        _patch_builder(MockBuilder)

        manager.add_camera("cam1", cfg)
        assert manager._shared_detector is shared_det   # el singleton compartido

        await manager.remove_camera("cam1")
        assert "cam1" not in manager.cameras            # sacada del registro
        assert shared_det.released is False             # NO liberado en remove (F2)

        await manager.shutdown()
        assert shared_det.released is True              # liberado UNA vez en shutdown

    # Idempotente: remover algo inexistente no rompe.
    await manager.remove_camera("cam1")


@pytest.mark.asyncio
async def test_activate_camera_keeps_others_alive(manager):
    """B1 Paso 0: sin single-slot, activar cam2 NO baja cam1 (abrir B no mata A).

    Reemplaza al viejo `test_activate_camera_single_slot`: el contrato se invirtió
    a propósito. La garantía "un solo YOLO" pasa al scheduler (Paso 1)."""
    cfg1 = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    cfg2 = DictConfig({'vision': {'source': 's2', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)

        await manager.activate_camera("cam1", cfg1)
        assert "cam1" in manager.cameras

        await manager.activate_camera("cam2", cfg2)
        # Ambas vivas: activar la segunda no liberó la primera.
        assert "cam2" in manager.cameras
        assert "cam1" in manager.cameras
        assert manager.cameras["cam1"].state.is_running
        assert manager.cameras["cam2"].state.is_running


@pytest.mark.asyncio
async def test_activate_camera_idempotent_same_source(manager):
    """Re-entrar con la misma fuente no recrea la cámara (idempotencia)."""
    cfg = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)

        await manager.activate_camera("cam1", cfg)
        first = manager.cameras["cam1"]

        await manager.activate_camera("cam1", cfg)
        assert manager.cameras["cam1"] is first  # misma instancia, no recreó


@pytest.mark.asyncio
async def test_sweep_idle_disables_render_without_mjpeg(manager, mock_broadcaster):
    """B1 Paso 0: el watchdog apaga el RENDER (no baja la cámara) sin MJPEG.

    Reemplaza al viejo `test_sweep_idle_releases_camera_without_consumers`: la
    acción cambió de `remove_camera` a apagar render; el muestreo sobrevive.
    Incluye el caso clave: con SSE pero sin MJPEG el render igual se apaga (el
    SSE no consume frames)."""
    mock_broadcaster.subscribed_cameras.return_value = ["cam1"]  # SSE activo, sin MJPEG
    cfg = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        await manager.activate_camera("cam1", cfg)
        assert manager.cameras["cam1"].state.render_enabled is True

        # Primera pasada: marca idle, NO apaga todavía.
        assert await manager._sweep_idle(now=1000.0) == []
        assert manager.cameras["cam1"].state.render_enabled is True

        # Pasada superado el timeout: apaga SOLO el render.
        disabled = await manager._sweep_idle(now=1000.0 + manager.idle_timeout_s + 1)
        assert disabled == ["cam1"]
        # La cámara sigue viva y muestreando; solo se apagó el render.
        assert "cam1" in manager.cameras
        assert manager.cameras["cam1"].state.is_running
        assert manager.cameras["cam1"].state.render_enabled is False
        assert manager.cameras["cam1"].state.latest_frame_raw is None


@pytest.mark.asyncio
async def test_mjpeg_consumer_keeps_render_on_then_watchdog_disables(manager, mock_broadcaster):
    """Con consumidor MJPEG el render se mantiene; al irse, el watchdog lo apaga.

    Reemplaza al viejo `test_mjpeg_consumer_keeps_camera_alive`: la cámara NO se
    baja en ningún caso; lo que se gestiona es el render."""
    mock_broadcaster.subscribed_cameras.return_value = []
    cfg = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        await manager.activate_camera("cam1", cfg)

        manager.add_mjpeg_consumer("cam1")
        # Con consumidor MJPEG, ni un `now` enorme apaga el render.
        assert await manager._sweep_idle(now=10_000.0) == []
        assert manager.cameras["cam1"].state.render_enabled is True

        manager.remove_mjpeg_consumer("cam1")
        # Sin MJPEG: marca idle y apaga el render tras el timeout.
        await manager._sweep_idle(now=20_000.0)
        disabled = await manager._sweep_idle(now=20_000.0 + manager.idle_timeout_s + 1)
        assert disabled == ["cam1"]
        # La cámara sigue viva; solo se apagó el render.
        assert "cam1" in manager.cameras
        assert manager.cameras["cam1"].state.is_running
        assert manager.cameras["cam1"].state.render_enabled is False
