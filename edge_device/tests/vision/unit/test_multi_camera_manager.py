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
async def test_start_stop_camera(manager):
    """C1.6 migrado: is_running vive en state.is_running (no atributo directo).

    Sin OpenCVVisualizer hardcoded (Caso B roto): el pipeline_mock yield
    MagicMocks sin tocar cv2.putText, porque el render condicional solo se
    ejecuta si `state.renderer is not None`.
    """
    config = DictConfig({'vision': {'zones': {}}})

    pipeline_mock = MagicMock()
    pipeline_mock.run.return_value = iter([(MagicMock(), MagicMock())])

    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        MockBuilder.return_value.build_pipeline.return_value = pipeline_mock

        manager.add_camera("cam1", config)

        # Start.
        await manager.start_camera("cam1")
        assert manager.cameras["cam1"].state.is_running
        assert "cam1" in manager._tasks

        # Stop.
        await manager.stop_camera("cam1")
        assert not manager.cameras["cam1"].state.is_running
        assert "cam1" not in manager._tasks
        pipeline_mock.stop.assert_called_once()

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
async def test_remove_camera_releases_model_and_unregisters(manager):
    config = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    detector = _FakeDetector()
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder, detector=detector)

        manager.add_camera("cam1", config)
        assert manager.cameras["cam1"].state.detector is detector

        await manager.remove_camera("cam1")

        assert "cam1" not in manager.cameras          # sacada del registro
        assert detector.released is True              # modelo liberado

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


@pytest.mark.asyncio
async def test_pipeline_render_gated_muestreo_always_runs(manager):
    """B1 Paso 0 — separación muestreo/render: con `render_enabled=False` el loop
    NO guarda frames ni anota (render off), PERO el muestreo sigue: drena el
    aggregator y publica al broadcaster."""
    import numpy as np
    from src.vision.domain.entities import DetectedVehicle, Frame, FrameAnalysis

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    frame = Frame(id=0, timestamp=0.0, image=img)
    vehicle = DetectedVehicle(id="1", type="car", confidence=0.9, bbox=(0, 0, 2, 2), timestamp=0.0)
    analysis = FrameAnalysis(
        frame_id=0, timestamp=0.0, vehicles=[vehicle], unique_vehicles=1,
        zones={}, detection_ran=True,
    )

    renderer = MagicMock()
    renderer.render.return_value = img
    aggregator = MagicMock()
    aggregator.flush.return_value = ["td-1"]  # un agregado pendiente de publicar

    cfg = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        pipeline = MagicMock()
        pipeline.run.return_value = iter([(frame, analysis)])
        MockBuilder.return_value.build_pipeline.return_value = pipeline
        MockBuilder.return_value.get_components.return_value = {
            'detector': _FakeDetector(), 'aggregator': aggregator,
        }

        manager.add_camera("cam1", cfg, renderer=renderer)
        manager.cameras["cam1"].state.render_enabled = False  # render apagado
        await manager.start_camera("cam1")
        await manager._tasks["cam1"]  # pipeline finito → el task termina solo

    # Render OFF: no se guardó frame ni se anotó.
    assert manager.cameras["cam1"].state.latest_frame_raw is None
    assert manager.cameras["cam1"].state.latest_frame_processed is None
    renderer.render.assert_not_called()

    # Muestreo ON: el aggregator se drenó y se publicó al broadcaster.
    assert aggregator.flush.called
    manager.broadcaster.publish.assert_awaited_once_with("td-1")


@pytest.mark.asyncio
async def test_muestreo_survives_watchdog_render_off(manager, mock_broadcaster):
    """B1 Paso 0: tras apagar el render por watchdog, el muestreo sigue publicando.

    Demuestra extremo a extremo que el watchdog no toca el muestreo: apaga el
    render y, con la cámara ya en ese estado, el pipeline sigue drenando el
    aggregator → broadcaster."""
    import numpy as np
    from src.vision.domain.entities import Frame, FrameAnalysis

    mock_broadcaster.subscribed_cameras.return_value = []
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    frame = Frame(id=0, timestamp=0.0, image=img)
    analysis = FrameAnalysis(
        frame_id=0, timestamp=0.0, vehicles=[], unique_vehicles=0,
        zones={}, detection_ran=True,
    )
    aggregator = MagicMock()
    aggregator.flush.return_value = ["td-post-watchdog"]

    cfg = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        pipeline = MagicMock()
        pipeline.run.return_value = iter([(frame, analysis)])
        MockBuilder.return_value.build_pipeline.return_value = pipeline
        MockBuilder.return_value.get_components.return_value = {
            'detector': _FakeDetector(), 'aggregator': aggregator,
        }

        manager.add_camera("cam1", cfg)

        # Watchdog apaga el render (sin MJPEG): dos pasadas (marca + timeout).
        await manager._sweep_idle(now=1000.0)
        disabled = await manager._sweep_idle(now=1000.0 + manager.idle_timeout_s + 1)
        assert disabled == ["cam1"]
        assert manager.cameras["cam1"].state.render_enabled is False
        assert "cam1" in manager.cameras  # cámara viva

        # Con el render ya apagado, el muestreo sigue: el pipeline publica.
        await manager.start_camera("cam1")
        await manager._tasks["cam1"]

    mock_broadcaster.publish.assert_awaited_once_with("td-post-watchdog")
    assert manager.cameras["cam1"].state.latest_frame_raw is None  # render quedó off


@pytest.mark.asyncio
async def test_visual_box_persistence_without_double_count(manager):
    """C1: en skip frames (detection_ran=False) la visualización persiste los boxes
    del último frame inferido, PERO esos boxes nunca tocan métricas — el aggregator
    solo recibe el análisis real aguas arriba (acá: _run_camera_pipeline solo hace
    flush, nunca add)."""
    import numpy as np
    from src.vision.domain.entities import DetectedVehicle, Frame, FrameAnalysis

    img = np.zeros((4, 4, 3), dtype=np.uint8)
    frame_detect = Frame(id=0, timestamp=0.0, image=img)
    frame_skip = Frame(id=1, timestamp=0.01, image=img)

    vehicle = DetectedVehicle(id="1", type="car", confidence=0.9, bbox=(0, 0, 2, 2), timestamp=0.0)
    analysis_detect = FrameAnalysis(
        frame_id=0, timestamp=0.0, vehicles=[vehicle], unique_vehicles=1,
        zones={}, detection_ran=True,
    )
    analysis_skip = FrameAnalysis(
        frame_id=1, timestamp=0.01, vehicles=[], unique_vehicles=0,
        zones={}, detection_ran=False,
    )

    renderer = MagicMock()
    renderer.render.return_value = img
    aggregator = MagicMock()
    aggregator.flush.return_value = []

    cfg = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        pipeline = MagicMock()
        pipeline.run.return_value = iter([(frame_detect, analysis_detect), (frame_skip, analysis_skip)])
        MockBuilder.return_value.build_pipeline.return_value = pipeline
        MockBuilder.return_value.get_components.return_value = {
            'detector': _FakeDetector(), 'aggregator': aggregator,
        }

        manager.add_camera("cam1", cfg, renderer=renderer)
        await manager.start_camera("cam1")
        await manager._tasks["cam1"]  # el pipeline es finito → el task termina solo

    # En el skip frame el renderer recibió el análisis PERSISTIDO (detect), no el vacío.
    rendered = [call.args[1] for call in renderer.render.call_args_list]
    assert rendered == [analysis_detect, analysis_detect]

    # Los boxes persistidos NUNCA tocan métricas: el loop solo hace flush, nunca add.
    aggregator.add.assert_not_called()
    assert aggregator.flush.called
