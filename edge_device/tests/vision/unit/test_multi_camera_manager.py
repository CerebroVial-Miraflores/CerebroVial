import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from omegaconf import DictConfig
from src.vision.application.services.multi_camera import MultiCameraManager, CameraInstance

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
async def test_activate_camera_single_slot(manager):
    """Un solo YOLO vivo (D2): activar cam2 libera cam1."""
    cfg1 = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    cfg2 = DictConfig({'vision': {'source': 's2', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)

        await manager.activate_camera("cam1", cfg1)
        assert "cam1" in manager.cameras

        await manager.activate_camera("cam2", cfg2)
        assert "cam2" in manager.cameras
        assert "cam1" not in manager.cameras  # single-slot liberó la anterior


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
async def test_sweep_idle_releases_camera_without_consumers(manager, mock_broadcaster):
    """Watchdog (E4): una cámara sin consumidores se libera al superar el timeout."""
    mock_broadcaster.subscribed_cameras.return_value = []  # 0 consumidores SSE
    cfg = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        await manager.activate_camera("cam1", cfg)

        # Primera pasada: marca idle, NO libera todavía.
        assert await manager._sweep_idle(now=1000.0) == []
        assert "cam1" in manager.cameras

        # Pasada superado el timeout: libera.
        released = await manager._sweep_idle(now=1000.0 + manager.idle_timeout_s + 1)
        assert released == ["cam1"]
        assert "cam1" not in manager.cameras


@pytest.mark.asyncio
async def test_mjpeg_consumer_keeps_camera_alive(manager, mock_broadcaster):
    """Un consumidor MJPEG activo impide la auto-liberación; al irse, se libera."""
    mock_broadcaster.subscribed_cameras.return_value = []
    cfg = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        await manager.activate_camera("cam1", cfg)

        manager.add_mjpeg_consumer("cam1")
        # Con consumidor MJPEG, ni siquiera un `now` enorme la libera.
        assert await manager._sweep_idle(now=10_000.0) == []
        assert "cam1" in manager.cameras

        manager.remove_mjpeg_consumer("cam1")
        # Sin consumidores: marca idle y libera tras el timeout.
        await manager._sweep_idle(now=20_000.0)
        released = await manager._sweep_idle(now=20_000.0 + manager.idle_timeout_s + 1)
        assert released == ["cam1"]
        assert "cam1" not in manager.cameras


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
