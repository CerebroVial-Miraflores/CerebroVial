"""Tests de lifecycle del MultiCameraManager (topología B / 15Hz).

El manager ya NO crea un `CameraScheduler` por cámara: arranca un `QueuePushProducer`
(captura push) y registra la post-chain en el `BatchInferenceWorker` central
(singleton lazy, creado con la 1ª cámara junto al detector). Estos tests cubren el
lifecycle del manager (registro, render flags, alta/baja), NO la conducta del worker
(eso vive en test_batch_inference.py) ni del producer (test_queue_push_producer.py):
ambos se stubean en el boundary del manager para no spinear threads/tasks reales.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from omegaconf import DictConfig

from src.vision.application.services.multi_camera import (
    CameraInstance,
    InferenceCapacityError,
    MultiCameraManager,
)


@pytest.fixture(autouse=True)
def stub_create_detector():
    """add_camera crea el detector compartido vía `create_detector` (lazy). Se
    stubea para no cargar YOLO real."""
    with patch(
        'src.vision.application.services.multi_camera.create_detector',
        return_value=MagicMock(),
    ):
        yield


@pytest.fixture(autouse=True)
def stub_batched():
    """Stub del batch worker + producer en el boundary del manager: nada de tasks
    asyncio ni daemon threads reales. Devuelve (MockWorker, MockProducer) para
    inspección."""
    with patch('src.vision.application.services.multi_camera.BatchInferenceWorker') as MockWorker, \
         patch('src.vision.application.services.multi_camera.QueuePushProducer') as MockProducer:
        worker = MockWorker.return_value
        worker.start = MagicMock()
        worker.register = MagicMock()
        worker.unregister = MagicMock()
        worker.stop = AsyncMock()
        producer = MockProducer.return_value
        producer.start = MagicMock()
        producer.stop = MagicMock()
        yield MockWorker, MockProducer


@pytest.fixture
def mock_broadcaster():
    broadcaster = MagicMock()
    broadcaster.publish = AsyncMock()
    return broadcaster


@pytest.fixture
def manager(mock_broadcaster):
    return MultiCameraManager(mock_broadcaster)


def _patch_builder(MockBuilder):
    """El builder mockeado: CameraInstance llama build_source()/build_post_chain()
    y lee builder.source/builder.aggregator (auto-mocks). aggregator=None para que
    el teardown no itere un flush real."""
    MockBuilder.return_value.aggregator = None


def test_manager_default_knobs_preserve_behavior(manager):
    """Defaults de los knobs de instancia = comportamiento actual (15fps / imgsz 640)."""
    assert manager.analyze_fps == 15
    assert manager.imgsz == 640


def test_add_camera(manager):
    config = DictConfig({'vision': {'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        camera = manager.add_camera("cam1", config)

        assert "cam1" in manager.cameras
        assert isinstance(camera, CameraInstance)
        assert camera.state.camera_id == "cam1"
        assert camera.state.renderer is None
        # Topología B: source + post_chain armados; sin pipeline (None).
        assert camera.state.source is not None
        assert camera.state.post_chain is not None
        assert camera.state.pipeline is None


def test_add_duplicate_camera(manager):
    config = DictConfig({'vision': {'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        manager.add_camera("cam1", config)
        with pytest.raises(ValueError):
            manager.add_camera("cam1", config)


@pytest.mark.asyncio
async def test_start_stop_camera(manager, stub_batched):
    """Start: registra la post-chain en el worker y arranca el producer. Stop:
    desregistra del worker, para el producer y limpia la referencia."""
    MockWorker, MockProducer = stub_batched
    config = DictConfig({'vision': {'zones': {}}})

    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        manager.add_camera("cam1", config)

        await manager.start_camera("cam1")
        assert manager.cameras["cam1"].state.is_running
        # Post-chain registrada en el worker; producer arrancado.
        MockWorker.return_value.register.assert_called_once()
        assert manager.cameras["cam1"].state.producer is MockProducer.return_value
        MockProducer.return_value.start.assert_called_once()

        await manager.stop_camera("cam1")
        assert not manager.cameras["cam1"].state.is_running
        MockWorker.return_value.unregister.assert_called_once_with("cam1")
        MockProducer.return_value.stop.assert_called_once()
        assert manager.cameras["cam1"].state.producer is None


@pytest.mark.asyncio
async def test_start_camera_order_detector_before_producer(manager, stub_batched):
    """Cuidado (b): el detector existe ANTES de que arranque el producer (que no
    fluyan frames sin modelo). El worker se crea con el detector compartido."""
    MockWorker, MockProducer = stub_batched
    config = DictConfig({'vision': {'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        manager.add_camera("cam1", config)
        assert manager._shared_detector is not None  # detector creado en add_camera

        await manager.start_camera("cam1")
        # El worker se construyó con el detector compartido como 1er arg posicional.
        assert MockWorker.call_args.args[0] is manager._shared_detector
        MockProducer.return_value.start.assert_called_once()


@pytest.mark.asyncio
async def test_start_camera_not_found(manager):
    with pytest.raises(ValueError):
        await manager.start_camera("non_existent")


def test_get_status(manager):
    config = DictConfig({'vision': {'source': 'test', 'zones': {'z1': {}}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        manager.add_camera("cam1", config)

        status = manager.get_status()
        assert "cam1" in status
        assert status["cam1"]["source"] == "test"
        assert "z1" in status["cam1"]["zones"]


# ---- baja dinámica + shutdown -----------------------------------------


class _FakeDetector:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


@pytest.mark.asyncio
async def test_remove_no_libera_shared_detector_shutdown_si(manager):
    """`remove_camera` NO libera el detector (singleton compartido); `shutdown()`
    sí, UNA vez."""
    shared_det = _FakeDetector()
    cfg = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder, \
         patch('src.vision.application.services.multi_camera.create_detector', return_value=shared_det):
        _patch_builder(MockBuilder)

        manager.add_camera("cam1", cfg)
        assert manager._shared_detector is shared_det

        await manager.remove_camera("cam1")
        assert "cam1" not in manager.cameras
        assert shared_det.released is False  # NO liberado en remove (lo usan las demás)

        await manager.shutdown()
        assert shared_det.released is True   # liberado UNA vez en shutdown

    await manager.remove_camera("cam1")  # idempotente


@pytest.mark.asyncio
async def test_shutdown_stops_batch_worker(manager, stub_batched):
    """shutdown() apaga el batch worker (su task asyncio) si fue creado."""
    MockWorker, _ = stub_batched
    cfg = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        await manager.activate_camera("cam1", cfg)  # crea el worker

        await manager.shutdown()
        MockWorker.return_value.stop.assert_awaited_once()
        assert manager._batch_worker is None


# ---- tope del contenedor de inferencia --------------------------------


@pytest.mark.asyncio
async def test_start_camera_rejected_beyond_cap(manager):
    manager.max_inference_cameras = 1
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        await manager.activate_camera("cam1", DictConfig({'vision': {'source': 's1', 'zones': {}}}))

        manager.add_camera("cam2", DictConfig({'vision': {'source': 's2', 'zones': {}}}))
        with pytest.raises(InferenceCapacityError, match="tope 1"):
            await manager.start_camera("cam2")
        assert not manager.cameras["cam2"].state.is_running


@pytest.mark.asyncio
async def test_activate_beyond_cap_cleans_up_the_rejected_camera(manager):
    manager.max_inference_cameras = 1
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        await manager.activate_camera("cam1", DictConfig({'vision': {'source': 's1', 'zones': {}}}))
        with pytest.raises(InferenceCapacityError):
            await manager.activate_camera("cam2", DictConfig({'vision': {'source': 's2', 'zones': {}}}))
        # cam2 NO queda registrada-pero-sin-arrancar; cam1 sigue viva.
        assert "cam2" not in manager.cameras
        assert manager.cameras["cam1"].state.is_running


@pytest.mark.asyncio
async def test_get_inference_status_reports_set_count_cap(manager):
    manager.max_inference_cameras = 4
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        await manager.activate_camera("cam1", DictConfig({'vision': {'source': 's1', 'zones': {}}}))
        await manager.activate_camera("cam2", DictConfig({'vision': {'source': 's2', 'zones': {}}}))

        st = manager.get_inference_status()
        assert set(st["inferring"]) == {"cam1", "cam2"}
        assert st["count"] == 2
        assert st["cap"] == 4
        assert st["capacity_used"] == 0.5


def test_inference_status_no_cap_default(manager):
    """Default sin tope: cap None, capacity_used None."""
    st = manager.get_inference_status()
    assert st["cap"] is None
    assert st["capacity_used"] is None
    assert st["count"] == 0


@pytest.mark.asyncio
async def test_activate_camera_keeps_others_alive(manager):
    cfg1 = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    cfg2 = DictConfig({'vision': {'source': 's2', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)

        await manager.activate_camera("cam1", cfg1)
        await manager.activate_camera("cam2", cfg2)
        assert "cam1" in manager.cameras and "cam2" in manager.cameras
        assert manager.cameras["cam1"].state.is_running
        assert manager.cameras["cam2"].state.is_running


@pytest.mark.asyncio
async def test_activate_camera_idempotent_same_source(manager):
    cfg = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)

        await manager.activate_camera("cam1", cfg)
        first = manager.cameras["cam1"]
        await manager.activate_camera("cam1", cfg)
        assert manager.cameras["cam1"] is first


# ---- watchdog / render (sin cambios de comportamiento) ----------------


@pytest.mark.asyncio
async def test_sweep_idle_disables_render_without_mjpeg(manager, mock_broadcaster):
    mock_broadcaster.subscribed_cameras.return_value = ["cam1"]  # SSE activo, sin MJPEG
    cfg = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        await manager.activate_camera("cam1", cfg)
        assert manager.cameras["cam1"].state.render_enabled is True

        assert await manager._sweep_idle(now=1000.0) == []
        assert manager.cameras["cam1"].state.render_enabled is True

        disabled = await manager._sweep_idle(now=1000.0 + manager.idle_timeout_s + 1)
        assert disabled == ["cam1"]
        assert manager.cameras["cam1"].state.is_running
        assert manager.cameras["cam1"].state.render_enabled is False
        assert manager.cameras["cam1"].state.latest_frame_raw is None


@pytest.mark.asyncio
async def test_mjpeg_consumer_keeps_render_on_then_watchdog_disables(manager, mock_broadcaster):
    mock_broadcaster.subscribed_cameras.return_value = []
    cfg = DictConfig({'vision': {'source': 's1', 'zones': {}}})
    with patch('src.vision.application.services.multi_camera.VisionApplicationBuilder') as MockBuilder:
        _patch_builder(MockBuilder)
        await manager.activate_camera("cam1", cfg)

        manager.add_mjpeg_consumer("cam1")
        assert await manager._sweep_idle(now=10_000.0) == []
        assert manager.cameras["cam1"].state.render_enabled is True

        manager.remove_mjpeg_consumer("cam1")
        await manager._sweep_idle(now=20_000.0)
        disabled = await manager._sweep_idle(now=20_000.0 + manager.idle_timeout_s + 1)
        assert disabled == ["cam1"]
        assert manager.cameras["cam1"].state.is_running
        assert manager.cameras["cam1"].state.render_enabled is False
