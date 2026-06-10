"""Unit tests de CameraScheduler (B1 1b).

Cubre: no-inferir-sin-frame-fresco, no-colgar ante fuente muerta, honestidad de
ventana (force_flush en la transición fresco→sin-señal), teardown B2, y la
equivalencia del ruteo (mismo TrafficData al broadcaster que el pipeline viejo).
"""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.vision.application.processors.smart_detection import DEFAULT_IMGSZ
from src.vision.application.services.camera_scheduler import CameraScheduler
from src.vision.domain.entities import Frame, FrameAnalysis, FrameSnapshot
from src.vision.domain.value_objects import SensorStatus
from src.vision.infrastructure.sources.threaded_capture import ThreadedCapture


# ---- Fakes ------------------------------------------------------------

class _FakeCapture:
    """LiveFrameSource fake: snapshot() controlable (valor fijo o callable)."""

    def __init__(self, snap) -> None:
        self._snap = snap
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def snapshot(self) -> FrameSnapshot:
        return self._snap() if callable(self._snap) else self._snap

    def stop(self) -> None:
        self.stopped = True


def _make_camera(chain, aggregator, *, render_enabled=False, renderer=None, imgsz=None):
    # config real (DictConfig): el scheduler relee cfg.vision.model.imgsz per-tick.
    model = {} if imgsz is None else {"imgsz": imgsz}
    config = OmegaConf.create({"vision": {"model": model}})
    state = SimpleNamespace(
        camera_id="camX",
        config=config,
        pipeline=SimpleNamespace(source=MagicMock(), processor_chain=chain),
        aggregator=aggregator,
        is_running=True,
        render_enabled=render_enabled,
        renderer=renderer,
        latest_frame_raw=None,
        latest_frame_processed=None,
        sensor_status=None,
        last_frame_age_seconds=None,
    )
    return SimpleNamespace(state=state)


def _frame(fid=0):
    return Frame(id=fid, timestamp=0.0, image=np.zeros((2, 2, 3), dtype=np.uint8))


def _analysis(fid=0, detection_ran=True):
    return FrameAnalysis(
        frame_id=fid, timestamp=0.0, vehicles=[], unique_vehicles=0,
        zones={}, detection_ran=detection_ran,
    )


def _fresh(frame=None, age=0.5):
    return FrameSnapshot(frame=frame or _frame(), age_seconds=age, live=True)


def _stale(frame=None, age=10.0):
    return FrameSnapshot(frame=frame or _frame(), age_seconds=age, live=True)


def _dead():
    return FrameSnapshot(frame=None, age_seconds=None, live=False)


def _scheduler(camera, capture, broadcaster=None):
    return CameraScheduler(
        camera,
        broadcaster or _broadcaster(),
        capture=capture,
        fresh_threshold_s=2.5,
        tick_interval_s=0.01,
    )


def _broadcaster():
    b = MagicMock()
    b.publish = AsyncMock()
    return b


# ---- No-inferir sin frame fresco -------------------------------------

@pytest.mark.asyncio
async def test_no_infiere_si_no_hay_frame_fuente_no_disponible():
    chain = MagicMock()
    agg = MagicMock()
    agg.flush.return_value = []
    cam = _make_camera(chain, agg)
    sch = _scheduler(cam, _FakeCapture(_dead()))

    await sch._tick(asyncio.get_running_loop())

    chain.process.assert_not_called()                 # NO inferencia
    assert cam.state.sensor_status == SensorStatus.FUENTE_NO_DISPONIBLE.value


@pytest.mark.asyncio
async def test_no_infiere_si_frame_viejo_sin_frame_fresco():
    chain = MagicMock()
    agg = MagicMock()
    agg.flush.return_value = []
    cam = _make_camera(chain, agg)
    sch = _scheduler(cam, _FakeCapture(_stale(age=10.0)))

    await sch._tick(asyncio.get_running_loop())

    chain.process.assert_not_called()
    assert cam.state.sensor_status == SensorStatus.SIN_FRAME_FRESCO.value


@pytest.mark.asyncio
async def test_infiere_y_publica_si_frame_fresco():
    frame = _frame(1)
    analysis = _analysis(1)
    chain = MagicMock()
    chain.process.return_value = analysis
    agg = MagicMock()
    agg.flush.return_value = ["td-1"]
    broadcaster = _broadcaster()
    cam = _make_camera(chain, agg)
    sch = _scheduler(cam, _FakeCapture(_fresh(frame=frame, age=0.5)), broadcaster)

    await sch._tick(asyncio.get_running_loop())

    chain.process.assert_called_once_with(frame, None)   # inferencia con prev=None
    broadcaster.publish.assert_awaited_once_with("td-1")  # drenó aggregator→SSE
    assert cam.state.sensor_status == SensorStatus.OK.value


# ---- imgsz: hot-reload per-tick + default de política -----------------

@pytest.mark.asyncio
async def test_imgsz_default_policy_cuando_ausente():
    """Sin imgsz en el cfg → el scheduler aplica la ÚNICA política DEFAULT_IMGSZ."""
    chain = MagicMock()
    chain.process.return_value = _analysis(1)
    agg = MagicMock()
    agg.flush.return_value = []
    cam = _make_camera(chain, agg)  # cfg sin model.imgsz
    sch = _scheduler(cam, _FakeCapture(_fresh()))

    await sch._tick(asyncio.get_running_loop())

    assert chain.imgsz == DEFAULT_IMGSZ


@pytest.mark.asyncio
async def test_hot_reload_imgsz_entre_ticks():
    """Cambiar cfg.vision.model.imgsz entre ticks → el próximo tick lo usa, sin
    reconstruir nada (imgsz va por-llamada)."""
    chain = MagicMock()
    chain.process.return_value = _analysis(1)
    agg = MagicMock()
    agg.flush.return_value = []
    cam = _make_camera(chain, agg, imgsz=320)
    sch = _scheduler(cam, _FakeCapture(_fresh()))
    loop = asyncio.get_running_loop()

    await sch._tick(loop)
    assert chain.imgsz == 320

    cam.state.config.vision.model.imgsz = 640  # mutación en caliente
    await sch._tick(loop)
    assert chain.imgsz == 640


# ---- Honestidad de ventana: force_flush en la transición -------------

@pytest.mark.asyncio
async def test_force_flush_solo_en_transicion_fresco_a_sin_senal():
    """En la transición OK→sin-señal se fuerza force_flush UNA vez (cierra ventana
    en la última señal). En sin-señal sostenido NO se vuelve a forzar."""
    frame = _frame(1)
    chain = MagicMock()
    chain.process.return_value = _analysis(1)
    agg = MagicMock()
    agg.flush.return_value = []
    agg.force_flush.return_value = ["td-cierre"]
    broadcaster = _broadcaster()
    cam = _make_camera(chain, agg)

    # snapshot conmutable: fresco → stale → stale.
    seq = [_fresh(frame=frame), _stale(), _stale()]
    box = {"i": 0}

    def next_snap():
        s = seq[box["i"]]
        box["i"] = min(box["i"] + 1, len(seq) - 1)
        return s

    sch = _scheduler(cam, _FakeCapture(next_snap), broadcaster)
    loop = asyncio.get_running_loop()

    await sch._tick(loop)  # fresco → OK
    await sch._tick(loop)  # transición → force_flush UNA vez
    await sch._tick(loop)  # sostenido → NO force_flush

    assert agg.force_flush.call_count == 1
    broadcaster.publish.assert_any_await("td-cierre")  # se publicó el cierre


# ---- No-colgar ante fuente muerta (loop real con ThreadedCapture) ----

@pytest.mark.asyncio
async def test_loop_no_se_cuelga_con_read_bloqueado():
    """Source cuyo read() bloquea para siempre: el scheduler tickea igual, 0
    inferencias, status FUENTE_NO_DISPONIBLE, y el loop termina al bajar is_running."""

    class _BlockingSource:
        def __init__(self):
            self._gate = threading.Event()
        def read(self):
            self._gate.wait()  # cuelga indefinidamente
            return None
        def release(self):
            self._gate.set()

    chain = MagicMock()
    agg = MagicMock()
    agg.flush.return_value = []
    cam = _make_camera(chain, agg)
    capture = ThreadedCapture(_BlockingSource())
    sch = CameraScheduler(cam, _broadcaster(), capture=capture, tick_interval_s=0.01)

    task = asyncio.create_task(sch.run())
    await asyncio.sleep(0.1)            # deja correr varios ticks
    assert not task.done()             # sigue vivo, no crasheó
    cam.state.is_running = False        # señal de fin

    # El loop debe terminar pronto (no colgarse en el read bloqueado).
    await asyncio.wait_for(task, timeout=2.0)

    chain.process.assert_not_called()  # nunca infirió
    assert cam.state.sensor_status == SensorStatus.FUENTE_NO_DISPONIBLE.value
    sch.stop()  # release() desbloquea el read colgado


@pytest.mark.asyncio
async def test_fuente_muerta_via_opencvsource_fakecapture():
    """Escenario nombrado en el plan: FakeCapture(isOpened=True, read=(False,None))
    vía create_source → OpenCVSource REAL → ThreadedCapture. read() devuelve None
    (EOF, loop off) → live=False → FUENTE_NO_DISPONIBLE, 0 inferencias, loop termina."""
    from src.vision.infrastructure.sources import create_source

    class _DeadCapture:
        def isOpened(self):
            return True
        def read(self):
            return False, None  # nunca entrega frame
        def set(self, *a, **k):
            return True
        def release(self):
            pass

    source = create_source("dead.mp4", loop=False, capture=_DeadCapture())
    chain = MagicMock()
    agg = MagicMock()
    agg.flush.return_value = []
    cam = _make_camera(chain, agg)
    capture = ThreadedCapture(source)
    sch = CameraScheduler(cam, _broadcaster(), capture=capture, tick_interval_s=0.01)

    task = asyncio.create_task(sch.run())
    await asyncio.sleep(0.1)
    cam.state.is_running = False
    await asyncio.wait_for(task, timeout=2.0)  # no se cuelga

    chain.process.assert_not_called()
    assert cam.state.sensor_status == SensorStatus.FUENTE_NO_DISPONIBLE.value
    sch.stop()


# ---- Teardown B2 ------------------------------------------------------

def test_stop_cierra_aggregator_y_captura_b2():
    chain = MagicMock()
    agg = MagicMock()
    cam = _make_camera(chain, agg)
    capture = _FakeCapture(_dead())
    sch = CameraScheduler(cam, _broadcaster(), capture=capture)

    sch.stop()

    assert capture.stopped is True
    agg.force_flush.assert_called_once()  # drena ventana en curso
    agg.stop.assert_called_once()         # B2: worker no queda colgado


# ---- Ruteo: muestreo + render (movido de test_multi_camera_manager) --
# B1 Paso 4: estas conductas las ejercía el path viejo a nivel manager
# (_run_camera_pipeline). Retirado el path viejo, son invariantes del `_route`
# del scheduler y se prueban acá, en su nivel. El viejo test de equivalencia
# (scheduler vs pipeline viejo) se retiró: ya no hay con qué equivaler.

@pytest.mark.asyncio
async def test_route_render_off_muestreo_sigue():
    """Con render_enabled=False el `_route` NO escribe latest_frame_* (render off),
    PERO el muestreo sigue: drena el aggregator y publica al broadcaster. Es la
    separación muestreo/render del Paso 0, ahora a nivel scheduler."""
    chain = MagicMock()
    agg = MagicMock()
    agg.flush.return_value = ["td-1"]
    cam = _make_camera(chain, agg, render_enabled=False)
    broadcaster = _broadcaster()
    sch = _scheduler(cam, _FakeCapture(_dead()), broadcaster)

    await sch._route(_frame(), _analysis(detection_ran=True))

    broadcaster.publish.assert_awaited_once_with("td-1")  # muestreo ON
    assert cam.state.latest_frame_raw is None             # render OFF
    assert cam.state.latest_frame_processed is None


@pytest.mark.asyncio
async def test_route_render_on_escribe_mjpeg_y_publica():
    """Con render_enabled=True el `_route` publica el TD del aggregator Y escribe el
    frame procesado vía renderer. Integra muestreo + render en un tick (convertido
    del viejo test de equivalencia, conservando su mitad nueva)."""
    chain = MagicMock()
    agg = MagicMock()
    agg.flush.return_value = ["TD"]
    renderer = MagicMock()
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    renderer.render.return_value = img
    cam = _make_camera(chain, agg, render_enabled=True, renderer=renderer)
    broadcaster = _broadcaster()
    sch = _scheduler(cam, _FakeCapture(_dead()), broadcaster)

    await sch._route(_frame(), _analysis(detection_ran=True))

    broadcaster.publish.assert_awaited_once_with("TD")    # muestreo
    renderer.render.assert_called_once()                  # render ON
    assert cam.state.latest_frame_raw is not None
    assert cam.state.latest_frame_processed is img         # frame anotado escrito


@pytest.mark.asyncio
async def test_route_persistencia_visual_boxes_sin_doble_conteo():
    """En skip frames (detection_ran=False) el render persiste los boxes del último
    frame inferido, PERO esos boxes NUNCA tocan métricas: el `_route` solo hace
    flush, nunca add (movido de test_multi_camera_manager)."""
    chain = MagicMock()
    agg = MagicMock()
    agg.flush.return_value = []
    renderer = MagicMock()
    renderer.render.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
    cam = _make_camera(chain, agg, render_enabled=True, renderer=renderer)
    sch = _scheduler(cam, _FakeCapture(_dead()))

    analysis_detect = _analysis(fid=0, detection_ran=True)
    analysis_skip = _analysis(fid=1, detection_ran=False)
    await sch._route(_frame(0), analysis_detect)
    await sch._route(_frame(1), analysis_skip)

    # En el skip frame el renderer recibió el análisis PERSISTIDO (detect), no el vacío.
    rendered = [call.args[1] for call in renderer.render.call_args_list]
    assert rendered == [analysis_detect, analysis_detect]
    # Los boxes persistidos NUNCA tocan métricas.
    agg.add.assert_not_called()
