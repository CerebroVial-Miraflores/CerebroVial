"""Manager for multiple independent camera pipelines.

Caso B (acoplamiento `application/` → `presentation/`) roto en Fase 5e:
`OpenCVVisualizer` ya NO se importa acá. En su lugar, el `CameraInstance`
acepta un `FrameRenderer` opcional (Protocol del dominio); el concrete
visualizer se inyecta desde fuera (Fase 6 lo cablea en `presentation/`).
Default `None` → no se anota el frame; `latest_frame_processed` queda igual
al raw.

Logging por archivo (§6.3 / §12).
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import DictConfig

from ...domain.protocols import FrameRenderer
from ...infrastructure.broadcast.realtime_broadcaster import RealtimeBroadcaster
from ..aggregators.async_aggregator import AsyncTrafficAggregator
from ..builders.pipeline_builder import VisionApplicationBuilder, create_detector
from .camera_scheduler import CameraScheduler

logger = logging.getLogger(__name__)


@dataclass
class CameraState:
    camera_id: str
    config: DictConfig
    pipeline: Any  # AsyncVisionPipeline
    is_running: bool = False
    # Separación muestreo/render (B1 Paso 0). El MUESTREO (loop del pipeline +
    # aggregator.flush()→publish) es permanente y NO depende de este flag. El
    # RENDER MJPEG (escritura de `latest_frame_*`, on-demand, visual) sí: solo
    # corre con `render_enabled=True`. Default True para preservar el
    # comportamiento de hoy (render activo al activar). Lo prende
    # `add_mjpeg_consumer` y lo apaga el watchdog tras quedar sin consumidor MJPEG.
    render_enabled: bool = True
    latest_frame_raw: Optional[Any] = None
    latest_frame_processed: Optional[Any] = None
    # Último `(FrameAnalysis, width, height)` con detección, servido al overlay del
    # front por `GET /detections/{id}/latest` (Fase 4 Mitad A, D-019). Lo escribe el
    # scheduler en `_route` UNGATED por `render_enabled`: la inferencia corre
    # permanente y el overlay sobre el video HLS debe consumir las cajas aunque
    # nadie consuma el MJPEG (con el swap a HLS directo no hay consumidor MJPEG, así
    # que el watchdog deja `render_enabled=False`). None hasta el primer frame con
    # detección. Guarda las dims del frame de inferencia para normalizar a [0,1].
    latest_detections: Optional[Any] = None
    renderer: Optional[FrameRenderer] = None  # Inyectado opcionalmente; Fase 6 lo cablea.
    # Conservado para drenar TrafficData y publicar al broadcaster desde el
    # coroutine. None si persistence.enabled=False (modo dev/test sin agregador).
    aggregator: Optional[AsyncTrafficAggregator] = None
    # Detector YOLO compartido del manager (B1 Paso 4: inyectado, mismo singleton
    # en todas las cámaras). Referencia conservada solo para limpiar el puntero en
    # `remove_camera`; el modelo NO se libera por-cámara (lo usan las demás) — se
    # libera una vez en `MultiCameraManager.shutdown()`.
    detector: Optional[Any] = None
    # Consumidores MJPEG activos del feed `/video/{id}` (C1). El SSE se cuenta
    # vía broadcaster.subscribed_cameras(); el MJPEG no tiene registro propio,
    # así que lo lleva el generador de video.py (inc al entrar, dec en finally).
    mjpeg_consumers: int = 0
    # B1 Paso 1b: si la cámara corre por el scheduler (captura threaded + inferencia
    # secuencial a 1 Hz), acá vive su instancia; None = path viejo (pipeline.run()).
    scheduler: Optional[Any] = None  # CameraScheduler
    # Estado de sensor del último tick del scheduler (D-018, NULL-con-motivo) +
    # edad del último frame. Solo se pueblan en cámaras scheduled; health los lee
    # (seam §3, opción ii). None en el path viejo.
    sensor_status: Optional[str] = None
    last_frame_age_seconds: Optional[float] = None


class CameraInstance:
    """Encapsulates a camera with its pipeline, optional renderer and aggregator."""

    def __init__(
        self,
        camera_id: str,
        config: DictConfig,
        builder: VisionApplicationBuilder,
        detector: Any,
        renderer: Optional[FrameRenderer] = None,
    ) -> None:
        # B1 Paso 4: el detector SIEMPRE se inyecta desde el manager (el singleton
        # compartido). Es requerido: ya no hay fallback de construcción propia — el
        # path viejo murió, así que un `CameraInstance` sin detector es un error de
        # programación, no un modo de operación (falla fuerte y temprano en vez de
        # cargar un modelo fantasma en silencio).
        pipeline = builder.build_pipeline(detector=detector)
        components = builder.get_components()
        self.state = CameraState(
            camera_id=camera_id,
            config=config,
            pipeline=pipeline,
            renderer=renderer,
            aggregator=components.get("aggregator"),
            detector=components.get("detector"),
        )



class MultiCameraManager:
    """
    Manages multiple camera pipelines simultaneously.
    Each camera runs in its own set of threads.
    """
    
    def __init__(
        self,
        broadcaster: RealtimeBroadcaster,
        idle_timeout_s: float = 45.0,
        watchdog_interval_s: float = 10.0,
    ) -> None:
        self.cameras: dict[str, CameraInstance] = {}
        self.broadcaster = broadcaster
        self._tasks: dict[str, asyncio.Task] = {}
        # Auto-liberación por timeout sin consumidores (C1, E4). `_idle_since`
        # marca, por cámara, el monotonic en que se quedó sin consumidores;
        # se limpia al reaparecer alguno y dispara la baja al superar el timeout.
        self.idle_timeout_s = idle_timeout_s
        self.watchdog_interval_s = watchdog_interval_s
        self._idle_since: dict[str, float] = {}
        self._watchdog_task: Optional[asyncio.Task] = None
        # B1 Paso 2: singletons compartidos por las cámaras del scheduler (D-018:
        # UN modelo YOLO + UN worker de inferencia por instancia de edge). Lazy:
        # se crean con la primera cámara scheduled y se liberan una vez en
        # `shutdown()`. Las cámaras del path viejo NO los usan (detector propio).
        self._shared_detector: Optional[Any] = None
        self._shared_executor: Optional[ThreadPoolExecutor] = None
        # Device de inferencia resuelto UNA vez en el arranque del server
        # (`select_device()` en run_server.py) e inyectado al detector compartido
        # cuando se crea lazy. `None` → el detector aplica su fallback cuda→mps→cpu
        # (ej. tests que instancian el manager sin pasar por el arranque).
        self.inference_device: Optional[str] = None

    def _shared_detector_for(self, config: DictConfig) -> Any:
        """Detector YOLO singleton de las cámaras del scheduler (D-018: UN modelo
        por instancia). Lazy: se construye con la config de la primera cámara
        scheduled (todas usan el mismo yolo11n del alta on-demand). Se libera una
        vez en `shutdown()`.

        INVARIANTE DE CARRERA: race-free SOLO si este helper permanece síncrono y
        sin `await` entre el check (`is None`) y el set. Hoy lo es (`create_detector`
        es síncrona y los callers lo invocan sin ceder el loop), así que el event
        loop no puede interleavear dos creaciones. Si un refactor vuelve esto async
        o mete un `await` en el medio, hace falta un lock — si no, doble creación.
        """
        if self._shared_detector is None:
            self._shared_detector = create_detector(
                config.vision, device=self.inference_device
            )
        return self._shared_detector

    def _shared_infer_executor(self) -> ThreadPoolExecutor:
        """Executor de inferencia singleton: UN worker que serializa TODA la
        inferencia del scheduler. Es el invariante de concurrencia del modelo
        compartido y, por el Benchmark §B.3 (cv2 colapsa torch a 1 thread; forzar
        multi-thread oversubscribe), también el óptimo. Lazy; se apaga una vez en
        `shutdown()`.

        INVARIANTE DE CARRERA: race-free SOLO si este helper permanece síncrono y
        sin `await` entre el check (`is None`) y el set. Si un refactor lo vuelve
        async o mete un `await` en el medio, hace falta un lock — si no, dos
        executors creados en paralelo.
        """
        if self._shared_executor is None:
            self._shared_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="vision-infer-shared"
            )
        return self._shared_executor

    def add_camera(
        self,
        camera_id: str,
        config: DictConfig,
        renderer: Optional[FrameRenderer] = None,
    ) -> CameraInstance:
        """Registra una cámara nueva.

        `renderer` (opcional) anota los frames procesados; default None →
        `latest_frame_processed` queda igual al raw. Fase 6 inyecta el
        OpenCVVisualizer adaptado al Protocol `FrameRenderer` desde
        `presentation/`.
        """
        if camera_id in self.cameras:
            raise ValueError(f"Camera {camera_id} already exists")

        # Inject camera_id into zones config (mutation in-place sobre el
        # OmegaConf entrante; el caller controla la mutabilidad).
        if 'zones' in config.vision:
            for zone_id, zone_cfg in config.vision.zones.items():
                if isinstance(zone_cfg, dict) and 'camera_id' not in zone_cfg:
                    zone_cfg['camera_id'] = camera_id

        builder = VisionApplicationBuilder(config)
        # B1 Paso 4: el detector compartido se inyecta SIEMPRE (scheduler único;
        # ya no hay path viejo ni ids fuera del scheduler). Todas las cámaras
        # comparten el mismo singleton del manager.
        shared_detector = self._shared_detector_for(config)
        camera = CameraInstance(
            camera_id, config, builder, detector=shared_detector, renderer=renderer
        )

        self.cameras[camera_id] = camera
        logger.info("Added camera: %s", camera_id)
        return camera

    async def activate_camera(
        self,
        camera_id: str,
        config: DictConfig,
        renderer: Optional[FrameRenderer] = None,
    ) -> CameraInstance:
        """Alta on-demand de una cámara.

        - Idempotente sobre el mismo `source`: si la cámara ya existe y corre con
          la misma fuente, no recrea (devuelve la instancia viva).
        - Si la fuente cambió, baja+alta.

        B1 Paso 0: se retiró el sweep single-slot. Activar una cámara YA NO baja
        las demás; varias pueden convivir. La garantía de "un solo YOLO vivo"
        (antes C1/D2) deja de imponerse acá y pasa al scheduler único con modelo
        compartido (D-018, Paso 1).
        """
        new_source = config.vision.source
        existing = self.cameras.get(camera_id)
        if existing is not None:
            same_source = existing.state.config.vision.source == new_source
            if same_source and existing.state.is_running:
                logger.info("Camera %s ya activa con la misma fuente; no-op", camera_id)
                return existing
            await self.remove_camera(camera_id)

        camera = self.add_camera(camera_id, config, renderer=renderer)
        await self.start_camera(camera_id)
        return camera

    async def start_camera(self, camera_id: str) -> None:
        """Starts processing for a camera."""
        if camera_id not in self.cameras:
            raise ValueError(f"Camera {camera_id} not found")

        camera = self.cameras[camera_id]
        if camera.state.is_running:
            logger.info("Camera %s already running", camera_id)
            return

        camera.state.is_running = True
        # B1 Paso 4: scheduler único. TODAS las cámaras corren por el
        # `CameraScheduler` (captura threaded + inferencia secuencial a 1 Hz). El
        # scheduler recibe el executor de inferencia COMPARTIDO (max_workers=1
        # global) → toda la inferencia del modelo compartido se serializa en un
        # solo worker. `_owns_executor=False` (porque se inyecta) hace que
        # `scheduler.stop()` NO lo apague: lo apaga el `shutdown()` del manager.
        scheduler = CameraScheduler(
            camera, self.broadcaster, infer_executor=self._shared_infer_executor()
        )
        camera.state.scheduler = scheduler
        task = asyncio.create_task(scheduler.run())
        logger.info("Started camera (scheduler): %s", camera_id)
        self._tasks[camera_id] = task

    async def stop_camera(self, camera_id: str) -> None:
        """Stops a specific camera."""
        if camera_id not in self.cameras:
            return

        camera = self.cameras[camera_id]
        camera.state.is_running = False
        # B1 Paso 4: scheduler único. La cámara dueña su teardown (captura + B2:
        # aggregator force_flush+stop). El `pipeline` se construyó pero NUNCA se
        # arrancó (`run()`/`start()`), así que no hay threads del pipeline que
        # parar — el `ThreadedCapture` del scheduler es quien libera el source.
        # `scheduler is None` solo si la cámara se agregó pero nunca se arrancó.
        if camera.state.scheduler is not None:
            camera.state.scheduler.stop()
            camera.state.scheduler = None

        if camera_id in self._tasks:
            self._tasks[camera_id].cancel()
            try:
                await self._tasks[camera_id]
            except asyncio.CancelledError:
                pass
            del self._tasks[camera_id]

        logger.info("Stopped camera: %s", camera_id)

    async def remove_camera(self, camera_id: str) -> None:
        """Baja dinámica (C1): para la cámara y la saca del registro.

        B1 Paso 4: ya NO libera el detector — todas las cámaras comparten el
        singleton del manager, así que liberarlo acá mataría el modelo de las
        demás (F2). El compartido se libera una sola vez en `shutdown()`. Acá solo
        se sueltan las referencias de esta cámara. Idempotente: si la cámara no
        existe, no-op.
        """
        if camera_id not in self.cameras:
            return

        await self.stop_camera(camera_id)

        camera = self.cameras.pop(camera_id, None)
        if camera is not None:
            # No se libera el detector: es el singleton compartido (lo usan las
            # demás). Solo soltamos las referencias de esta cámara.
            camera.state.detector = None
            camera.state.latest_frame_raw = None
            camera.state.latest_frame_processed = None
        logger.info("Removed camera: %s", camera_id)

    async def start_all(self):
        """Starts all registered cameras."""
        tasks = [self.start_camera(cam_id) for cam_id in self.cameras.keys()]
        await asyncio.gather(*tasks)

    async def stop_all(self):
        """Stops all cameras."""
        tasks = [self.stop_camera(cam_id) for cam_id in list(self.cameras.keys())]
        await asyncio.gather(*tasks)

    async def shutdown(self) -> None:
        """Teardown del manager (B1 Paso 4): para todas las cámaras y libera los
        singletons compartidos UNA vez — el detector YOLO y el executor de
        inferencia. Es la ÚNICA vía de release del detector compartido (las
        cámaras no lo liberan en `remove_camera` porque lo comparten). En
        contenedor el stop suele ser SIGKILL y esto no corre; queda para shutdown
        graceful, tests y deploys no-contenedor."""
        await self.stop_all()
        if self._shared_detector is not None and hasattr(self._shared_detector, "release"):
            self._shared_detector.release()
        self._shared_detector = None
        if self._shared_executor is not None:
            self._shared_executor.shutdown(wait=False)
            self._shared_executor = None

    def get_status(self) -> dict:
        """Returns status of all cameras."""
        return {
            cam_id: {
                "running": cam.state.is_running,
                "source": cam.state.config.vision.source,
                "zones": list(cam.state.config.vision.zones.keys()) if cam.state.config.vision.zones else []
            }
            for cam_id, cam in self.cameras.items()
        }

    def get_latest_frame(self, camera_id: str, processed: bool = False):
        """Returns the latest frame for a camera."""
        if camera_id not in self.cameras:
            return None
        
        camera = self.cameras[camera_id]
        if processed:
            return camera.state.latest_frame_processed
        return camera.state.latest_frame_raw

    def get_latest_detections(self, camera_id: str):
        """Último `(FrameAnalysis, width, height)` con detección de la cámara, o
        None si no existe o aún no infirió (Fase 4 Mitad A, D-019).

        Lo escribe el scheduler en cada tick ungated por `render_enabled` (la
        inferencia es permanente; el overlay HLS no consume el MJPEG). La
        normalización a [0,1] y el shaping JSON los hace el serializador de
        presentation al servir `GET /detections/{id}/latest`."""
        camera = self.cameras.get(camera_id)
        if camera is None:
            return None
        return camera.state.latest_detections

    # ---- Conteo de consumidores + auto-liberación (C1, E4) ------------

    def add_mjpeg_consumer(self, camera_id: str) -> None:
        """Registra un consumidor MJPEG activo del feed `/video/{id}`.

        Reenciende el render (B1 Paso 0): si el watchdog lo había apagado por
        ociosidad, un nuevo espectador MJPEG lo vuelve a prender. El SSE NO pasa
        por acá: consume el agregado, no frames.
        """
        camera = self.cameras.get(camera_id)
        if camera is not None:
            camera.state.mjpeg_consumers += 1
            camera.state.render_enabled = True
            self._idle_since.pop(camera_id, None)

    def remove_mjpeg_consumer(self, camera_id: str) -> None:
        """Da de baja un consumidor MJPEG (llamado en el `finally` del generador)."""
        camera = self.cameras.get(camera_id)
        if camera is not None and camera.state.mjpeg_consumers > 0:
            camera.state.mjpeg_consumers -= 1

    def _has_mjpeg_consumer(self, camera_id: str) -> bool:
        """True si la cámara tiene al menos un consumidor MJPEG.

        B1 Paso 0: el watchdog decide apagar el RENDER, que solo lo consume el
        MJPEG. El SSE recibe el agregado (no usa frames) y es irrelevante para
        esta decisión; por eso NO se consulta `broadcaster.subscribed_cameras()`.
        """
        camera = self.cameras.get(camera_id)
        if camera is None:
            return False
        return camera.state.mjpeg_consumers > 0

    async def _sweep_idle(self, now: float) -> list[str]:
        """Una pasada del watchdog: apaga el RENDER MJPEG ocioso hace > timeout.

        B1 Paso 0: la acción ya NO es bajar la cámara (`remove_camera`), sino
        apagar solo su render. El MUESTREO (is_running, loop del pipeline,
        aggregator→broadcaster) sobrevive: la cámara sigue contando y
        persistiendo. Ociosidad medida por MJPEG (no SSE), ver `_has_mjpeg_consumer`.

        Aislada del loop para testearla con un `now` controlado (sin wall-clock).
        Devuelve los camera_id cuyo render se apagó en esta pasada.
        """
        disabled: list[str] = []
        for camera_id in list(self.cameras.keys()):
            camera = self.cameras[camera_id]
            if self._has_mjpeg_consumer(camera_id):
                self._idle_since.pop(camera_id, None)
                continue
            if not camera.state.render_enabled:
                # Render ya apagado: nada que hacer, no re-marcar idle.
                self._idle_since.pop(camera_id, None)
                continue
            since = self._idle_since.get(camera_id)
            if since is None:
                self._idle_since[camera_id] = now
            elif now - since >= self.idle_timeout_s:
                logger.info(
                    "Camera %s sin consumidor MJPEG hace %.0fs; apagando render "
                    "(el muestreo sigue)",
                    camera_id,
                    now - since,
                )
                camera.state.render_enabled = False
                camera.state.latest_frame_raw = None
                camera.state.latest_frame_processed = None
                self._idle_since.pop(camera_id, None)
                disabled.append(camera_id)
        return disabled

    async def _watchdog_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.watchdog_interval_s)
                await self._sweep_idle(time.monotonic())
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - el watchdog no debe matar el server
            logger.exception("Watchdog de auto-liberación falló")

    def start_watchdog(self) -> None:
        """Arranca el task de auto-liberación (idempotente)."""
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())
            logger.info(
                "Watchdog de auto-liberación iniciado (timeout=%.0fs, intervalo=%.0fs)",
                self.idle_timeout_s,
                self.watchdog_interval_s,
            )

    async def stop_watchdog(self) -> None:
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None


