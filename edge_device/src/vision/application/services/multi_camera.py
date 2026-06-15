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
from .batch_inference import BatchInferenceWorker
from .queue_push_producer import QueuePushProducer

logger = logging.getLogger(__name__)


class InferenceCapacityError(Exception):
    """El contenedor de inferencia llegó a su tope (`max_inference_cameras`).

    Se levanta al intentar arrancar una cámara nueva cuando el set que ya infiere
    está lleno. La capa de presentación la mapea a HTTP 409 (NO se degrada en
    silencio: el tope real lo fija el operador según el harness de carga)."""


@dataclass
class CameraState:
    camera_id: str
    config: DictConfig
    # Topología B (15Hz): la cámara ya NO arma un AsyncVisionPipeline ni un
    # CameraScheduler. Tiene una `source` (la consume el producer-push) y una
    # `post_chain` (la registra el batch worker para el demux). `pipeline` queda
    # como None por compat de health (guard `if state.pipeline`).
    pipeline: Any = None
    source: Any = None       # FrameProducer (lo lee el QueuePushProducer)
    post_chain: Any = None   # FrameProcessor (Tracking→...→Aggregation) para el demux
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
    # Topología B: el producer-push per-cámara (corre source.read() en daemon thread
    # y empuja a la cola compartida del batch worker). None = cámara no arrancada.
    producer: Optional[Any] = None  # QueuePushProducer
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
        renderer: Optional[FrameRenderer] = None,
    ) -> None:
        # Topología B: NO se arma el AsyncVisionPipeline (head de detección). La
        # cámara aporta su `source` (la lee el producer-push) y su `post_chain`
        # (Tracking→Speed→Zone→Aggregation, la registra el batch worker). La
        # detección es central (un detector compartido en el worker), así que el
        # CameraInstance ya NO recibe detector.
        builder.build_source()
        post_chain = builder.build_post_chain()
        self.state = CameraState(
            camera_id=camera_id,
            config=config,
            source=builder.source,
            post_chain=post_chain,
            renderer=renderer,
            aggregator=builder.aggregator,
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
        # Topología B: worker de inferencia en lote singleton (uno por instancia de
        # edge / una GPU). Lazy: se crea con la 1ª cámara JUNTO al detector (no al
        # boot — respeta on-demand C1/D1 y el probe-not-load de FASE 1). Se apaga
        # en shutdown().
        self._batch_worker: Optional[BatchInferenceWorker] = None
        # Device de inferencia resuelto UNA vez en el arranque del server
        # (`select_device()` en run_server.py) e inyectado al detector compartido
        # cuando se crea lazy. `None` → el detector aplica su fallback cuda→mps→cpu
        # (ej. tests que instancian el manager sin pasar por el arranque).
        self.inference_device: Optional[str] = None
        # Knobs de config a nivel de instancia (run_server los setea desde cfg+env).
        # análisis a 25fps (nativo del HLS de Claro; pacea con `-re` en FullDecodeSource
        # → worker alimentado parejo, ByteTrack confirma), imgsz 640 (= nativo de
        # ultralytics). analyze_fps se inyecta a la fuente full-decode; imgsz al worker.
        # Override por env VISION_ANALYZE_FPS (p.ej. bajarlo en Docker-CPU con N cámaras).
        self.analyze_fps: int = 25
        self.imgsz: int = 640
        # Tope del contenedor de inferencia (cámaras infiriendo simultáneas). None =
        # sin tope efectivo (default → no rompe despliegues actuales). El operador lo
        # fija según el harness de carga (cámaras-por-contenedor a 15Hz).
        self.max_inference_cameras: Optional[int] = None

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

    def _batch_worker_for(self, detector) -> BatchInferenceWorker:
        """Batch worker singleton (topología B): lazy, se crea con la 1ª cámara
        JUNTO al detector (cuidado (b): NO al boot — el detector es lazy y FASE 1
        dejó el boot en probe-not-load). Comparte el executor de inferencia y el
        broadcaster. Arranca su task asyncio acá (hay event loop: start_camera es
        async). Se apaga en `shutdown()`.

        INVARIANTE: este helper es síncrono (sin `await` entre el check y el set),
        así el event loop no interleavea dos creaciones.
        """
        if self._batch_worker is None:
            self._batch_worker = BatchInferenceWorker(
                detector, self._shared_infer_executor(), self.broadcaster,
                imgsz=self.imgsz,
            )
            self._batch_worker.start()
        return self._batch_worker

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

        # Knob de instancia: inyecta analyze_fps en el cfg de la cámara para que
        # build_source lo pase a FullDecodeSource (cadencia de análisis; no toca el
        # stream que ve el browser). Default 15 = comportamiento actual.
        config.vision.analyze_fps = self.analyze_fps

        builder = VisionApplicationBuilder(config)
        # Crea el detector compartido (lazy) acá, ANTES de que arranque el worker
        # en start_camera (cuidado (b): el orden es detector → worker → registro →
        # producer). El detector NO va al CameraInstance (la detección es central
        # en el worker); el instance solo arma source + post_chain.
        self._shared_detector_for(config)
        camera = CameraInstance(camera_id, config, builder, renderer=renderer)

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
        try:
            await self.start_camera(camera_id)
        except InferenceCapacityError:
            # No dejar la cámara registrada-pero-sin-arrancar si el tope la rechazó.
            await self.remove_camera(camera_id)
            raise
        return camera

    async def start_camera(self, camera_id: str) -> None:
        """Starts processing for a camera."""
        if camera_id not in self.cameras:
            raise ValueError(f"Camera {camera_id} not found")

        camera = self.cameras[camera_id]
        if camera.state.is_running:
            logger.info("Camera %s already running", camera_id)
            return

        # Tope del contenedor: rechazar una cámara NUEVA si el set que ya infiere
        # llegó al cap. No se degrada en silencio (error claro → 409 en presentación).
        if self.max_inference_cameras is not None:
            running = sum(1 for c in self.cameras.values() if c.state.is_running)
            if running >= self.max_inference_cameras:
                raise InferenceCapacityError(
                    f"contenedor lleno: tope {self.max_inference_cameras} cámaras "
                    f"infiriendo ({running} activas)"
                )

        camera.state.is_running = True
        # Topología B — ORDEN ESTRICTO (cuidado (b)): detector creado → worker
        # cableado con el detector → post-chain registrada → RECIÉN AHÍ arranca el
        # producer. Así ningún frame fluye antes de que el worker tenga detector +
        # post-chain (si no, frames encolados sin destino / sin modelo).
        detector = self._shared_detector_for(camera.state.config)  # idempotente
        worker = self._batch_worker_for(detector)
        worker.register(
            camera_id,
            camera.state.post_chain,
            aggregator=camera.state.aggregator,
            state=camera.state,
        )
        loop = asyncio.get_running_loop()
        producer = QueuePushProducer(
            camera_id,
            camera.state.source,
            worker,
            camera.state,
            broadcaster=self.broadcaster,
            loop=loop,
        )
        camera.state.producer = producer
        producer.start()
        logger.info("Started camera (batched): %s", camera_id)

    async def stop_camera(self, camera_id: str) -> None:
        """Stops a specific camera."""
        if camera_id not in self.cameras:
            return

        camera = self.cameras[camera_id]
        camera.state.is_running = False
        # Topología B teardown. ORDEN: desregistrar del worker (que el demux deje de
        # rutear a esta post-chain) → parar el producer (corta el push + libera el
        # source) → teardown del aggregator (force_flush+stop, antes lo hacía
        # scheduler.stop()). `producer is None` solo si la cámara se agregó pero
        # nunca se arrancó.
        if self._batch_worker is not None:
            self._batch_worker.unregister(camera_id)
        if camera.state.producer is not None:
            camera.state.producer.stop()
            camera.state.producer = None
        if camera.state.aggregator is not None:
            try:
                camera.state.aggregator.force_flush()  # cierra la ventana en curso
                camera.state.aggregator.stop()         # el worker thread no queda colgado
            except Exception:
                logger.exception("aggregator teardown falló")

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
        # Topología B: apagar el batch worker (su task asyncio) antes de soltar el
        # detector compartido que consume.
        if self._batch_worker is not None:
            await self._batch_worker.stop()
            self._batch_worker = None
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

    def get_inference_status(self) -> dict:
        """Estado del contenedor de inferencia (lo consume la UX del front).

        `inferring` = ids de las cámaras corriendo; `count` = cuántas; `cap` = tope
        (None = sin tope); `capacity_used` = count/cap (None si no hay tope)."""
        inferring = [cid for cid, c in self.cameras.items() if c.state.is_running]
        cap = self.max_inference_cameras
        return {
            "inferring": inferring,
            "count": len(inferring),
            "cap": cap,
            "capacity_used": (len(inferring) / cap) if cap else None,
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


