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
from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import DictConfig

from ...domain.protocols import FrameRenderer
from ...infrastructure.broadcast.realtime_broadcaster import RealtimeBroadcaster
from ..aggregators.async_aggregator import AsyncTrafficAggregator
from ..builders.pipeline_builder import VisionApplicationBuilder

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
    renderer: Optional[FrameRenderer] = None  # Inyectado opcionalmente; Fase 6 lo cablea.
    # Conservado para drenar TrafficData y publicar al broadcaster desde el
    # coroutine. None si persistence.enabled=False (modo dev/test sin agregador).
    aggregator: Optional[AsyncTrafficAggregator] = None
    # Detector YOLO de esta cámara. Conservado para liberarlo en la baja
    # dinámica (C1): tras `remove_camera` el modelo NO debe quedar en memoria.
    detector: Optional[Any] = None
    # Consumidores MJPEG activos del feed `/video/{id}` (C1). El SSE se cuenta
    # vía broadcaster.subscribed_cameras(); el MJPEG no tiene registro propio,
    # así que lo lleva el generador de video.py (inc al entrar, dec en finally).
    mjpeg_consumers: int = 0


class CameraInstance:
    """Encapsulates a camera with its pipeline, optional renderer and aggregator."""

    def __init__(
        self,
        camera_id: str,
        config: DictConfig,
        builder: VisionApplicationBuilder,
        renderer: Optional[FrameRenderer] = None,
    ) -> None:
        pipeline = builder.build_pipeline()
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
        task = asyncio.create_task(self._run_camera_pipeline(camera))
        self._tasks[camera_id] = task
        logger.info("Started camera: %s", camera_id)

    async def _run_camera_pipeline(self, camera: CameraInstance):
        """Main loop: drena TrafficData del aggregator y los publica al broadcaster.

        Cruce thread→async (6d): el worker thread del aggregator NUNCA llama
        publish; deposita TrafficData en su output_queue (thread-safe). Acá,
        ya dentro del event loop, hacemos `flush()` (sync no-blocking) y luego
        `await publish(td)` por item. Sin run_coroutine_threadsafe, sin event
        loop en el worker.

        El broadcast por-frame (shape viejo `FrameAnalysis` flat) quedó eliminado
        en 6d: el SSE emite el shape §6.2 que es por-ventana (TrafficData), no
        por-frame.
        """
        # Persistencia VISUAL de boxes (C1): con detect_every_n_frames>1, los skip
        # frames llegan con vehicles=[] (detection_ran=False). Para no parpadear,
        # dibujamos sobre ellos el último análisis INFERIDO. Es visual-only: el
        # aggregator ya consumió el análisis real aguas arriba (en la cadena), así
        # que las métricas NO se tocan. Se actualiza en cada detection frame —
        # incluso con 0 vehículos— para limpiar los boxes cuando la calle se vacía
        # (sin boxes fantasma).
        last_render_analysis = None
        try:
            for frame, analysis in camera.state.pipeline.run():
                if not camera.state.is_running:
                    break

                # MUESTREO (permanente, independiente del render): drenar el
                # aggregator y publicar al broadcaster. Es la ÚNICA fuente de
                # métricas/persistencia; corre siempre, haya o no render MJPEG.
                if camera.state.aggregator is not None:
                    for td in camera.state.aggregator.flush():
                        await self.broadcaster.publish(td)

                # RENDER MJPEG (on-demand, visual): solo si hay alguien mirando.
                # B1 Paso 0: gateado por `render_enabled` para no copiar/anotar
                # frames que nadie consume. NO afecta al muestreo de arriba.
                if camera.state.render_enabled and hasattr(frame, 'image') and frame.image is not None:
                    # Elegir qué análisis dibujar: el real si corrió detección este
                    # frame (y actualizar el cache), o el último inferido en skip frames.
                    render_analysis = analysis
                    if analysis is not None and analysis.detection_ran:
                        last_render_analysis = analysis
                    elif last_render_analysis is not None:
                        render_analysis = last_render_analysis

                    camera.state.latest_frame_raw = frame.image.copy()

                    processed_frame = frame.image.copy()
                    if render_analysis and camera.state.renderer is not None:
                        # Caso B roto (Fase 5e): el renderer es opcional vía
                        # Protocol FrameRenderer; no se importa de presentation/.
                        processed_frame = camera.state.renderer.render(frame, render_analysis)
                    camera.state.latest_frame_processed = processed_frame

                # Yield control to avoid blocking the event loop.
                await asyncio.sleep(0)

        except Exception:
            logger.exception("Camera %s failed", camera.state.camera_id)
            camera.state.is_running = False

    async def stop_camera(self, camera_id: str) -> None:
        """Stops a specific camera."""
        if camera_id not in self.cameras:
            return

        camera = self.cameras[camera_id]
        camera.state.is_running = False
        camera.state.pipeline.stop()

        if camera_id in self._tasks:
            self._tasks[camera_id].cancel()
            try:
                await self._tasks[camera_id]
            except asyncio.CancelledError:
                pass
            del self._tasks[camera_id]

        logger.info("Stopped camera: %s", camera_id)

    async def remove_camera(self, camera_id: str) -> None:
        """Baja dinámica (C1): para la cámara, libera su modelo YOLO y la saca
        del registro.

        A diferencia de `stop_camera` (que solo suelta el `cv2.VideoCapture`),
        acá liberamos también el detector YOLO (`detector.release()`) y borramos
        la `CameraInstance` de `self.cameras` para que el modelo no quede
        referenciado en memoria. Idempotente: si la cámara no existe, no-op.
        """
        if camera_id not in self.cameras:
            return

        await self.stop_camera(camera_id)

        camera = self.cameras.pop(camera_id, None)
        if camera is not None:
            detector = camera.state.detector
            if detector is not None and hasattr(detector, "release"):
                detector.release()
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


