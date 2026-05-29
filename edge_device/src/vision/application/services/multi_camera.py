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
    latest_frame_raw: Optional[Any] = None
    latest_frame_processed: Optional[Any] = None
    renderer: Optional[FrameRenderer] = None  # Inyectado opcionalmente; Fase 6 lo cablea.
    # Conservado para drenar TrafficData y publicar al broadcaster desde el
    # coroutine. None si persistence.enabled=False (modo dev/test sin agregador).
    aggregator: Optional[AsyncTrafficAggregator] = None


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
        aggregator = builder.get_components().get("aggregator")
        self.state = CameraState(
            camera_id=camera_id,
            config=config,
            pipeline=pipeline,
            renderer=renderer,
            aggregator=aggregator,
        )



class MultiCameraManager:
    """
    Manages multiple camera pipelines simultaneously.
    Each camera runs in its own set of threads.
    """
    
    def __init__(self, broadcaster: RealtimeBroadcaster) -> None:
        self.cameras: dict[str, CameraInstance] = {}
        self.broadcaster = broadcaster
        self._tasks: dict[str, asyncio.Task] = {}

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
        try:
            for frame, analysis in camera.state.pipeline.run():
                if not camera.state.is_running:
                    break

                if camera.state.aggregator is not None:
                    for td in camera.state.aggregator.flush():
                        await self.broadcaster.publish(td)

                # Store frames for video streaming (ALWAYS update).
                if hasattr(frame, 'image') and frame.image is not None:
                    camera.state.latest_frame_raw = frame.image.copy()

                    processed_frame = frame.image.copy()
                    if analysis and camera.state.renderer is not None:
                        # Caso B roto (Fase 5e): el renderer es opcional vía
                        # Protocol FrameRenderer; no se importa de presentation/.
                        processed_frame = camera.state.renderer.render(frame, analysis)
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


