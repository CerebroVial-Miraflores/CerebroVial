"""
Manager for multiple independent camera pipelines.
"""
import asyncio
from typing import Dict, List
from omegaconf import DictConfig
from ..pipelines.async_pipeline import AsyncVisionPipeline
from ..builders.pipeline_builder import VisionApplicationBuilder
from ...infrastructure.broadcast.realtime_broadcaster import RealtimeBroadcaster

class CameraInstance:
    """Encapsulates a camera with its pipeline and configuration."""
    
    def __init__(self, camera_id: str, config: DictConfig, builder: VisionApplicationBuilder):
        self.camera_id = camera_id
        self.config = config
        self.pipeline = builder.build_pipeline()
        self.is_running = False

class MultiCameraManager:
    """
    Manages multiple camera pipelines simultaneously.
    Each camera runs in its own set of threads.
    """
    
    def __init__(self, broadcaster: RealtimeBroadcaster):
        self.cameras: Dict[str, CameraInstance] = {}
        self.broadcaster = broadcaster
        self._tasks: Dict[str, asyncio.Task] = {}

    def add_camera(self, camera_id: str, config: DictConfig) -> CameraInstance:
        """
        Adds a new camera to the system.
        
        Args:
            camera_id: Unique identifier (e.g., "CAM_001")
            config: Hydra configuration for this camera
        """
        if camera_id in self.cameras:
            raise ValueError(f"Camera {camera_id} already exists")
        
        # Inject camera_id into zones config
        if 'zones' in config.vision:
            for zone_id, zone_cfg in config.vision.zones.items():
                if isinstance(zone_cfg, dict) and 'camera_id' not in zone_cfg:
                    zone_cfg['camera_id'] = camera_id
        
        builder = VisionApplicationBuilder(config)
        camera = CameraInstance(camera_id, config, builder)
        
        self.cameras[camera_id] = camera
        print(f"[MultiCamera] Added camera: {camera_id}")
        
        return camera

    async def start_camera(self, camera_id: str):
        """Starts processing for a camera."""
        if camera_id not in self.cameras:
            raise ValueError(f"Camera {camera_id} not found")
        
        camera = self.cameras[camera_id]
        if camera.is_running:
            print(f"[MultiCamera] Camera {camera_id} already running")
            return
        
        camera.is_running = True
        
        # Create async task for this pipeline
        task = asyncio.create_task(
            self._run_camera_pipeline(camera)
        )
        self._tasks[camera_id] = task
        print(f"[MultiCamera] Started camera: {camera_id}")

    async def _run_camera_pipeline(self, camera: CameraInstance):
        """
        Main loop for a camera.
        Processes frames and broadcasts to broadcaster.
        """
        try:
            for frame, analysis in camera.pipeline.run():
                if not camera.is_running:
                    break
                
                # Serialize and broadcast
                if analysis:
                    data = self.broadcaster.serialize_analysis(analysis, camera.camera_id)
                    await self.broadcaster.broadcast(camera.camera_id, data)
                
                # Yield control to avoid blocking event loop
                await asyncio.sleep(0)
                
        except Exception as e:
            print(f"[ERROR] Camera {camera.camera_id} failed: {e}")
            camera.is_running = False

    async def stop_camera(self, camera_id: str):
        """Stops a specific camera."""
        if camera_id not in self.cameras:
            return
        
        camera = self.cameras[camera_id]
        camera.is_running = False
        camera.pipeline.stop()
        
        # Cancel task
        if camera_id in self._tasks:
            self._tasks[camera_id].cancel()
            try:
                await self._tasks[camera_id]
            except asyncio.CancelledError:
                pass
            del self._tasks[camera_id]
        
        print(f"[MultiCamera] Stopped camera: {camera_id}")

    async def start_all(self):
        """Starts all registered cameras."""
        tasks = [self.start_camera(cam_id) for cam_id in self.cameras.keys()]
        await asyncio.gather(*tasks)

    async def stop_all(self):
        """Stops all cameras."""
        tasks = [self.stop_camera(cam_id) for cam_id in list(self.cameras.keys())]
        await asyncio.gather(*tasks)

    def get_status(self) -> Dict:
        """Returns status of all cameras."""
        return {
            cam_id: {
                "running": cam.is_running,
                "source": cam.config.vision.source,
                "zones": list(cam.config.vision.zones.keys()) if cam.config.vision.zones else []
            }
            for cam_id, cam in self.cameras.items()
        }
