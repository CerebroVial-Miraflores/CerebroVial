"""
Manager for multiple independent camera pipelines.
"""
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from omegaconf import DictConfig
import asyncio
from ..pipelines.async_pipeline import AsyncVisionPipeline
from ..builders.pipeline_builder import VisionApplicationBuilder
from ...infrastructure.broadcast.realtime_broadcaster import RealtimeBroadcaster
from ...presentation.visualization.opencv_visualizer import OpenCVVisualizer


@dataclass
class CameraState:
    camera_id: str
    config: DictConfig
    pipeline: Any # AsyncVisionPipeline
    is_running: bool = False
    latest_frame_raw: Optional[Any] = None
    latest_frame_processed: Optional[Any] = None
    last_broadcast: float = 0.0
    visualizer: Optional[OpenCVVisualizer] = None # Visualizer is initialized later


class CameraInstance:
    """Encapsulates a camera with its pipeline and configuration."""
    
    def __init__(self, camera_id: str, config: DictConfig, builder: VisionApplicationBuilder):
        self.state = CameraState(
            camera_id=camera_id,
            config=config,
            pipeline=builder.build_pipeline()
        )
        
        # Initialize visualizer
        zones_config = {}
        if config.vision.zones:
            for k, v in config.vision.zones.items():
                # Filter zones for this camera
                if 'polygon' in v and v.get('camera_id') == camera_id:
                    zones_config[k] = list(v['polygon'])
        
        self.state.visualizer = OpenCVVisualizer(zones_config=zones_config)



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
        if camera.state.is_running:
            print(f"[MultiCamera] Camera {camera_id} already running")
            return
        
        camera.state.is_running = True
        
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
            for frame, analysis in camera.state.pipeline.run():
                if not camera.state.is_running:
                    break
                
                # Serialize and broadcast
                if analysis:
                    current_time = time.time()
                    if current_time - camera.state.last_broadcast >= 2:
                        # print(f"[MultiCamera] Broadcasting analysis for {camera.state.camera_id}")
                        data = self.broadcaster.serialize_analysis(analysis, camera.state.camera_id)
                        await self.broadcaster.broadcast(camera.state.camera_id, data)
                        camera.state.last_broadcast = current_time
                    
                # Store frames for video streaming (ALWAYS update)
                if hasattr(frame, 'image'):
                    # Raw frame
                    camera.state.latest_frame_raw = frame.image.copy()
                    
                    # Processed frame - Use current analysis or last known good analysis if needed
                    processed_frame = frame.image.copy()
                    if analysis:
                        processed_frame = camera.state.visualizer.draw(processed_frame, analysis)
                    camera.state.latest_frame_processed = processed_frame

                
                # Yield control to avoid blocking event loop
                await asyncio.sleep(0)
                
        except Exception as e:
            print(f"[ERROR] Camera {camera.state.camera_id} failed: {e}")
            camera.state.is_running = False

    async def stop_camera(self, camera_id: str):
        """Stops a specific camera."""
        if camera_id not in self.cameras:
            return
        
        camera = self.cameras[camera_id]
        camera.state.is_running = False
        camera.state.pipeline.stop()
        
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


