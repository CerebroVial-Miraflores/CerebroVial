import cv2
import threading
import time
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from ..application.pipeline import VisionPipeline
from ..infrastructure.visualization import OpenCVVisualizer

app = FastAPI(title="CerebroVial Vision API")

class VisionService:
    """
    Service layer for Vision API.
    Encapsulates pipeline execution and frame streaming.
    """
    def __init__(self, pipeline: VisionPipeline, visualizer: OpenCVVisualizer):
        self.pipeline = pipeline
        self.visualizer = visualizer
        self.output_frame = None
        self.lock = threading.Lock()
        self._start_processing()

    def _start_processing(self):
        t = threading.Thread(target=self._process_frames)
        t.daemon = True
        t.start()

    def _process_frames(self):
        for frame, analysis in self.pipeline.run():
            if analysis:
                frame.image = self.visualizer.draw(frame.image, analysis)
            
            with self.lock:
                self.output_frame = frame.image.copy()

    def generate_stream(self):
        while True:
            with self.lock:
                if self.output_frame is None:
                    time.sleep(0.01)
                    continue
                
                (flag, encodedImage) = cv2.imencode(".jpg", self.output_frame)
                if not flag:
                    continue
                    
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                   bytearray(encodedImage) + b'\r\n')
            
            time.sleep(0.03)

    def get_metrics(self):
        if self.pipeline.metrics_collector:
            return self.pipeline.metrics_collector.get_metrics().to_dict()
        return {"error": "Metrics not available"}

# Singleton instance
_service: Optional[VisionService] = None

def get_vision_service() -> VisionService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Vision service not initialized")
    return _service

def set_pipeline(p: VisionPipeline, v: OpenCVVisualizer):
    global _service
    _service = VisionService(p, v)

@app.get("/video_feed")
def video_feed(service: VisionService = Depends(get_vision_service)):
    return StreamingResponse(service.generate_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
def status():
    return {"status": "running", "pipeline_active": _service is not None}

@app.get("/metrics")
async def get_metrics(service: VisionService = Depends(get_vision_service)):
    return service.get_metrics()
