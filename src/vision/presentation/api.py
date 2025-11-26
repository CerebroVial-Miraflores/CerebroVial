import cv2
import threading
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ..application.pipeline import VisionPipeline
from ..infrastructure.visualization import OpenCVVisualizer

app = FastAPI(title="CerebroVial Vision API")

# Global state
pipeline: VisionPipeline = None
visualizer: OpenCVVisualizer = None
output_frame = None
lock = threading.Lock()

def set_pipeline(p: VisionPipeline, v: OpenCVVisualizer):
    global pipeline, visualizer
    pipeline = p
    visualizer = v
    # Start processing in a background thread
    t = threading.Thread(target=process_frames)
    t.daemon = True
    t.start()

def process_frames():
    global output_frame, lock
    if not pipeline:
        return

    for frame, analysis in pipeline.run():
        if analysis:
            # Draw visualizations
            frame.image = visualizer.draw(frame.image, analysis)
        
        with lock:
            output_frame = frame.image.copy()

def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                continue
            
            # Encode frame as JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
                
        # Yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')
        
        # Limit frame rate for streaming to save bandwidth
        time.sleep(0.03)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
def status():
    return {"status": "running", "pipeline_active": pipeline is not None}
