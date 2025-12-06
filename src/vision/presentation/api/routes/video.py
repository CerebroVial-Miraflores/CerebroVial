"""
API for video streaming.
"""
import cv2
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from .cameras import get_manager

app = FastAPI()

@app.get("/video/{camera_id}")
async def video_feed(camera_id: str, type: str = "raw"):
    """
    MJPEG video stream for a specific camera.
    type: 'raw' or 'processed'
    """
    manager = get_manager()
    
    async def frame_generator():
        try:
            while True:
                # Check if camera exists and is running
                if camera_id not in manager.cameras:
                    break
                
                camera = manager.cameras[camera_id]
                if not camera.state.is_running:
                    break
                    
                # Get latest frame
                processed = (type == "processed")
                frame = manager.get_latest_frame(camera_id, processed=processed)
                
                if frame is not None:
                    # Encode to JPEG
                    try:
                        (flag, encodedImage) = cv2.imencode(".jpg", frame)
                        if flag:
                            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                                   bytearray(encodedImage) + b'\r\n')
                    except Exception as e:
                        print(f"[ERROR] Encoding failed for {camera_id}: {e}")
                        continue
                
                # Control framerate (approx 24 fps)
                await asyncio.sleep(0.04)
        except Exception as e:
            print(f"[ERROR] Video stream failed for {camera_id}: {e}")
            
    return StreamingResponse(
        frame_generator(), 
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )
