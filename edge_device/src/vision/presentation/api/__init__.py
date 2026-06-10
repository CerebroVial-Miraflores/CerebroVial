"""
API package.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import cameras, health, state, streaming, video
from ...infrastructure.broadcast.realtime_broadcaster import RealtimeBroadcaster

# Initialize main app
app = FastAPI(title="CerebroVial Vision API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(cameras.app.router, tags=["cameras"])
app.include_router(streaming.app.router, tags=["streaming"])
app.include_router(state.app.router, tags=["state"])
app.include_router(health.app.router, tags=["health"])
app.include_router(video.app.router, tags=["video"])

# Initialize shared components
broadcaster = RealtimeBroadcaster()
cameras.init_manager(broadcaster)
streaming.init_broadcaster(broadcaster)


@app.on_event("shutdown")
async def _shutdown_vision_manager() -> None:
    """B1 Paso 2: teardown del manager al bajar la app — libera el detector
    compartido y apaga el executor de inferencia una vez (contrapeso del release
    por-cámara que se sacó en `remove_camera`). En contenedor el stop suele ser
    SIGKILL (no corre); queda para shutdown graceful y deploys no-contenedor."""
    await cameras.get_manager().shutdown()

# Compatibility wrapper for run_server.py
def set_pipeline(pipeline, visualizer):
    """
    Legacy compatibility. 
    In the new architecture, we add a default camera with this pipeline.
    """
    cameras.get_manager()
    # We can't easily inject an existing pipeline into the manager because the manager builds them.
    # But run_server.py builds a pipeline first. 
    # For now, we might need to refactor run_server.py to use the manager directly 
    # or just ignore this if we want to start fresh.
    pass
