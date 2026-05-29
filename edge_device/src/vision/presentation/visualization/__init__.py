"""Visualization adapters (FrameRenderer concretos)."""
from .opencv_visualizer import OpenCVVisualizer
from .factory import build_visualizer_from_vision_cfg

__all__ = ["OpenCVVisualizer", "build_visualizer_from_vision_cfg"]
