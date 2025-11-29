"""
Domain repositories for the Computer Vision module.
"""
from typing import Protocol
from .entities import TrafficData

class TrafficRepository(Protocol):
    """
    Abstract base class for saving traffic data.
    """
    def save(self, data: TrafficData):
        ...
