"""
Domain repositories for the Computer Vision module.
"""
from typing import Protocol

from .entities import TrafficData


class TrafficRepository(Protocol):
    """Protocol for persisting traffic aggregates. Write-only in MVP1."""
    def save(self, data: TrafficData) -> None: ...
