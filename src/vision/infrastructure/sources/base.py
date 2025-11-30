"""
Base classes and configuration for video sources.
"""
from typing import Optional, Iterator
from abc import ABC, abstractmethod
from pydantic import BaseModel, field_validator, Field
from ...domain.protocols import FrameProducer
from ...domain.entities import Frame
from ....common.exceptions import SourceError

class SourceConfig(BaseModel):
    """Validated configuration for video sources"""
    buffer_size: int = Field(3, ge=1, le=120, description="OpenCV buffer size")
    target_width: Optional[int] = Field(None, gt=0, description="Target width in pixels")
    target_height: Optional[int] = Field(None, gt=0, description="Target height in pixels")
    format: str = Field("best", description="YouTube format")

    @field_validator('target_width', 'target_height')
    @classmethod
    def validate_resolution(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v % 2 != 0:
            raise ValueError('Resolution must be even number for video encoding')
        return v

class SourceFactory(ABC):
    """
    Abstract factory for creating video sources.
    """
    
    @abstractmethod
    def create(self, config: str, **kwargs) -> FrameProducer:
        pass
    
    @abstractmethod
    def can_handle(self, config: str, source_type: str) -> bool:
        pass

    def _create_config(self, **kwargs) -> SourceConfig:
        return SourceConfig(**kwargs)
