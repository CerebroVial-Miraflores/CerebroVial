class VisionError(Exception):
    """Base exception for all vision module errors."""
    pass

class DetectionError(VisionError):
    """Raised when vehicle detection fails."""
    pass

class SourceError(VisionError):
    """Raised when video source operations fail."""
    pass

class ConfigurationError(VisionError):
    """Raised when configuration is invalid."""
    pass
