import logging
import time
from functools import wraps
from typing import Callable

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with a standard format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

def log_execution_time(logger: logging.Logger):
    """
    Decorator to measure and log execution time of a function.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                # Log only if it takes significant time (e.g. > 10ms) or debug is on
                if logger.isEnabledFor(logging.DEBUG) or elapsed > 0.01:
                    logger.debug(f"{func.__name__} executed in {elapsed:.3f}s")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {e}", exc_info=True)
                raise
        return wrapper
    return decorator
