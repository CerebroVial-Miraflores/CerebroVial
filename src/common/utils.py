"""
Common utilities shared across all modules.
"""

import logging

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger with standard configuration.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
