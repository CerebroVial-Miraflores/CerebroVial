import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "CerebroVial", log_level: str = "INFO", log_to_file: bool = True):
    """
    Configura un logger con formato consistente y opción de guardado en archivo.
    
    :param name: Nombre del logger
    :param log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param log_to_file: Si True, guarda logs en archivo además de consola
    :return: Logger configurado
    """
    # Crear logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Evitar duplicar handlers si ya existe
    if logger.handlers:
        return logger
    
    # Formato detallado para logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para consola (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (opcional)
    if log_to_file:
        # Crear directorio de logs si no existe
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Nombre de archivo con timestamp
        log_filename = f"cerebrovial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = log_dir / log_filename
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Archivo guarda más detalle
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logs guardándose en: {log_path}")
    
    return logger


# Logger global para uso rápido
logger = setup_logger()