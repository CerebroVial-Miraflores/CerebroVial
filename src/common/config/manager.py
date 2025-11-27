from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional

class ConfigManager:
    """Centraliza la carga y validación de configuración"""
    
    def __init__(self, config_dir: Path = Path("conf")):
        self.config_dir = config_dir
    
    def load_vision_config(self, profile: str = "default") -> DictConfig:
        """Carga configuración de visión con validación"""
        config_path = self.config_dir / "vision" / f"{profile}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        cfg = OmegaConf.load(config_path)
        # Validación básica
        required_keys = ['source', 'model', 'performance']
        for key in required_keys:
            if key not in cfg:
                raise ValueError(f"Missing required config key: {key}")
        
        return cfg
