import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os

class Config:
    """
    Clase para cargar y acceder a la configuración desde archivo YAML.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa la configuración.
        
        :param config_path: Ruta al archivo YAML. Si es None, usa el default.
        """
        if config_path is None:
            # Buscar config por defecto
            script_dir = Path(__file__).parent.parent.parent
            config_path = script_dir / 'configs' / 'default_config.yaml'
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga el archivo YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Expandir rutas relativas
        config = self._expand_paths(config)
        
        return config
    
    def _expand_paths(self, config: Dict) -> Dict:
        """
        Expande rutas relativas a absolutas basadas en el directorio del script.
        """
        base_dir = Path(__file__).parent.parent
        
        if 'paths' in config:
            for key, value in config['paths'].items():
                if isinstance(value, str) and not os.path.isabs(value):
                    config['paths'][key] = str(base_dir / value)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración usando notación de punto.
        
        Ejemplo: config.get('model.path') -> 'yolov8n.pt'
        
        :param key: Clave en notación de punto (ej: 'model.path')
        :param default: Valor por defecto si no existe
        :return: Valor de configuración
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_vehicle_classes(self) -> list:
        """
        Retorna la lista de IDs de clases de vehículos a detectar.
        
        :return: Lista de IDs [2, 3, 5, 7]
        """
        classes = self.get('vehicle_classes', {})
        return list(classes.values())
    
    def get_line_y(self, frame_height: int) -> int:
        """
        Calcula la posición Y de la línea de conteo.
        
        :param frame_height: Altura del frame en píxeles
        :return: Posición Y en píxeles
        """
        ratio = self.get('counter.line_y_ratio', 0.5)
        return int(frame_height * ratio)
    
    def get_visualization_colors(self) -> Dict[str, tuple]:
        """
        Retorna los colores configurados para visualización.
        
        :return: Dict con colores en formato RGB tuple
        """
        viz_config = self.get('visualization', {})
        return {
            'line': tuple(viz_config.get('line_color', [0, 255, 0])),
            'text': tuple(viz_config.get('text_color', [255, 255, 255])),
            'crossing': tuple(viz_config.get('crossing_indicator_color', [0, 0, 255]))
        }
    
    def __repr__(self):
        return f"Config(path={self.config_path})"


# Instancia global para fácil acceso
_global_config = None

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Carga la configuración global.
    
    :param config_path: Ruta opcional al archivo de configuración
    :return: Instancia de Config
    """
    global _global_config
    _global_config = Config(config_path)
    return _global_config

def get_config() -> Config:
    """
    Obtiene la configuración global. Si no existe, carga la default.
    
    :return: Instancia de Config
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config