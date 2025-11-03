import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class DataPersistence:
    """
    Maneja el guardado de resultados de conteo en JSON y CSV.
    """
    
    def __init__(self, output_dir: str = "data/results"):
        """
        Inicializa el sistema de persistencia.
        
        :param output_dir: Directorio donde guardar los resultados
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
    
    def save_results(self, 
                    counter,
                    video_info: Dict,
                    processing_info: Dict,
                    time_series: Optional[List[Dict]] = None) -> Dict[str, str]:
        """
        Guarda los resultados del procesamiento en JSON y CSV.
        
        :param counter: Instancia de VehicleCounter con los resultados
        :param video_info: Información del video (path, resolución, fps, etc.)
        :param processing_info: Info del procesamiento (frames procesados, duración, etc.)
        :param time_series: Lista opcional de conteos por intervalo temporal
        :return: Dict con las rutas de los archivos guardados
        """
        session_end = datetime.now()
        duration = (session_end - self.session_start).total_seconds()
        
        # Preparar datos
        results = {
            "session_id": self.session_id,
            "timestamp_start": self.session_start.isoformat(),
            "timestamp_end": session_end.isoformat(),
            "processing_duration_sec": round(duration, 2),
            
            "video_info": video_info,
            "processing_info": processing_info,
            
            "counts": {
                "total": counter.get_count(),
                "by_type": counter.get_counts_summary(),
                "by_class_id": counter.get_counts_by_class()
            }
        }
        
        # Agregar serie temporal si existe
        if time_series:
            results["time_series"] = time_series
        
        # Guardar JSON
        json_path = self.output_dir / f"results_{self.session_id}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Guardar CSV con resumen
        csv_summary_path = self.output_dir / f"summary_{self.session_id}.csv"
        self._save_summary_csv(csv_summary_path, results)
        
        # Guardar CSV de serie temporal si existe
        csv_timeseries_path = None
        if time_series:
            csv_timeseries_path = self.output_dir / f"timeseries_{self.session_id}.csv"
            self._save_timeseries_csv(csv_timeseries_path, time_series)
        
        return {
            "json": str(json_path),
            "csv_summary": str(csv_summary_path),
            "csv_timeseries": str(csv_timeseries_path) if csv_timeseries_path else None
        }
    
    def _save_summary_csv(self, path: Path, results: Dict):
        """Guarda un CSV con el resumen de conteo."""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Encabezado
            writer.writerow(['Métrica', 'Valor'])
            
            # Información general
            writer.writerow(['ID de Sesión', results['session_id']])
            writer.writerow(['Fecha/Hora Inicio', results['timestamp_start']])
            writer.writerow(['Duración (seg)', results['processing_duration_sec']])
            writer.writerow([])
            
            # Conteo total
            writer.writerow(['Total de Vehículos', results['counts']['total']])
            writer.writerow([])
            
            # Desglose por tipo
            writer.writerow(['Tipo de Vehículo', 'Cantidad'])
            for vehicle_type, count in results['counts']['by_type'].items():
                writer.writerow([vehicle_type, count])
    
    def _save_timeseries_csv(self, path: Path, time_series: List[Dict]):
        """Guarda un CSV con la serie temporal de conteos."""
        if not time_series:
            return
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            # Obtener todas las claves posibles (timestamps + tipos de vehículos)
            fieldnames = ['timestamp_sec', 'interval_total']
            
            # Agregar columnas para cada tipo de vehículo si existen
            if time_series and 'by_type' in time_series[0]:
                vehicle_types = time_series[0]['by_type'].keys()
                fieldnames.extend(vehicle_types)
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in time_series:
                row = {
                    'timestamp_sec': entry['timestamp_sec'],
                    'interval_total': entry['count']
                }
                
                # Agregar conteos por tipo si existen
                if 'by_type' in entry:
                    row.update(entry['by_type'])
                
                writer.writerow(row)


class TimeSeriesTracker:
    """
    Rastrea conteos de vehículos en intervalos de tiempo.
    Útil para análisis temporal del tráfico.
    """
    
    def __init__(self, interval_seconds: int = 60, fps: float = 30.0):
        """
        Inicializa el rastreador de series temporales.
        
        :param interval_seconds: Duración de cada intervalo en segundos
        :param fps: FPS del video para calcular frames por intervalo
        """
        self.interval_seconds = interval_seconds
        self.frames_per_interval = int(fps * interval_seconds)
        
        self.current_frame = 0
        self.current_interval = 1
        
        self.time_series = []
        self.interval_counts = {}  # Conteos del intervalo actual
    
    def update(self, frame_number: int, counter):
        """
        Actualiza el rastreador y guarda datos si completó un intervalo.
        
        :param frame_number: Número del frame actual
        :param counter: Instancia de VehicleCounter con conteos actuales
        """
        self.current_frame = frame_number
        
        # Verificar si completamos un intervalo
        if self.current_frame >= self.frames_per_interval * self.current_interval:
            timestamp_sec = self.current_interval * self.interval_seconds
            
            # Obtener conteos del intervalo
            current_counts = counter.get_counts_by_class()
            
            # Calcular conteos del intervalo (diferencia con intervalo anterior)
            interval_data = {
                'timestamp_sec': timestamp_sec,
                'count': sum(current_counts.values()),
                'by_type': counter.get_counts_summary()
            }
            
            self.time_series.append(interval_data)
            self.current_interval += 1
    
    def get_time_series(self) -> List[Dict]:
        """Retorna la serie temporal completa."""
        return self.time_series
    
    def reset(self):
        """Reinicia el rastreador."""
        self.current_frame = 0
        self.current_interval = 1
        self.time_series = []
        self.interval_counts = {}