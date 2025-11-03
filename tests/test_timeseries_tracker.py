import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.data_persistence import TimeSeriesTracker
from logic.vehicle_counter import VehicleCounter


class TestTimeSeriesTracker(unittest.TestCase):
    """Tests para TimeSeriesTracker"""
    
    def setUp(self):
        """Configuración antes de cada test"""
        self.fps = 30.0
        self.interval_sec = 10
        self.tracker = TimeSeriesTracker(
            interval_seconds=self.interval_sec,
            fps=self.fps
        )
        self.counter = VehicleCounter(line_y=100, direction='down')
    
    def test_initialization(self):
        """Test: Inicialización correcta"""
        self.assertEqual(self.tracker.interval_seconds, self.interval_sec)
        self.assertEqual(self.tracker.frames_per_interval, 300)  # 30fps * 10sec
        self.assertEqual(len(self.tracker.time_series), 0)
    
    def test_single_interval(self):
        """Test: Completar un intervalo"""
        # Simular 300 frames (10 segundos)
        for frame in range(1, 301):
            self.tracker.update(frame, self.counter)
        
        # Debería haber guardado 1 intervalo
        self.assertEqual(len(self.tracker.time_series), 1)
        self.assertEqual(self.tracker.time_series[0]['timestamp_sec'], 10)
    
    def test_multiple_intervals(self):
        """Test: Completar múltiples intervalos"""
        # Simular 3 intervalos (900 frames = 30 segundos)
        for frame in range(1, 901):
            self.tracker.update(frame, self.counter)
        
        # Deberían haberse guardado 3 intervalos
        self.assertEqual(len(self.tracker.time_series), 3)
        self.assertEqual(self.tracker.time_series[0]['timestamp_sec'], 10)
        self.assertEqual(self.tracker.time_series[1]['timestamp_sec'], 20)
        self.assertEqual(self.tracker.time_series[2]['timestamp_sec'], 30)
    
    def test_interval_with_counts(self):
        """Test: Intervalos con conteos de vehículos"""
        # Simular vehículos cruzando durante el primer intervalo
        for frame in range(1, 151):
            # Agregar un vehículo cada 50 frames
            if frame % 50 == 0:
                track_id = frame // 50
                self.counter.update(track_id, [50, 80, 60, 95], class_id=2)
                self.counter.update(track_id, [50, 100, 60, 115], class_id=2)
            
            self.tracker.update(frame, self.counter)
        
        # Completar el intervalo
        for frame in range(151, 301):
            self.tracker.update(frame, self.counter)
        
        # Verificar que se guardó el conteo
        self.assertEqual(len(self.tracker.time_series), 1)
        self.assertGreater(self.tracker.time_series[0]['count'], 0)
    
    def test_reset(self):
        """Test: Reset limpia la serie temporal"""
        # Generar datos
        for frame in range(1, 301):
            self.tracker.update(frame, self.counter)
        
        self.assertEqual(len(self.tracker.time_series), 1)
        
        # Reset
        self.tracker.reset()
        
        self.assertEqual(len(self.tracker.time_series), 0)
        self.assertEqual(self.tracker.current_frame, 0)
        self.assertEqual(self.tracker.current_interval, 1)


class TestTimeSeriesTrackerEdgeCases(unittest.TestCase):
    """Tests para casos extremos"""
    
    def test_zero_fps(self):
        """Test: FPS cero o negativo"""
        with self.assertRaises(ValueError):
            TimeSeriesTracker(interval_seconds=10, fps=0)
    
    def test_partial_interval(self):
        """Test: Intervalo incompleto no se guarda"""
        tracker = TimeSeriesTracker(interval_seconds=10, fps=30)
        counter = VehicleCounter(line_y=100, direction='down')
        
        # Solo 200 frames (menos de un intervalo completo)
        for frame in range(1, 201):
            tracker.update(frame, counter)
        
        # No debería haber guardado ningún intervalo
        self.assertEqual(len(tracker.time_series), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)