import unittest
import sys
from pathlib import Path

# Agregar src al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from logic.vehicle_counter import VehicleCounter


class TestVehicleCounter(unittest.TestCase):
    """Tests para la clase VehicleCounter"""
    
    def setUp(self):
        """Configuración antes de cada test"""
        self.line_y = 100
    
    def test_initialization(self):
        """Test: Inicialización correcta del contador"""
        counter = VehicleCounter(line_y=self.line_y, direction='down')
        
        self.assertEqual(counter.line_y, self.line_y)
        self.assertEqual(counter.direction, 'down')
        self.assertEqual(counter.get_count(), 0)
        self.assertEqual(len(counter.counted_ids), 0)
    
    def test_invalid_direction(self):
        """Test: Dirección inválida lanza error"""
        with self.assertRaises(ValueError):
            VehicleCounter(line_y=100, direction='left')
    
    def test_vehicle_crosses_line_down(self):
        """Test: Vehículo cruza línea de arriba hacia abajo"""
        counter = VehicleCounter(line_y=100, direction='down')
        
        # Vehículo empieza arriba de la línea
        box1 = [50, 50, 60, 80]  # y2=80 (arriba de línea)
        crossed = counter.update(track_id=1, box=box1, class_id=2)
        self.assertFalse(crossed)
        self.assertEqual(counter.get_count(), 0)
        
        # Vehículo cruza la línea
        box2 = [50, 90, 60, 110]  # y2=110 (abajo de línea)
        crossed = counter.update(track_id=1, box=box2, class_id=2)
        self.assertTrue(crossed)
        self.assertEqual(counter.get_count(), 1)
        
        # Vehículo ya fue contado, no debe contarse de nuevo
        box3 = [50, 110, 60, 130]
        crossed = counter.update(track_id=1, box=box3, class_id=2)
        self.assertFalse(crossed)
        self.assertEqual(counter.get_count(), 1)
    
    def test_vehicle_crosses_line_up(self):
        """Test: Vehículo cruza línea de abajo hacia arriba"""
        counter = VehicleCounter(line_y=100, direction='up')
        
        # Vehículo empieza abajo de la línea
        box1 = [50, 110, 60, 130]  # y2=130 (abajo de línea)
        crossed = counter.update(track_id=1, box=box1, class_id=2)
        self.assertFalse(crossed)
        self.assertEqual(counter.get_count(), 0)
        
        # Vehículo cruza la línea
        box2 = [50, 70, 60, 90]  # y2=90 (arriba de línea)
        crossed = counter.update(track_id=1, box=box2, class_id=2)
        self.assertTrue(crossed)
        self.assertEqual(counter.get_count(), 1)
    
    def test_multiple_vehicles(self):
        """Test: Múltiples vehículos cruzando"""
        counter = VehicleCounter(line_y=100, direction='down')
        
        # Vehículo 1 cruza
        counter.update(track_id=1, box=[50, 50, 60, 80], class_id=2)
        counter.update(track_id=1, box=[50, 90, 60, 110], class_id=2)
        
        # Vehículo 2 cruza
        counter.update(track_id=2, box=[100, 60, 110, 85], class_id=3)
        counter.update(track_id=2, box=[100, 95, 110, 115], class_id=3)
        
        # Vehículo 3 cruza
        counter.update(track_id=3, box=[150, 70, 160, 90], class_id=5)
        counter.update(track_id=3, box=[150, 100, 160, 120], class_id=5)
        
        self.assertEqual(counter.get_count(), 3)
    
    def test_count_by_class(self):
        """Test: Conteo separado por clase de vehículo"""
        counter = VehicleCounter(line_y=100, direction='down')
        
        # 2 autos (class_id=2)
        counter.update(track_id=1, box=[50, 80, 60, 95], class_id=2)
        counter.update(track_id=1, box=[50, 100, 60, 115], class_id=2)
        
        counter.update(track_id=2, box=[70, 80, 80, 95], class_id=2)
        counter.update(track_id=2, box=[70, 100, 80, 115], class_id=2)
        
        # 1 moto (class_id=3)
        counter.update(track_id=3, box=[90, 80, 95, 95], class_id=3)
        counter.update(track_id=3, box=[90, 100, 95, 115], class_id=3)
        
        # 1 bus (class_id=5)
        counter.update(track_id=4, box=[110, 80, 120, 95], class_id=5)
        counter.update(track_id=4, box=[110, 100, 120, 115], class_id=5)
        
        self.assertEqual(counter.get_count(), 4)
        
        counts = counter.get_counts_by_class()
        self.assertEqual(counts[2], 2)  # 2 autos
        self.assertEqual(counts[3], 1)  # 1 moto
        self.assertEqual(counts[5], 1)  # 1 bus
        self.assertEqual(counts[7], 0)  # 0 camiones
    
    def test_vehicle_not_crossing(self):
        """Test: Vehículo que no cruza la línea no se cuenta"""
        counter = VehicleCounter(line_y=100, direction='down')
        
        # Vehículo se mueve pero siempre arriba de la línea
        counter.update(track_id=1, box=[50, 50, 60, 70], class_id=2)
        counter.update(track_id=1, box=[55, 55, 65, 75], class_id=2)
        counter.update(track_id=1, box=[60, 60, 70, 80], class_id=2)
        
        self.assertEqual(counter.get_count(), 0)
    
    def test_reset(self):
        """Test: Reset limpia todos los contadores"""
        counter = VehicleCounter(line_y=100, direction='down')
        
        # Agregar algunos conteos
        counter.update(track_id=1, box=[50, 80, 60, 95], class_id=2)
        counter.update(track_id=1, box=[50, 100, 60, 115], class_id=2)
        
        self.assertEqual(counter.get_count(), 1)
        
        # Reset
        counter.reset()
        
        self.assertEqual(counter.get_count(), 0)
        self.assertEqual(len(counter.counted_ids), 0)
        self.assertEqual(len(counter.track_history), 0)
    
    def test_get_counts_summary(self):
        """Test: Resumen legible de conteos"""
        counter = VehicleCounter(line_y=100, direction='down')
        
        counter.update(track_id=1, box=[50, 80, 60, 95], class_id=2)
        counter.update(track_id=1, box=[50, 100, 60, 115], class_id=2)
        
        summary = counter.get_counts_summary()
        
        self.assertIn('Autos', summary)
        self.assertEqual(summary['Autos'], 1)


class TestVehicleCounterEdgeCases(unittest.TestCase):
    """Tests para casos extremos"""
    
    def test_vehicle_exactly_on_line(self):
        """Test: Vehículo exactamente en la línea"""
        counter = VehicleCounter(line_y=100, direction='down')
        
        # Vehículo empieza justo en la línea
        counter.update(track_id=1, box=[50, 80, 60, 100], class_id=2)
        self.assertEqual(counter.get_count(), 0)
        
        # Se mueve hacia abajo
        counter.update(track_id=1, box=[50, 90, 60, 110], class_id=2)
        self.assertEqual(counter.get_count(), 0)  # No debería contar
    
    def test_vehicle_jumps_over_line(self):
        """Test: Vehículo salta la línea (frames perdidos)"""
        counter = VehicleCounter(line_y=100, direction='down')
        
        # Vehículo arriba
        counter.update(track_id=1, box=[50, 50, 60, 70], class_id=2)
        
        # Vehículo abajo (saltó la línea entre frames)
        counter.update(track_id=1, box=[50, 120, 60, 140], class_id=2)
        
        # Debería contarse igual
        self.assertEqual(counter.get_count(), 1)


if __name__ == '__main__':
    # Ejecutar tests con verbosidad
    unittest.main(verbosity=2)