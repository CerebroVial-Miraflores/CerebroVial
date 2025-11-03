#!/usr/bin/env python3
"""
Script para ejecutar todos los tests de CerebroVial.

Uso:
    python run_tests.py              # Ejecutar todos los tests
    python run_tests.py -v           # Modo verbose
    python run_tests.py TestClass    # Ejecutar solo una clase de tests
"""

import unittest
import sys
from pathlib import Path

# Agregar directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def run_all_tests(verbosity=2):
    """
    Descubre y ejecuta todos los tests en el directorio tests/.
    
    :param verbosity: Nivel de detalle (1=básico, 2=detallado)
    """
    # Descubrir todos los tests
    loader = unittest.TestLoader()
    tests_dir = Path(__file__).parent
    suite = loader.discover(str(tests_dir), pattern='test_*.py')
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Retornar código de salida
    return 0 if result.wasSuccessful() else 1


def print_test_summary():
    """Imprime un resumen de los tests disponibles."""
    print("=" * 70)
    print("CerebroVial - Suite de Tests")
    print("=" * 70)
    print("\nTests disponibles:")
    print("  - test_vehicle_counter.py: Tests del contador de vehículos")
    print("  - test_timeseries_tracker.py: Tests del rastreador temporal")
    print("\nUso:")
    print("  python run_tests.py              # Ejecutar todos")
    print("  python run_tests.py -v           # Modo verbose")
    print("  python run_tests.py --help       # Ver esta ayuda")
    print("=" * 70)


if __name__ == '__main__':
    # Verificar argumentos
    if '--help' in sys.argv or '-h' in sys.argv:
        print_test_summary()
        sys.exit(0)
    
    # Determinar verbosidad
    verbosity = 2 if '-v' in sys.argv or '--verbose' in sys.argv else 1
    
    print("\n🧪 Ejecutando tests de CerebroVial...\n")
    
    # Ejecutar tests
    exit_code = run_all_tests(verbosity=verbosity)
    
    # Mensaje final
    if exit_code == 0:
        print("\n✅ Todos los tests pasaron exitosamente!")
    else:
        print("\n❌ Algunos tests fallaron. Revisa los errores arriba.")
    
    sys.exit(exit_code)