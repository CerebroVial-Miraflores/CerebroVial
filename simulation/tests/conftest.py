"""Pytest config: setea SUMO_HOME + PATH antes de cargar tests."""
from __future__ import annotations

import os
from pathlib import Path


_SIM_ROOT = Path(__file__).resolve().parent.parent
_SUMO_PKG = _SIM_ROOT / ".venv" / "lib" / "python3.11" / "site-packages" / "sumo"

# Escenario A (F0): wheel autocontenido. SUMO_HOME apunta al wheel del venv.
os.environ.setdefault("SUMO_HOME", str(_SUMO_PKG))
# Asegurar que sumo/sumo-gui/netgenerate/duarouter resuelvan al wheel.
_BIN = str(_SUMO_PKG / "bin")
if _BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _BIN + os.pathsep + os.environ.get("PATH", "")


def pytest_configure(config):
    """Registra el marker `slow` (corridas SUMO de 24h del perfil-día)."""
    config.addinivalue_line(
        "markers",
        "slow: corrida SUMO larga (p. ej. perfil-día 24h); deseleccionar con -m 'not slow'.",
    )
