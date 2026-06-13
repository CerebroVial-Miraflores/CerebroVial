"""Unit tests para select_device (selección de device + banner de arranque).

torch se inyecta como módulo falso en sys.modules para correr sin torch
instalado: solo se ejercita la lógica de selección por disponibilidad
(cuda→mps→cpu, NUNCA por SO) y el banner (verde con GPU, ROJO a stderr en CPU
con el hint según plataforma).
"""
import sys
from types import SimpleNamespace

import pytest

from src.vision.infrastructure.detection import device as device_mod


def _fake_torch(*, cuda=False, mps=True, cuda_name="FakeGPU"):
    """torch falso con los flags de disponibilidad bajo control."""
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            get_device_name=lambda i: cuda_name,
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps),
        ),
    )


@pytest.fixture
def inject_torch(monkeypatch):
    def _inject(**kwargs):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(**kwargs))
    return _inject


def test_cuda_disponible_devuelve_cuda(inject_torch, capsys):
    inject_torch(cuda=True, mps=False)
    assert device_mod.select_device() == "cuda"
    out = capsys.readouterr()
    assert "device=cuda" in out.out
    assert "FakeGPU" in out.out
    assert out.err == ""  # sin banner rojo


def test_mps_disponible_devuelve_mps(inject_torch, capsys):
    inject_torch(cuda=False, mps=True)
    assert device_mod.select_device() == "mps"
    out = capsys.readouterr()
    assert "device=mps" in out.out
    assert out.err == ""


def test_cuda_gana_a_mps(inject_torch):
    # Prioridad cuda > mps cuando ambos están disponibles.
    inject_torch(cuda=True, mps=True)
    assert device_mod.select_device() == "cuda"


def test_sin_acelerador_cae_a_cpu_con_banner_rojo(inject_torch, capsys):
    inject_torch(cuda=False, mps=False)
    assert device_mod.select_device() == "cpu"
    out = capsys.readouterr()
    # Banner ROJO a stderr, no a stdout.
    assert "SIN ACCESO A GPU" in out.err
    assert device_mod._RED in out.err


def test_hint_mac_cuando_cpu_en_darwin(inject_torch, monkeypatch, capsys):
    inject_torch(cuda=False, mps=False)
    monkeypatch.setattr(device_mod.platform, "system", lambda: "Darwin")
    device_mod.select_device()
    assert "Docker sobre Mac" in capsys.readouterr().err


def test_hint_windows_cuando_cpu_en_windows(inject_torch, monkeypatch, capsys):
    inject_torch(cuda=False, mps=False)
    monkeypatch.setattr(device_mod.platform, "system", lambda: "Windows")
    device_mod.select_device()
    assert "--gpus all" in capsys.readouterr().err


def test_seleccion_es_por_disponibilidad_no_por_so(inject_torch, monkeypatch, capsys):
    # En Darwin pero CON mps disponible → mps (no cae a CPU por ser Mac).
    inject_torch(cuda=False, mps=True)
    monkeypatch.setattr(device_mod.platform, "system", lambda: "Darwin")
    assert device_mod.select_device() == "mps"
    assert capsys.readouterr().err == ""


def test_min_hz_y_n_cameras_en_el_banner(inject_torch, capsys):
    inject_torch(cuda=False, mps=False)
    device_mod.select_device(min_hz=15, n_cameras=11)
    err = capsys.readouterr().err
    assert "15Hz" in err
    assert "11 camaras" in err
