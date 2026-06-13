"""Selección del device de inferencia (cuda/mps/cpu) + banner de arranque.

Se elige por DISPONIBILIDAD de hardware, NO por sistema operativo:
`torch.cuda.is_available()` → `cuda`; si no, `torch.backends.mps.is_available()`
→ `mps`; si no, `cpu`. El SO se usa SOLO para el texto del hint del banner.

Por qué importa el matiz: en Mac, que MPS sirva depende de nativo-vs-Docker
(Docker en Mac NO ve la GPU de Apple). Ese es justo el caso que el banner rojo
tiene que cazar: si caés a CPU dentro de Docker en Mac, conviene levantar el
edge_device en local (fuera de Docker) para usar MPS.

`select_device()` se llama UNA vez en el arranque del server
(`scripts/run_server.py`). Resolverlo al boot NO carga el modelo: `is_available()`
es solo un probe de hardware, así que respeta el diseño on-demand (C1/D1) —
el YOLO sigue cargando lazy en la 1ª cámara; acá solo se mueven el probe y el
banner al boot.
"""
import platform
import sys

_RED = "\033[1;31m"
_GREEN = "\033[1;32m"
_RESET = "\033[0m"


def select_device(min_hz: int = 15, n_cameras: int = 11) -> str:
    """Devuelve el device de inferencia disponible e imprime un banner.

    - cuda/mps disponible → línea verde a stdout.
    - sin acelerador → banner ROJO a stderr con hint según plataforma, porque a
      `min_hz`Hz × `n_cameras` cámaras la CPU no rinde.

    `min_hz`/`n_cameras` son solo para el texto del banner (no cambian la lógica).
    """
    import torch

    if torch.cuda.is_available():
        print(f"{_GREEN}[edge_device] GPU NVIDIA ({torch.cuda.get_device_name(0)}) -> device=cuda{_RESET}")
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        print(f"{_GREEN}[edge_device] GPU de Apple -> device=mps{_RESET}")
        return "mps"

    system = platform.system()  # 'Darwin' | 'Windows' | 'Linux'
    if system == "Darwin":
        hint = ("Estas en Docker sobre Mac, que NO da acceso a la GPU de Apple.\n"
                "  -> Levanta el edge_device en LOCAL (fuera de Docker) para usar MPS.")
    elif system == "Windows":
        hint = ("No se detecto GPU NVIDIA. Revisa drivers y, si usas Docker,\n"
                "  -> corre con `--gpus all` (+ nvidia-container-toolkit).")
    else:
        hint = "No se detecto GPU NVIDIA. Revisa drivers/CUDA; en Docker usa `--gpus all`."

    bar = "=" * 68
    print(f"{_RED}\n{bar}\n"
          f"  !!!  SIN ACCESO A GPU  -  el edge_device va a correr en CPU  !!!\n"
          f"  A {min_hz}Hz x {n_cameras} camaras esto NO rinde en CPU.\n"
          f"  {hint}\n{bar}{_RESET}\n", file=sys.stderr)
    return "cpu"
