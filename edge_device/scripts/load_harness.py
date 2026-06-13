"""Entrypoint del harness de carga (topología B / 15Hz).

Uso (local smoke, sin pesos):
    python scripts/load_harness.py --sweep 1,2,4 --source synthetic --stub-detector \
        --duration 10 --warmup 3

Uso (run fiel en CUDA, lo lanza el operador):
    python scripts/load_harness.py --sweep 1,2,4,8,11 --source hls:<url1,url2> \
        --fps 15 --duration 45 --warmup 8 --max-batch 16 --max-wait 0.05
    # (en paralelo: nvidia-smi -l 1 para la utilización de GPU)

La lógica vive en src/vision/tooling/load_harness.py (importable/testeable).
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.tooling.load_harness import main  # noqa: E402

if __name__ == "__main__":
    main()
