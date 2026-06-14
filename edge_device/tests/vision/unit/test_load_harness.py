"""Tests del harness de carga (topología B / 15Hz) — sin GPU, sin red, stub detector.

Cubren: el productor synthetic emite a ~target fps; la agregación de métricas
(fps, p50/p95, drops, batch) es correcta sobre datos sintéticos conocidos; un smoke
run N=2 completa y produce la tabla con las columnas esperadas; el módulo es
importable sin forzar CUDA/red.
"""
import pytest

from src.vision.tooling.load_harness import (
    HarnessMetrics,
    StubDetector,
    SyntheticSource,
    _percentile,
    format_table,
    run_one,
)


# ---- SyntheticSource: pacing a target fps -----------------------------


def test_synthetic_source_emits_at_target_fps():
    # Reloj y sleep fakes: la fuente debe pedir dormir ~1/fps entre frames.
    t = {"now": 0.0}
    slept = []

    def clock():
        return t["now"]

    def sleep(d):
        slept.append(d)
        t["now"] += d  # avanzar el reloj como si durmiera

    src = SyntheticSource(fps=15, width=8, height=8, clock=clock, sleep=sleep)
    f0 = src.read()
    f1 = src.read()
    f2 = src.read()

    assert (f0.id, f1.id, f2.id) == (0, 1, 2)
    assert f0.image.shape == (8, 8, 3)
    # Entre frames consecutivos se pace ~1/15s.
    assert slept[-1] == pytest.approx(1 / 15, rel=0.01)


# ---- Agregación de métricas -------------------------------------------


def test_percentile():
    vals = [10, 20, 30, 40, 50]
    assert _percentile(vals, 0.5) == 30
    assert _percentile([], 0.5) == 0.0
    assert _percentile(vals, 0.95) == pytest.approx(48.0)


def test_metrics_summary_computes_rates_and_percentiles():
    m = HarnessMetrics()
    # Ventana [0, 2]s. cam_0 con 30 salidas, cam_1 con 15 → fps 15 y 7.5.
    for i in range(30):
        m.record_demux("cam_0", latency_ms=100.0, t=i * (2.0 / 30))
    for i in range(15):
        m.record_demux("cam_1", latency_ms=200.0, t=i * (2.0 / 15))
    for i in range(10):
        m.record_batch(4, t=i * 0.2)

    s = m.summary(["cam_0", "cam_1"], device="stub", window_start=0.0, window_end=2.0, dropped=5)

    assert s["N"] == 2
    assert s["fps_mean"] == pytest.approx((15 + 7.5) / 2, rel=0.05)
    assert s["fps_min"] == pytest.approx(7.5, rel=0.05)
    assert s["batch_mean"] == pytest.approx(4.0)
    assert s["dropped"] == 5
    # 45 procesados + 5 dropeados → drop_rate = 5/50.
    assert s["drop_rate"] == pytest.approx(0.1, rel=0.01)
    # latencias mezcladas 100/200 → p50 entre ambas, p95 cerca de 200.
    assert 100.0 <= s["e2e_p50_ms"] <= 200.0
    assert s["e2e_p95_ms"] == pytest.approx(200.0)


def test_metrics_summary_counts_starved_camera_as_zero():
    """Una cámara sin salidas en la ventana cuenta como 0 fps (no se omite del min)."""
    m = HarnessMetrics()
    for i in range(20):
        m.record_demux("cam_0", 50.0, t=i * 0.05)
    s = m.summary(["cam_0", "cam_1"], "stub", 0.0, 1.0, 0)
    assert s["fps_min"] == 0.0  # cam_1 starved


def test_summary_ignores_events_outside_window():
    m = HarnessMetrics()
    m.record_demux("cam_0", 10.0, t=-1.0)   # antes (warmup)
    m.record_demux("cam_0", 10.0, t=5.0)    # después
    m.record_demux("cam_0", 10.0, t=0.5)    # dentro
    s = m.summary(["cam_0"], "stub", 0.0, 1.0, 0)
    assert s["infers_s"] == pytest.approx(1.0)  # 1 evento en 1s


# ---- Tabla ------------------------------------------------------------


def test_format_table_has_expected_columns():
    rows = [{
        "device": "stub", "N": 2, "fps_mean": 15.0, "fps_min": 14.0, "dropped": 0,
        "drop_rate": 0.0, "e2e_p50_ms": 5.0, "e2e_p95_ms": 9.0, "batch_mean": 2.0,
        "infers_s": 30.0,
    }]
    table = format_table(rows, source="synthetic")
    for col in ("device", "N", "fps_mean", "dropped", "e2e_p95_ms", "batch_mean", "infers_s"):
        assert col in table
    # synthetic se rotula como techo de inferencia (no N de deploy).
    assert "TECHO DE INFERENCIA" in table


# ---- Smoke run e2e (stub detector, synthetic) -------------------------


@pytest.mark.asyncio
async def test_smoke_run_one_produces_metrics_row():
    row = await run_one(
        2, fps=30, duration=0.4, warmup=0.1, source="synthetic",
        detector=StubDetector(), max_batch=8, max_wait=0.02, width=8, height=8,
    )
    # Completa y produce una fila con las columnas esperadas y device del detector.
    assert row["N"] == 2
    assert row["device"] == "stub"
    for col in ("fps_mean", "fps_min", "dropped", "drop_rate", "e2e_p50_ms",
                "e2e_p95_ms", "batch_mean", "infers_s"):
        assert col in row
    # Con stub instantáneo y 2 cámaras, el path corre y demuxea (fps > 0).
    assert row["fps_mean"] > 0
    # La tabla se arma.
    assert "device" in format_table([row], "synthetic")
