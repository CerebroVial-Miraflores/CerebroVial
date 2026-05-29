"""Agrega KPIs desde los outputs de una corrida sumo (CT-07.6).

KPIs producidos:
    mean_travel_time_s          (de tripinfo.parquet: mean duration)
    total_delay_s               (de tripinfo.parquet: sum timeLoss)
    throughput_veh_per_h        (count arrived / sim duration × 3600)
    max_queue_m_by_direction    (de lanearea.xml maxJamLengthInMeters)
    mean_queue_m_by_direction   (de lanearea.xml meanMaxJamLengthInMeters)

Inputs:
    tripinfo.parquet, summary.parquet (single-write Parquet ok)
    lanearea.xml (multi-interval — fallback XML de F1)

Sin TraCI live.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq


DIRECTIONS = ["N", "S", "E", "W"]
DET_BY_DIR = {
    "N": ["LA_N_0", "LA_N_1", "LA_N_2"],
    "S": ["LA_S_0", "LA_S_1", "LA_S_2"],
    "E": ["LA_E_0", "LA_E_1"],
    "W": ["LA_W_0", "LA_W_1"],
}
DET_DIR = {det: d for d, dets in DET_BY_DIR.items() for det in dets}


@dataclass
class KPIs:
    mode: str
    pattern: str
    seed: int
    sim_duration_s: float
    mean_travel_time_s: float
    total_delay_s: float
    throughput_veh_per_h: float
    max_queue_m_by_dir: dict[str, float] = field(default_factory=dict)
    mean_queue_m_by_dir: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "mode": self.mode,
            "pattern": self.pattern,
            "seed": self.seed,
            "sim_duration_s": self.sim_duration_s,
            "mean_travel_time_s": self.mean_travel_time_s,
            "total_delay_s": self.total_delay_s,
            "throughput_veh_per_h": self.throughput_veh_per_h,
        }
        for direc in DIRECTIONS:
            d[f"max_queue_m_{direc}"] = self.max_queue_m_by_dir.get(direc, 0.0)
            d[f"mean_queue_m_{direc}"] = self.mean_queue_m_by_dir.get(direc, 0.0)
        return d


def _to_float(val) -> float:
    """SUMO Parquet a veces serializa numéricos como string ("0.00"). Coerce."""
    if val is None or val == "":
        return 0.0
    return float(val)


def _read_tripinfo(parquet_path: Path) -> tuple[float, float, int]:
    """Retorna (mean_duration_s, total_timeLoss_s, n_arrived).

    SUMO 1.26 prefija columnas con 'tripinfo_' en Parquet.
    """
    table = pq.read_table(str(parquet_path))
    cols = set(table.column_names)
    dur_col = "tripinfo_duration" if "tripinfo_duration" in cols else "duration"
    tl_col = "tripinfo_timeLoss" if "tripinfo_timeLoss" in cols else "timeLoss"
    if dur_col not in cols:
        raise RuntimeError(f"tripinfo sin columna duration. cols={sorted(cols)}")

    durations = [_to_float(v) for v in table.column(dur_col).to_pylist()]
    time_losses = (
        [_to_float(v) for v in table.column(tl_col).to_pylist()]
        if tl_col in cols
        else [0.0] * len(durations)
    )
    mean_dur = sum(durations) / len(durations) if durations else 0.0
    return mean_dur, sum(time_losses), len(durations)


def _read_summary_duration(parquet_path: Path) -> float:
    """Última `step_time` de summary.parquet — duración real de la simulación."""
    table = pq.read_table(str(parquet_path))
    cols = set(table.column_names)
    time_col = "step_time" if "step_time" in cols else "time"
    if time_col not in cols:
        raise RuntimeError(f"summary sin columna time/step_time. cols={sorted(cols)}")
    times = [_to_float(v) for v in table.column(time_col).to_pylist()]
    return max(times) if times else 0.0


def _read_lanearea_queue_per_dir(
    lanearea_xml: Path,
) -> tuple[dict[str, float], dict[str, float]]:
    """Retorna (max_queue_m_per_dir, mean_queue_m_per_dir).

    Agrega maxJamLengthInMeters sobre los lanes de la dirección (suma)
    por bucket; max y mean sobre los buckets.
    """
    root = ET.parse(lanearea_xml).getroot()
    # {direction: {bucket_t: sum_max_jam}}
    per_bucket: dict[str, dict[float, float]] = {d: {} for d in DIRECTIONS}
    for interval in root.findall("interval"):
        det_id = interval.attrib["id"]
        d = DET_DIR.get(det_id)
        if d is None:
            continue
        bucket_t = float(interval.attrib["end"])
        max_jam = float(interval.attrib.get("maxJamLengthInMeters", "0"))
        per_bucket[d][bucket_t] = per_bucket[d].get(bucket_t, 0.0) + max_jam

    max_per_dir = {}
    mean_per_dir = {}
    for d in DIRECTIONS:
        vals = list(per_bucket[d].values())
        if vals:
            max_per_dir[d] = max(vals)
            mean_per_dir[d] = sum(vals) / len(vals)
        else:
            max_per_dir[d] = 0.0
            mean_per_dir[d] = 0.0
    return max_per_dir, mean_per_dir


def collect(
    mode: str,
    pattern: str,
    seed: int,
    out_dir: Path,
) -> KPIs:
    """Agrega KPIs de una corrida (out_dir contiene summary.parquet, tripinfo.parquet, lanearea.xml)."""
    summary = out_dir / "summary.parquet"
    tripinfo = out_dir / "tripinfo.parquet"
    lanearea = out_dir / "lanearea.xml"
    if not (summary.exists() and tripinfo.exists() and lanearea.exists()):
        raise FileNotFoundError(
            f"Faltan outputs en {out_dir}: summary={summary.exists()} "
            f"tripinfo={tripinfo.exists()} lanearea={lanearea.exists()}"
        )

    sim_duration = _read_summary_duration(summary)
    mean_dur, total_tl, n_arrived = _read_tripinfo(tripinfo)
    max_q, mean_q = _read_lanearea_queue_per_dir(lanearea)

    throughput = (n_arrived * 3600.0 / sim_duration) if sim_duration > 0 else 0.0

    return KPIs(
        mode=mode,
        pattern=pattern,
        seed=seed,
        sim_duration_s=sim_duration,
        mean_travel_time_s=mean_dur,
        total_delay_s=total_tl,
        throughput_veh_per_h=throughput,
        max_queue_m_by_dir=max_q,
        mean_queue_m_by_dir=mean_q,
    )
