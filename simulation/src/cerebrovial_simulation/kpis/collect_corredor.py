"""Agrega KPIs y veredicto de spillback de una corrida baseline del corredor Larco
(ETAPA 2).

A diferencia de ``collect.py`` (direcciones N/S/E/W de la red sintética
``miraflores_4way``), este collector trabaja sobre la topología REAL del corredor:
cola por-edge (con foco en los links internos entre TLS), ventana post-warmup,
teleports/descartes y un veredicto explícito spillback-útil / gridlock / free-flow.

Reúsa los lectores genéricos de tripinfo/summary de ``collect.py``.

Outputs leídos (en out_dir):
    tripinfo.parquet, summary.parquet, lanearea.xml

Uso:
    python -m cerebrovial_simulation.kpis.collect_corredor --out <dir> \
        [--warmup 600] [--params conf/corredor_larco/demand_params.yaml]
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from .collect import _read_summary_duration, _to_float


# --- Topología instrumentada (edge → detectores, largo del edge en metros) ---
EDGE_LEN = {
    "129466113#0": 247.55,   # aproximación principal S→N (fuente sur → Benavides)
    "279893875#1": 96.92,    # link interno Benavides→Schell
    "279893875#2": 48.00,    # aproximación a Schell
    "279893875#4": 48.26,    # link interno Schell→Diez Canseco
    "344159559#2": 153.34,   # transversal Benavides Este
    "430180649": 141.46,     # transversal Schell Este
}
DET_BY_EDGE = {
    "129466113#0": ["e2_larcoS_0", "e2_larcoS_1", "e2_larcoS_2"],
    "279893875#1": ["e2_benSch_0", "e2_benSch_1", "e2_benSch_2"],
    "279893875#2": ["e2_tarSch_0", "e2_tarSch_1", "e2_tarSch_2"],
    "279893875#4": ["e2_schDc_0", "e2_schDc_1", "e2_schDc_2"],
    "344159559#2": ["e2_benE_0", "e2_benE_1", "e2_benE_2"],
    "430180649": ["e2_schE_0", "e2_schE_1", "e2_schE_2"],
}
DET_EDGE = {det: e for e, dets in DET_BY_EDGE.items() for det in dets}
# Edges-label legible.
EDGE_LABEL = {
    "129466113#0": "larcoS_approach",
    "279893875#1": "benavides_to_schell(INT)",
    "279893875#2": "tarata_to_schell",
    "279893875#4": "schell_to_diezcanseco(INT)",
    "344159559#2": "benavides_E_approach",
    "430180649": "schell_E_approach",
}
# Links internos del corredor donde mirar spillback.
INTERNAL_LINKS = ["279893875#1", "279893875#2", "279893875#4"]
# Umbral de spillback: cola ≥ SPILL_FRAC × largo del edge.
SPILL_FRAC = 0.9
SPILL_MIN_BUCKETS = 3  # buckets consecutivos para confirmar spillback sostenido


@dataclass
class EdgeQueue:
    edge: str
    label: str
    length_m: float
    max_queue_m: float          # pico de cola (max sobre carriles, max sobre buckets)
    mean_queue_m: float         # cola media en la ventana
    fill_frac_peak: float       # max_queue_m / length_m
    spillback: bool             # alcanzó SPILL_FRAC sostenido SPILL_MIN_BUCKETS buckets
    max_consec_spill_buckets: int


@dataclass
class CorredorKPIs:
    out_dir: str
    sim_duration_s: float
    warmup_s: int
    # throughput / inserción
    generated_veh: float        # demanda teórica (Σ vehsPerHour × dur / 3600)
    arrived_veh: int            # completaron viaje (tripinfo, full-run)
    throughput_veh_per_h: float
    completion_ratio: float     # arrived / generated
    teleports_total: int
    discarded_total: int        # vehículos que no pudieron insertarse (backlog)
    final_running: int
    final_ended: int
    # demora / tiempo de viaje (full-run y post-warmup)
    net_mean_travel_time_s: float
    net_mean_timeloss_s: float
    corridor_mean_travel_time_s: float       # solo vehículos f_larco_S_N (S→N completo)
    corridor_n: int
    net_mean_travel_time_s_postwarm: float
    corridor_mean_travel_time_s_postwarm: float
    # colas
    edge_queues: list = field(default_factory=list)
    # veredicto
    verdict: str = ""
    verdict_reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edge_queues"] = [asdict(eq) if not isinstance(eq, dict) else eq for eq in self.edge_queues]
        return d


def _read_tripinfo_rich(parquet_path: Path, warmup_s: int):
    """Lee tripinfo con id/depart/duration/timeLoss.

    Retorna dict con agregados full-run y post-warmup, de red y del corredor
    (vehículos cuyo id arranca con 'f_larco_S_N').
    """
    table = pq.read_table(str(parquet_path))
    cols = set(table.column_names)

    def col(name: str):
        c = f"tripinfo_{name}"
        return c if c in cols else (name if name in cols else None)

    id_c, dur_c, tl_c, dep_c = col("id"), col("duration"), col("timeLoss"), col("depart")
    if dur_c is None:
        raise RuntimeError(f"tripinfo sin columna duration. cols={sorted(cols)}")

    ids = table.column(id_c).to_pylist() if id_c else [""] * table.num_rows
    durs = [_to_float(v) for v in table.column(dur_c).to_pylist()]
    tls = [_to_float(v) for v in table.column(tl_c).to_pylist()] if tl_c else [0.0] * len(durs)
    deps = [_to_float(v) for v in table.column(dep_c).to_pylist()] if dep_c else [0.0] * len(durs)

    def _agg(mask):
        sel = [d for d, m in zip(durs, mask) if m]
        tsel = [t for t, m in zip(tls, mask) if m]
        n = len(sel)
        return (
            n,
            (sum(sel) / n if n else 0.0),
            (sum(tsel) / n if n else 0.0),
        )

    all_mask = [True] * len(durs)
    corr_mask = [str(i).startswith("f_larco_S_N.") for i in ids]
    postwarm_mask = [dp >= warmup_s for dp in deps]
    corr_pw_mask = [c and p for c, p in zip(corr_mask, postwarm_mask)]

    n_all, net_tt, net_tl = _agg(all_mask)
    n_corr, corr_tt, _ = _agg(corr_mask)
    _, net_tt_pw, _ = _agg(postwarm_mask)
    _, corr_tt_pw, _ = _agg(corr_pw_mask)
    return {
        "n_arrived": n_all,
        "net_tt": net_tt,
        "net_tl": net_tl,
        "corr_tt": corr_tt,
        "corr_n": n_corr,
        "net_tt_pw": net_tt_pw,
        "corr_tt_pw": corr_tt_pw,
    }


def _read_summary_extras(parquet_path: Path) -> dict:
    """Suma teleports/descartes y captura los conteos finales de la summary."""
    table = pq.read_table(str(parquet_path))
    cols = set(table.column_names)

    def series(name: str):
        c = f"step_{name}" if f"step_{name}" in cols else (name if name in cols else None)
        return [_to_float(v) for v in table.column(c).to_pylist()] if c else []

    teleports = series("teleports")
    discarded = series("discarded")
    running = series("running")
    ended = series("ended")
    return {
        "teleports_total": int(sum(teleports)),
        "discarded_total": int(sum(discarded)),
        "final_running": int(running[-1]) if running else 0,
        "final_ended": int(ended[-1]) if ended else 0,
    }


def _read_queues(lanearea_xml: Path, warmup_s: int) -> list[EdgeQueue]:
    """Cola por-edge en la ventana post-warmup.

    Por bucket: cola del edge = max de maxJamLengthInMeters sobre sus carriles.
    Sobre buckets (end > warmup): pico y media. Spillback = SPILL_FRAC×L sostenido.
    """
    root = ET.parse(lanearea_xml).getroot()
    # edge -> {bucket_end: max_jam_over_lanes}
    per_bucket: dict[str, dict[float, float]] = {e: {} for e in DET_BY_EDGE}
    for interval in root.findall("interval"):
        det_id = interval.attrib.get("id", "")
        edge = DET_EDGE.get(det_id)
        if edge is None:
            continue
        end = float(interval.attrib.get("end", "0"))
        if end <= warmup_s:
            continue
        jam = float(interval.attrib.get("maxJamLengthInMeters", "0"))
        cur = per_bucket[edge].get(end)
        per_bucket[edge][end] = jam if cur is None else max(cur, jam)

    result = []
    for edge, length in EDGE_LEN.items():
        buckets = sorted(per_bucket[edge].items())
        vals = [v for _, v in buckets]
        max_q = max(vals) if vals else 0.0
        mean_q = sum(vals) / len(vals) if vals else 0.0
        # buckets consecutivos sobre el umbral
        thr = SPILL_FRAC * length
        consec = best = 0
        for v in vals:
            if v >= thr:
                consec += 1
                best = max(best, consec)
            else:
                consec = 0
        result.append(EdgeQueue(
            edge=edge,
            label=EDGE_LABEL[edge],
            length_m=length,
            max_queue_m=round(max_q, 1),
            mean_queue_m=round(mean_q, 1),
            fill_frac_peak=round(max_q / length, 2) if length else 0.0,
            spillback=best >= SPILL_MIN_BUCKETS,
            max_consec_spill_buckets=best,
        ))
    return result


def _classify(kpis: CorredorKPIs) -> tuple[str, str]:
    """Veredicto de gate: spillback_util / gridlock_total / free_flow / congestion_parcial."""
    spill_internal = [eq for eq in kpis.edge_queues
                      if eq.edge in INTERNAL_LINKS and eq.spillback]
    any_spill = len(spill_internal) > 0
    # gridlock: throughput colapsado + teleports/descartes altos + running atascado
    gridlock = (
        kpis.completion_ratio < 0.5
        and (kpis.teleports_total > 20 or kpis.discarded_total > 0.3 * max(kpis.generated_veh, 1))
    )
    # free-flow: sin spillback en ningún link interno y colas chicas
    max_fill_internal = max(
        (eq.fill_frac_peak for eq in kpis.edge_queues if eq.edge in INTERNAL_LINKS),
        default=0.0,
    )
    if gridlock:
        return ("gridlock_total", (
            f"completion_ratio={kpis.completion_ratio:.2f} (<0.5), "
            f"teleports={kpis.teleports_total}, discarded={kpis.discarded_total}. "
            "No medible → BAJAR demanda."))
    if any_spill:
        links = ", ".join(f"{eq.label}({eq.max_consec_spill_buckets}b)" for eq in spill_internal)
        return ("spillback_util", (
            f"spillback sostenido en link(s) interno(s): {links}; "
            f"throughput={kpis.throughput_veh_per_h:.0f} veh/h con arrived>0. "
            "GATE CONFIRMADO."))
    if max_fill_internal < 0.5 and kpis.teleports_total == 0:
        return ("free_flow", (
            f"sin spillback (relleno pico link interno {max_fill_internal:.0%} < 50%), "
            "teleports=0. Causa probable: descarga en 3 carriles, no 2 → SUBIR demanda."))
    return ("congestion_parcial", (
        f"colas presentes (relleno pico link interno {max_fill_internal:.0%}) pero sin "
        "spillback sostenido ≥0.9·L. Revisar serie temporal; posible subir demanda al margen."))


def collect_corredor(out_dir: Path, warmup_s: int, params_path: Path | None) -> CorredorKPIs:
    summary = out_dir / "summary.parquet"
    tripinfo = out_dir / "tripinfo.parquet"
    lanearea = out_dir / "lanearea.xml"
    for p in (summary, tripinfo, lanearea):
        if not p.exists():
            raise FileNotFoundError(f"Falta output: {p}")

    sim_dur = _read_summary_duration(summary)
    trip = _read_tripinfo_rich(tripinfo, warmup_s)
    extras = _read_summary_extras(summary)
    queues = _read_queues(lanearea, warmup_s)

    # demanda teórica generada
    generated = 0.0
    if params_path and params_path.exists():
        with open(params_path) as f:
            cfg = yaml.safe_load(f)
        dur = cfg.get("duration_s", sim_dur)
        generated = sum(v for v in cfg.get("flows_vph", {}).values() if v > 0) * dur / 3600.0

    arrived = trip["n_arrived"]
    throughput = (arrived * 3600.0 / sim_dur) if sim_dur > 0 else 0.0
    completion = (arrived / generated) if generated > 0 else 0.0

    kpis = CorredorKPIs(
        out_dir=str(out_dir),
        sim_duration_s=sim_dur,
        warmup_s=warmup_s,
        generated_veh=round(generated, 1),
        arrived_veh=arrived,
        throughput_veh_per_h=round(throughput, 1),
        completion_ratio=round(completion, 3),
        teleports_total=extras["teleports_total"],
        discarded_total=extras["discarded_total"],
        final_running=extras["final_running"],
        final_ended=extras["final_ended"],
        net_mean_travel_time_s=round(trip["net_tt"], 1),
        net_mean_timeloss_s=round(trip["net_tl"], 1),
        corridor_mean_travel_time_s=round(trip["corr_tt"], 1),
        corridor_n=trip["corr_n"],
        net_mean_travel_time_s_postwarm=round(trip["net_tt_pw"], 1),
        corridor_mean_travel_time_s_postwarm=round(trip["corr_tt_pw"], 1),
        edge_queues=queues,
    )
    kpis.verdict, kpis.verdict_reason = _classify(kpis)
    return kpis


def _print_report(k: CorredorKPIs) -> None:
    print("=" * 72)
    print("BASELINE corredor Larco — peak S→N | control fijo | seed fija")
    print(f"  out_dir={k.out_dir}  sim={k.sim_duration_s:.0f}s  warmup={k.warmup_s}s")
    print("-" * 72)
    print("THROUGHPUT / INSERCIÓN")
    print(f"  generados (demanda)     : {k.generated_veh:.0f} veh")
    print(f"  completados (arrived)   : {k.arrived_veh} veh   completion={k.completion_ratio:.2f}")
    print(f"  throughput              : {k.throughput_veh_per_h:.0f} veh/h")
    print(f"  teleports / discarded   : {k.teleports_total} / {k.discarded_total}")
    print(f"  final running / ended   : {k.final_running} / {k.final_ended}")
    print("DEMORA / TIEMPO DE VIAJE")
    print(f"  red  TT medio           : {k.net_mean_travel_time_s:.1f}s   "
          f"(post-warmup {k.net_mean_travel_time_s_postwarm:.1f}s)")
    print(f"  red  timeLoss medio     : {k.net_mean_timeloss_s:.1f}s")
    print(f"  corredor S→N TT medio   : {k.corridor_mean_travel_time_s:.1f}s   "
          f"(post-warmup {k.corridor_mean_travel_time_s_postwarm:.1f}s, n={k.corridor_n})")
    print("COLAS POR EDGE (ventana post-warmup; (INT)=link interno entre TLS)")
    for eq in k.edge_queues:
        flag = "  <<< SPILLBACK" if eq.spillback else ""
        print(f"  {eq.label:28s} L={eq.length_m:6.1f}m  max={eq.max_queue_m:6.1f}m "
              f"({eq.fill_frac_peak:.0%})  mean={eq.mean_queue_m:6.1f}m{flag}")
    print("-" * 72)
    print(f"VEREDICTO: {k.verdict.upper()}")
    print(f"  {k.verdict_reason}")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="dir con summary/tripinfo/lanearea")
    ap.add_argument("--warmup", type=int, default=600, help="segundos de warmup a descartar")
    ap.add_argument("--params", type=Path, default=None,
                    help="demand_params.yaml para calcular demanda generada")
    ap.add_argument("--json", type=Path, default=None, help="ruta opcional para volcar KPIs JSON")
    args = ap.parse_args()

    kpis = collect_corredor(args.out, args.warmup, args.params)
    _print_report(kpis)
    if args.json:
        args.json.write_text(json.dumps(kpis.to_dict(), indent=2, ensure_ascii=False))
        print(f"  KPIs JSON → {args.json}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
