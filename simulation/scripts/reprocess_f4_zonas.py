"""Reproceso y consolidación de F4 (DHU-030): produce las DOS tablas del reporte a partir de
los crudos de ``run_multiseed_f4_zonas.py`` (en runs/, gitignoreado):

  (1) FRANJAS GLOBALES — net_timeLoss de red entera por franja B2 (AM/valle/PM/día), fijo vs
      adaptativo, por seed + agregado, con RD% por franja. APROXIMADO: agrupa por cohorte de
      SALIDA (filtra tripinfo por tripinfo_depart); un auto que sale al final de una franja
      derrama demora a la siguiente. Fuente: tripinfo.parquet de cada brazo.

  (2) KPI POR ZONA — por cada uno de los 3 cruces PMU, por franja, agrega el edgeData de sus
      aproches: timeLoss/waitingTime totales (suma sobre edges×horas), density/speed medios
      (ponderados por sampledSeconds), throughput (entered, suma), fijo vs adaptativo, RD% por
      cruce×franja. EXACTO por franja: edgeData mide el INTERVALO, no la cohorte. Fuente:
      edgedata_zonas_pmu.xml de cada brazo.

Sanity: el net_timeLoss global del 'día completo' por seed debe reproducir el de F5
(metrics.json / multiseed_54tls_resultado.csv) — se imprime el cotejo.

Salidas VERSIONABLES (respaldo citable): f4_franjas_global.csv, f4_zonas_kpi.csv,
f4_resultado.json en conf/miraflores_red_completa/. Los crudos quedan en runs/.

Uso:
    python reprocess_f4_zonas.py                 # todos los seeds presentes en runs/f4_*
    python reprocess_f4_zonas.py --seeds 42      # solo seed 42 (smoke)
"""
from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
EXP_DIR = SIM_ROOT / "conf" / "miraflores_red_completa"
RUNS = EXP_DIR / "runs"
CONF = EXP_DIR / "zonas_pmu_f4.json"
F5_CSV = EXP_DIR / "multiseed_54tls_resultado.csv"

CANONICAL_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# Lectura interpretativa del run CANÓNICO de 10 seeds (42-51). El corazón del reporte es el
# contraste waitingTime-vs-timeLoss en los D/F — las tablas crudas no lo capturan, así que el
# artefacto consolidado lo lleva. Solo se emite si están los 10 seeds (ver main()).
HALLAZGOS = {
    "titular_global": "RD% día completo +20,67% ± 1,81 (10 seeds, pareado) = el titular de F5 "
        "exacto; F4 mide el MISMO experimento sin perturbarlo (edgeData es pura medición).",
    "franjas": "El valle (10-17) es la franja más fuerte y estable (+22,3% ± 0,48); los picos "
        "rinden algo menos y con más dispersión (AM +19,8%, PM +18,5%). Sentido físico: en "
        "saturación de pico hay menos margen para que el re-timing ayude; en carga media el "
        "adaptativo tiene espacio para optimizar.",
    "arequipa_domina": "arequipa_angamos (E/H, el cruce peor-saturado) confirma la priorización: "
        "+30,7% ± 2,3 de timeLoss en el día, parejo en las 4 franjas, SD ajustada, throughput "
        "pareado. El sistema concentra el beneficio donde la saturación es peor.",
    "costo_local_d_f": "Los dos cruces D/F (28julio_lapaz, ricardopalma_paseo) tienen costo local "
        "REAL en timeLoss (−17,5% y −14,0% en el día), signo consistente en los 10 seeds, no "
        "artefacto de un seed. SD ancha (9-28) por ser cruces chicos: las celdas de pico son no "
        "concluyentes (SD > |media|), el día completo está a ~1,5-1,9 SD de cero (sugestivo).",
    "trade_off_cola_vs_fluidez": "Hallazgo central (lo destrabó waitingTime): en waitingTime los "
        "TRES cruces MEJORAN (+13% a +42%), D/F incluidos. O sea el re-timing reduce el tiempo "
        "PARADO en cola en todos; en los D/F el timeLoss total empeora porque los autos pasan más "
        "tiempo moviéndose lento (verdes más cortos/frecuentes → menos cola pero más stop-and-go): "
        "se cambia espera-detenido por circulación-lenta, no se penaliza la cola.",
    "conclusion": "El control adaptativo reduce la cola (waitingTime) en los tres cruces "
        "saturados. El tiempo total (timeLoss) solo mejora en el más saturado (arequipa, E/H); en "
        "los dos D/F empeora levemente por el cambio espera→circulación-lenta. El neto de red es "
        "+20,67% porque arequipa domina en magnitud y el resto de los 54 nodos aporta. El sistema "
        "prioriza donde la saturación es peor, a un costo local de fluidez en cruces secundarios — "
        "sin penalizar la cola ni siquiera ahí.",
    "sanity": "Byte-idéntico a F5 en los 10 seeds (incluidos 47-50 que F5 había perdido, ahora "
        "regenerados). Throughput pareado en las 120 celdas (máx 10 veh de divergencia sobre "
        "~1000) → las diferencias de demora son reasignación real de verde, no volumen.",
}

CONFD = json.loads(CONF.read_text())
FRANJAS = CONFD["franjas"]
ZONAS = CONFD["zonas"]


# ----------------------------- (1) FRANJAS GLOBALES (tripinfo) -----------------------------
def global_franjas_for_seed(seed: int) -> list[dict]:
    fijo = RUNS / f"f4_fijo_seed{seed:03d}" / "tripinfo.parquet"
    adap = RUNS / f"f4_adaptivo_seed{seed:03d}_end86400" / "tripinfo.parquet"
    if not (fijo.exists() and adap.exists()):
        return []

    def per_franja(p: Path) -> dict:
        t = pq.read_table(p, columns=["tripinfo_depart", "tripinfo_timeLoss"])
        dep = pc.cast(t.column("tripinfo_depart"), "float64")
        tl = pc.cast(t.column("tripinfo_timeLoss"), "float64")
        out = {}
        for fr in FRANJAS:
            mask = pc.and_(pc.greater_equal(dep, fr["begin_s"]), pc.less(dep, fr["end_s"]))
            sub = pc.filter(tl, mask)
            out[fr["nombre"]] = (sub.length(), pc.mean(sub).as_py() if sub.length() else None)
        return out

    f, a = per_franja(fijo), per_franja(adap)
    rows = []
    for fr in FRANJAS:
        fn, fm = f[fr["nombre"]]
        an, am = a[fr["nombre"]]
        rd = (fm - am) / fm * 100 if (fm and am is not None) else None
        rows.append({
            "seed": seed, "franja": fr["nombre"],
            "fijo_net_timeloss_s": round(fm, 4) if fm else None,
            "adap_net_timeloss_s": round(am, 4) if am is not None else None,
            "rd_pct": round(rd, 4) if rd is not None else None,
            "n_veh_fijo": fn, "n_veh_adap": an,
        })
    return rows


# ------------------------------- (2) KPI POR ZONA (edgeData) -------------------------------
def _parse_edgedata(p: Path) -> dict:
    """{edge_id: [(begin, end, {attr:float}), ...]} de un edgedata_zonas_pmu.xml."""
    root = ET.parse(p).getroot()
    by_edge: dict[str, list] = {}
    for interval in root.findall("interval"):
        b, e = float(interval.get("begin")), float(interval.get("end"))
        for edge in interval.findall("edge"):
            eid = edge.get("id")
            attrs = {}
            for k in ("timeLoss", "waitingTime", "density", "speed", "sampledSeconds",
                      "entered", "left"):
                v = edge.get(k)
                attrs[k] = float(v) if v is not None else 0.0
            by_edge.setdefault(eid, []).append((b, e, attrs))
    return by_edge


def _agg_zone_franja(by_edge: dict, edges: list[str], fr: dict) -> dict:
    """Agrega los aproches de una zona sobre las horas de una franja."""
    tl = wt = ent = lef = ss = 0.0
    sp_w = den_w = 0.0  # ponderados por sampledSeconds
    for eid in edges:
        for (b, e, a) in by_edge.get(eid, []):
            if b >= fr["begin_s"] and e <= fr["end_s"]:
                tl += a["timeLoss"]
                wt += a["waitingTime"]
                ent += a["entered"]
                lef += a["left"]
                ss += a["sampledSeconds"]
                sp_w += a["speed"] * a["sampledSeconds"]
                den_w += a["density"] * a["sampledSeconds"]
    speed = sp_w / ss if ss else None        # m/s, ponderado por tiempo-vehículo
    density = den_w / ss if ss else None      # veh/km, ponderado por tiempo-vehículo
    return {"timeloss_total_s": tl, "waiting_total_s": wt, "throughput_entered": ent,
            "left": lef, "speed_mean_ms": speed, "density_mean": density}


def zonas_kpi_for_seed(seed: int) -> list[dict]:
    fijo = RUNS / f"f4_fijo_seed{seed:03d}" / "edgedata_zonas_pmu.xml"
    adap = RUNS / f"f4_adaptivo_seed{seed:03d}_end86400" / "edgedata_zonas_pmu.xml"
    if not (fijo.exists() and adap.exists()):
        return []
    bf, ba = _parse_edgedata(fijo), _parse_edgedata(adap)
    rows = []
    for z in ZONAS:
        for fr in FRANJAS:
            f = _agg_zone_franja(bf, z["edges_aproche"], fr)
            a = _agg_zone_franja(ba, z["edges_aproche"], fr)

            def _rd(fv, av):  # RD% = (fijo-adap)/fijo*100; >0 = adaptativo mejora
                return round((fv - av) / fv * 100, 4) if fv else None

            def _per_veh(total, thru):  # demora por vehículo-paso por el aproche
                return total / thru if thru else None

            # por-vehículo: total / throughput (vehículos que entraron a los aproches).
            # Normaliza el total (que mezcla demora × volumen) → comparable entre zonas de
            # distinto tamaño; es la "demora que siente cada conductor" en el cruce.
            f_tl_pv = _per_veh(f["timeloss_total_s"], f["throughput_entered"])
            a_tl_pv = _per_veh(a["timeloss_total_s"], a["throughput_entered"])
            f_wt_pv = _per_veh(f["waiting_total_s"], f["throughput_entered"])
            a_wt_pv = _per_veh(a["waiting_total_s"], a["throughput_entered"])
            rows.append({
                "seed": seed, "zona": z["nombre"], "nivel_los": z["nivel_los"],
                "franja": fr["nombre"],
                # --- demora: total, por-vehículo, waiting (total + por-vehículo) ---
                "fijo_timeloss_total_s": round(f["timeloss_total_s"], 2),
                "adap_timeloss_total_s": round(a["timeloss_total_s"], 2),
                "rd_pct_timeloss": _rd(f["timeloss_total_s"], a["timeloss_total_s"]),
                "fijo_timeloss_per_veh_s": round(f_tl_pv, 4) if f_tl_pv is not None else None,
                "adap_timeloss_per_veh_s": round(a_tl_pv, 4) if a_tl_pv is not None else None,
                "rd_pct_timeloss_per_veh": _rd(f_tl_pv, a_tl_pv) if f_tl_pv else None,
                "fijo_waiting_total_s": round(f["waiting_total_s"], 2),
                "adap_waiting_total_s": round(a["waiting_total_s"], 2),
                "rd_pct_waiting": _rd(f["waiting_total_s"], a["waiting_total_s"]),
                "fijo_waiting_per_veh_s": round(f_wt_pv, 4) if f_wt_pv is not None else None,
                "adap_waiting_per_veh_s": round(a_wt_pv, 4) if a_wt_pv is not None else None,
                "rd_pct_waiting_per_veh": _rd(f_wt_pv, a_wt_pv) if f_wt_pv else None,
                # --- contexto ---
                "fijo_density_mean": round(f["density_mean"], 4) if f["density_mean"] else None,
                "adap_density_mean": round(a["density_mean"], 4) if a["density_mean"] else None,
                "fijo_speed_mean_ms": round(f["speed_mean_ms"], 4) if f["speed_mean_ms"] else None,
                "adap_speed_mean_ms": round(a["speed_mean_ms"], 4) if a["speed_mean_ms"] else None,
                "fijo_throughput": int(f["throughput_entered"]),
                "adap_throughput": int(a["throughput_entered"]),
            })
    return rows


# --------------------------------- agregación entre seeds ---------------------------------
def _mean_sd(xs: list[float]) -> tuple[float | None, float | None]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd


def _f5_sanity(seeds: list[int], global_rows: list[dict]) -> list[dict]:
    if not F5_CSV.exists():
        return []
    f5 = {int(r["seed"]): r for r in csv.DictReader(open(F5_CSV, newline=""))}
    out = []
    for s in seeds:
        gr = next((r for r in global_rows
                   if r["seed"] == s and r["franja"] == "dia_completo"), None)
        if gr and s in f5:
            out.append({"seed": s, "f4_dia_fijo": gr["fijo_net_timeloss_s"],
                        "f5_fijo": float(f5[s]["fijo_timeloss"]),
                        "f4_dia_adap": gr["adap_net_timeloss_s"],
                        "f5_adap": float(f5[s]["adap_timeloss"])})
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Reproceso/consolidación F4 (franjas + zonas).")
    ap.add_argument("--seeds", default=None, help="rango 'a-b' o lista; default = los presentes")
    args = ap.parse_args(argv)

    if args.seeds:
        if "-" in args.seeds:
            a, b = args.seeds.split("-")
            seeds = list(range(int(a), int(b) + 1))
        else:
            seeds = [int(x) for x in args.seeds.split(",")]
    else:
        seeds = sorted(int(d.name.split("seed")[1]) for d in RUNS.glob("f4_fijo_seed*"))

    global_rows, zonas_rows = [], []
    for s in seeds:
        global_rows.extend(global_franjas_for_seed(s))
        zonas_rows.extend(zonas_kpi_for_seed(s))
    present = sorted({r["seed"] for r in global_rows})

    # agregados entre seeds
    global_agg = []
    for fr in FRANJAS:
        sub = [r for r in global_rows if r["franja"] == fr["nombre"]]
        fm, fsd = _mean_sd([r["fijo_net_timeloss_s"] for r in sub])
        am, asd = _mean_sd([r["adap_net_timeloss_s"] for r in sub])
        rdm, rdsd = _mean_sd([r["rd_pct"] for r in sub])
        global_agg.append({"franja": fr["nombre"], "n_seeds": len(sub),
                           "fijo_mean_s": round(fm, 4) if fm else None,
                           "adap_mean_s": round(am, 4) if am else None,
                           "rd_pct_mean": round(rdm, 4) if rdm is not None else None,
                           "rd_pct_sd": round(rdsd, 4) if rdsd is not None else None})

    zonas_agg = []
    for z in ZONAS:
        for fr in FRANJAS:
            sub = [r for r in zonas_rows if r["zona"] == z["nombre"]
                   and r["franja"] == fr["nombre"]]

            def _m(key):
                return _mean_sd([r[key] for r in sub])

            ftl, _ = _m("fijo_timeloss_total_s")
            atl, _ = _m("adap_timeloss_total_s")
            rdm, rdsd = _m("rd_pct_timeloss")
            ftlpv, _ = _m("fijo_timeloss_per_veh_s")
            atlpv, _ = _m("adap_timeloss_per_veh_s")
            rdpvm, rdpvsd = _m("rd_pct_timeloss_per_veh")
            fwtpv, _ = _m("fijo_waiting_per_veh_s")
            awtpv, _ = _m("adap_waiting_per_veh_s")
            rdwtm, rdwtsd = _m("rd_pct_waiting_per_veh")
            fthr, _ = _m("fijo_throughput")
            athr, _ = _m("adap_throughput")
            zonas_agg.append({"zona": z["nombre"], "nivel_los": z["nivel_los"],
                              "franja": fr["nombre"], "n_seeds": len(sub),
                              "fijo_timeloss_total_mean_s": round(ftl, 2) if ftl else None,
                              "adap_timeloss_total_mean_s": round(atl, 2) if atl else None,
                              "rd_pct_timeloss_mean": round(rdm, 4) if rdm is not None else None,
                              "rd_pct_timeloss_sd": round(rdsd, 4) if rdsd is not None else None,
                              "fijo_timeloss_per_veh_mean_s": round(ftlpv, 4) if ftlpv else None,
                              "adap_timeloss_per_veh_mean_s": round(atlpv, 4) if atlpv else None,
                              "rd_pct_timeloss_per_veh_mean": round(rdpvm, 4) if rdpvm is not None else None,
                              "rd_pct_timeloss_per_veh_sd": round(rdpvsd, 4) if rdpvsd is not None else None,
                              "fijo_waiting_per_veh_mean_s": round(fwtpv, 4) if fwtpv else None,
                              "adap_waiting_per_veh_mean_s": round(awtpv, 4) if awtpv else None,
                              "rd_pct_waiting_per_veh_mean": round(rdwtm, 4) if rdwtm is not None else None,
                              "rd_pct_waiting_per_veh_sd": round(rdwtsd, 4) if rdwtsd is not None else None,
                              "fijo_throughput_mean": round(fthr, 1) if fthr else None,
                              "adap_throughput_mean": round(athr, 1) if athr else None})

    sanity = _f5_sanity(present, global_rows)

    _write_csv(EXP_DIR / "f4_franjas_global.csv", global_rows)
    _write_csv(EXP_DIR / "f4_zonas_kpi.csv", zonas_rows)
    es_canonico = present == CANONICAL_SEEDS
    result = {
        "_meta": {
            "fase": "F4 (DHU-030) — KPI por zona + franjas, reproceso de las corridas con edgeData",
            "seeds_presentes": present,
            "franjas_global": "APROXIMADO (cohorte de salida: filtra tripinfo por depart)",
            "kpi_zona": "EXACTO por franja (edgeData mide el intervalo, no la cohorte)",
            "como_se_regenera": "python simulation/scripts/reprocess_f4_zonas.py "
                                "(lee runs/f4_* + zonas_pmu_f4.json)",
        },
        "hallazgos": HALLAZGOS if es_canonico else
            {"_nota": f"lectura interpretativa solo para el run canónico de 10 seeds "
                      f"{CANONICAL_SEEDS}; presentes={present} (parcial, sin hallazgos)"},
        "franjas_global_por_seed": global_rows,
        "franjas_global_agregado": global_agg,
        "zonas_kpi_por_seed": zonas_rows,
        "zonas_kpi_agregado": zonas_agg,
        "sanity_vs_f5": sanity,
    }
    (EXP_DIR / "f4_resultado.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # ---- impresión legible (lo que reporto en el stop gate) ----
    print(f"\nF4 reproceso — seeds presentes: {present}")
    print("\n=== (1) FRANJAS GLOBALES (net_timeLoss red entera, mean por vehículo) ===")
    print(f"{'franja':16} {'fijo_s':>9} {'adap_s':>9} {'RD%':>8} {'±sd':>7} {'n':>3}")
    for r in global_agg:
        print(f"{r['franja']:16} {r['fijo_mean_s'] or 0:9.3f} {r['adap_mean_s'] or 0:9.3f} "
              f"{r['rd_pct_mean'] or 0:+7.2f}% {r['rd_pct_sd'] or 0:6.2f} {r['n_seeds']:>3}")
    print("\n=== (2) KPI POR ZONA — RD% (>0 = adaptativo mejora) ===")
    print(f"{'zona':18} {'franja':14} | {'TL tot':>7} {'±sd':>6} | {'TL/veh':>7} {'±sd':>6} | "
          f"{'wait/veh':>8} {'±sd':>6} | {'thru f→a':>11} | n")
    for r in zonas_agg:
        print(f"{r['zona']:18} {r['franja']:14} | "
              f"{r['rd_pct_timeloss_mean'] or 0:+6.1f}% {r['rd_pct_timeloss_sd'] or 0:6.2f} | "
              f"{r['rd_pct_timeloss_per_veh_mean'] or 0:+6.1f}% {r['rd_pct_timeloss_per_veh_sd'] or 0:6.2f} | "
              f"{r['rd_pct_waiting_per_veh_mean'] or 0:+7.1f}% {r['rd_pct_waiting_per_veh_sd'] or 0:6.2f} | "
              f"{r['fijo_throughput_mean'] or 0:5.0f}→{r['adap_throughput_mean'] or 0:<5.0f} | {r['n_seeds']}")
    if sanity:
        print("\n=== SANITY vs F5 (día completo global; deben coincidir) ===")
        for s in sanity:
            print(f"  seed {s['seed']}: fijo F4 {s['f4_dia_fijo']:.3f} vs F5 {s['f5_fijo']:.3f} | "
                  f"adap F4 {s['f4_dia_adap']:.3f} vs F5 {s['f5_adap']:.3f}")
    print(f"\nVersionables → {EXP_DIR.name}/{{f4_franjas_global.csv, f4_zonas_kpi.csv, f4_resultado.json}}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
