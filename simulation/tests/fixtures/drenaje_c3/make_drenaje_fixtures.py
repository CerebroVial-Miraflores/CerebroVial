"""Recorte de fixtures C3 para el test de aceptación del evaluador de drenaje (D-014, B3.2.a).

Lee los edgeData crudos de las corridas C3 (scale 1.1 = drena, scale 1.3 = colapsa), que
viven en ``scratch/`` (gitignored, ~434/445 MB c/u), y escribe fixtures VERSIONADOS recortados:

- **Recorte temporal a las franjas de pico CON MARGEN de ±30 min.** Picos D-014: AM 07-09h
  ``[25200,32400)``, PM 18-20h ``[64800,72000)``. El fixture conserva ``[23400,34200)`` (AM) y
  ``[63000,73800)`` (PM). El margen deja la **transición pico→no-pico** en el fixture, para que el
  test verifique que el evaluador acota la búsqueda del dip a la ventana de pico correcta.
- **Strip a los 3 atributos que el dip consume: ``id``, ``sampledSeconds``, ``speed``.** El
  evaluador solo lee esos para la media de red ponderada por ``sampledSeconds``; los otros ~14
  atributos del ``<edge>`` no se leen, así que versionarlos es peso sin cobertura. **Se preserva la
  estructura XML exacta** (mismos tags ``<interval>``/``<edge>``, mismo anidamiento): solo se quitan
  atributos no consumidos — NO se aplana a CSV ni se renombra nada, para que el parser del evaluador
  (iterparse) corra sobre el fixture idéntico a como correría sobre el crudo.
- **Descarte de edges con ``sampledSeconds=0``** (peso nulo → no alteran la media ponderada).

El ``stats.xml`` NO se recorta (es chico, ~2.3K): se copia entero por separado.

**Verificación de equivalencia (parte del entregable):** tras escribir, recomputa la media de red
ponderada por ``sampledSeconds`` de un intervalo de pico desde el crudo Y desde el stripeado, y
exige que den idénticas. Si difieren (incluso por redondeo inesperado), aborta: el strip habría
cambiado la semántica y el fixture no sería metric-fiel al crudo.

Solo corre donde existan las corridas C3 crudas (scratch/, este disco). El fixture resultante es el
artefacto versionado; este script documenta su procedencia y lo hace reproducible.

Uso:  python simulation/tests/fixtures/drenaje_c3/make_drenaje_fixtures.py
"""
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]  # fixtures/drenaje_c3 -> tests -> simulation -> repo-root
_SCRATCH = _REPO / "scratch/b2_miraflores"

# Picos D-014 (begin en segundos) + margen de 30 min (1800 s) a cada lado.
_MARGIN = 1800
_AM = (25200 - _MARGIN, 32400 + _MARGIN)  # [23400, 34200)
_PM = (64800 - _MARGIN, 72000 + _MARGIN)  # [63000, 73800)

# Corridas C3 a promover: (tag, veredicto esperado).
_RUNS = [("s11", "drena"), ("s13", "colapsa")]

# Intervalo de pico AM usado para la verificación de equivalencia (07:30h).
_VERIFY_BEGIN = 27000


def _in_windows(begin: int) -> bool:
    return (_AM[0] <= begin < _AM[1]) or (_PM[0] <= begin < _PM[1])


def crop_edgedata(src: Path, dst: Path) -> tuple[int, int]:
    """Recorta ``src`` (edgeData crudo) a las ventanas de pico+margen, stripeando cada ``<edge>`` a
    ``id``/``sampledSeconds``/``speed`` y descartando los de ``sampledSeconds=0``. Streaming (el
    crudo es voluminoso). Devuelve (intervalos, edges)."""
    n_int = n_edge = 0
    keep = False
    with open(dst, "w") as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n<meandata>\n')
        for ev, el in ET.iterparse(str(src), events=("start", "end")):
            if ev == "start" and el.tag == "interval":
                keep = _in_windows(int(float(el.get("begin"))))
                if keep:
                    out.write(
                        f'    <interval begin="{el.get("begin")}" '
                        f'end="{el.get("end")}" id="{el.get("id")}">\n'
                    )
                    n_int += 1
            elif ev == "end" and el.tag == "edge":
                if keep and float(el.get("sampledSeconds", "0") or 0) > 0.0:
                    # Strip: solo los atributos que el evaluador consume, estructura intacta.
                    out.write(
                        f'        <edge id="{el.get("id")}" '
                        f'sampledSeconds="{el.get("sampledSeconds")}" '
                        f'speed="{el.get("speed")}"/>\n'
                    )
                    n_edge += 1
                el.clear()
            elif ev == "end" and el.tag == "interval":
                if keep:
                    out.write("    </interval>\n")
                el.clear()
        out.write("</meandata>\n")
    return n_int, n_edge


def mean_kmh_for_interval(path: Path, target_begin: int) -> float | None:
    """Media de velocidad de red (km/h) ponderada por ``sampledSeconds`` del intervalo cuyo
    ``begin == target_begin``. Mismo cómputo que hará el evaluador. ``None`` si no está."""
    in_target = False
    num = den = 0.0
    for ev, el in ET.iterparse(str(path), events=("start", "end")):
        if ev == "start" and el.tag == "interval":
            in_target = int(float(el.get("begin"))) == target_begin
        elif ev == "end" and el.tag == "edge":
            if in_target:
                ss = float(el.get("sampledSeconds", "0") or 0)
                sp = el.get("speed")
                if ss > 0.0 and sp is not None:
                    num += ss * float(sp)
                    den += ss
            el.clear()
        elif ev == "end" and el.tag == "interval":
            done = in_target
            el.clear()
            if done:
                break  # ya procesamos el intervalo objetivo; no seguir leyendo el crudo entero
    return (3.6 * num / den) if den > 0 else None


def main() -> None:
    for tag, verdict in _RUNS:
        run = _SCRATCH / f"sweep24_{tag}"
        src_edge = run / f"edgedata_{tag}.xml"
        src_stats = run / f"stats_{tag}.xml"
        if not src_edge.exists() or not src_stats.exists():
            raise SystemExit(
                f"FALTA la corrida C3 cruda {run} (edgedata/stats). Este script solo corre donde "
                f"existan las corridas C3 en scratch/ (este disco)."
            )
        dst_edge = _HERE / f"edgedata_{tag}.xml"
        dst_stats = _HERE / f"stats_{tag}.xml"

        shutil.copyfile(src_stats, dst_stats)  # stats entero (chico)
        n_int, n_edge = crop_edgedata(src_edge, dst_edge)

        src_mb = src_edge.stat().st_size / 1e6
        dst_mb = dst_edge.stat().st_size / 1e6
        print(
            f"[{tag} · {verdict}] edgedata: {src_mb:.0f}MB -> {dst_mb:.2f}MB "
            f"({n_int} intervalos, {n_edge} edges con sampledSeconds>0)  |  stats: copiado entero"
        )

        # Verificación de equivalencia: mean_kmh(crudo) == mean_kmh(stripeado) en un pico.
        raw = mean_kmh_for_interval(src_edge, _VERIFY_BEGIN)
        cut = mean_kmh_for_interval(dst_edge, _VERIFY_BEGIN)
        ok = raw is not None and cut is not None and raw == cut
        print(
            f"   equivalencia mean_kmh @begin={_VERIFY_BEGIN}: crudo={raw!r}  stripeado={cut!r}  "
            f"{'IDÉNTICO ✓' if ok else 'DIFIEREN ✗'}"
        )
        if not ok:
            raise SystemExit(
                f"[{tag}] el strip cambió mean_kmh (crudo={raw} vs stripeado={cut}): el fixture NO "
                f"es metric-fiel al crudo. Abortar (no se versiona un fixture que rompe la cadena C3)."
            )

    print("\nVentanas conservadas (begin s): "
          f"AM [{_AM[0]},{_AM[1]})  PM [{_PM[0]},{_PM[1]})  (picos D-014 ±30min margen)")


if __name__ == "__main__":
    main()
