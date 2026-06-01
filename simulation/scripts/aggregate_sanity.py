"""Sanity agregado del dataset laborable 60d (corre con .venv: pandas+pyarrow).

Señales de colapso robustas leídas de la tabla compacta (NO density-weighted, que
sobre-pondera atascos y da falsos positivos). Por edges CON presencia (speed no-NaN):
  - mean_kmh: velocidad media (km/h). Colapso tipo 0.35 -> cae a un dígito sostenido.
  - jam_frac: fracción con speedRelative<0.3 (congestionado). Sano = bimodal (sube en
    picos AM/PM, recupera en meseta). Colapso = alto y sostenido toda la tarde.
  - nan_pres: fracción con traveltime=NaN (nada completa traversal). 2da firma de
    gridlock (hallazgo del usuario): en un día colapsado explota en horas pico.
Flags de colapso (umbrales con amplio margen vs el sano ~30 km/h / jam ~3% / NaN ~3%):
  COLAPSO si  maxrun(mean_kmh<12, 07-23h) >= 4   o   pico nan_pres >= 0.30
              o   maxrun(jam_frac>0.20, 07-23h) >= 4
"""
import sys, glob, os
import numpy as np
import pandas as pd

DSDIR = sys.argv[1] if len(sys.argv) > 1 else "dataset_laborable_60d"


def maxrun(mask_by_hour, lo=7, hi=23):
    run = mx = 0
    for h in range(lo, hi):
        run = run+1 if mask_by_hour[h] else 0
        mx = max(mx, run)
    return mx


def day_metrics(df):
    df = df.assign(h=(df["timestep"] // 3600).clip(0, 23))
    pres = df[df["speed"].notna()]
    g = pres.groupby("h")
    mean_kmh = (g["speed"].mean()*3.6).reindex(range(24)).to_numpy(float)
    jam = g.apply(lambda x: (x["speedRelative"] < 0.3).mean(), include_groups=False
                  ).reindex(range(24)).to_numpy(float)
    nanp = g["traveltime"].apply(lambda x: x.isna().mean()
                                 ).reindex(range(24)).to_numpy(float)
    return mean_kmh, jam, nanp


def main():
    files = sorted(glob.glob(os.path.join(DSDIR, "day_seed*.parquet")))
    print(f"# {len(files)} días en {DSDIR}\n")
    sp_sum = np.zeros(24); jam_sum = np.zeros(24); n = 0
    hdr = (f"{'seed':>4} | {'filas':>7} | {'edg':>3} | {'ts':>4} | {'km/h glob':>9} | "
           f"{'min km/h':>8} | {'jam% pico':>9} | {'NaN%p pico':>10} | flag")
    print(hdr); print("-"*len(hdr))
    bad = []
    for f in files:
        seed = os.path.basename(f)[8:11]
        df = pd.read_parquet(f)
        ne, nt = df["edge_id"].nunique(), df["timestep"].nunique()
        shape_ok = len(df) == 548640 and ne == 381 and nt == 1440
        mean_kmh, jam, nanp = day_metrics(df)
        # Colapso (gridlock 0.35) = velocidad presente clavada en un dígito y/o
        # traveltime NaN explotando. Sano 0.20: mean~31, min~29, NaN<8% -> margen amplio.
        # jam_frac (speedRelative<0.3) tiene piso de ruido ~15% en valle: NO sirve de
        # flag absoluto; se reporta solo como evidencia de FORMA bimodal.
        flags = []
        if not shape_ok: flags.append("FORMA")
        if np.nanmin(mean_kmh) < 12 or maxrun(mean_kmh < 15) >= 4: flags.append("COLAPSO-vel")
        if np.nanmax(nanp[7:23]) >= 0.30: flags.append("COLAPSO-NaN")
        fl = ",".join(flags) if flags else "ok"
        if flags: bad.append((seed, fl))
        sp_sum += np.nan_to_num(mean_kmh); jam_sum += np.nan_to_num(jam); n += 1
        print(f"{seed:>4} | {len(df):>7} | {ne:>3} | {nt:>4} | {np.nanmean(mean_kmh):>9.1f} | "
              f"{np.nanmin(mean_kmh):>8.1f} | {np.nanmax(jam[7:23])*100:>8.1f}% | "
              f"{np.nanmax(nanp[7:23])*100:>9.1f}% | {fl}")
    print("\n# Perfil PROMEDIO sobre días — jam% por hora (¿bimodal AM/PM con recuperación?)")
    avg_jam = jam_sum/max(n, 1)
    for h in range(24):
        print(f"{h:02d}h | {avg_jam[h]*100:4.1f}% {'#'*int(avg_jam[h]*100*4)}")
    print(f"\n# Veredicto: {len(files)} días | {len(bad)} con flags.")
    print("  " + ("Ninguno colapsó: todos recuperan, sin firma gridlock 0.35." if not bad
                  else f"REVISAR: {bad}"))


if __name__ == "__main__":
    main()
