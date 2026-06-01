# C1 — Barrido de calibración del multiplicador global `--scale` (DESCARTABLE)

**Fecha:** 2026-06-01 · scratch, NO commitear, NO es nivel definitivo (lo decide el usuario).

## Método
Multiplicador global = SUMO `--scale` sobre el `.rou.xml` ya ruteado del gate.
`--scale` descarta/duplica vehículos de forma uniforme → preserva la FORMA temporal
y la ponderación espacial (arteria 5:1), sólo baja el NIVEL. Aísla *nivel* sobre un
set de rutas fijo (calibración más limpia que regenerar). scale=1.0 = placeholder gate.
Mismo net, mismo control fijo (Webster del net.xml), semilla default, ventanas acotadas
(headless, NO 24h). Reproduce el gate exacto a scale 1.0 (validación del harness).

**Denominador de "% sin insertar":** `inserted + waiting` = demanda atendida a ese scale
(los descartados por --scale NO son "trips sin insertar"). `loaded` queda ~6700 a todo
scale (= todos los trips que parten en la ventana); el scale recorta ANTES de insertar.

## Barrido AM peak (06-09h)
| scale | atendida(ins+wait) | inserted | waiting | wait% | tel_tot | jam | yield | wLane | km/h | lectura |
|------:|-----:|-----:|----:|-----:|----:|----:|----:|----:|----:|---|
| 1.00 | 6700 | 4509 | 2191 | 32.7% | 4810 | 3313 | 1045 | 452 | 6.0 | placeholder — GRIDLOCK |
| 0.60 | 4020 | 3497 | 523 | 13.0% | 2395 | 1698 | 479 | 218 | 12.2 | saturado |
| 0.50 | 3350 | 3135 | 215 | 6.4% | 1427 | 1012 | 269 | 146 | 15.3 | saturado |
| 0.45 | 3015 | 2882 | 133 | 4.4% | 1015 | 709 | 195 | 111 | 16.6 | sobre el borde |
| **0.40** | 2680 | 2626 | 54 | **2.0%** | 509 | **339** | 106 | 64 | **18.3** | **BORDE DEL CLIFF** |
| 0.35 | 2345 | 2331 | 14 | 0.6% | 199 | 127 | 53 | 19 | 19.2 | funcional c/margen |
| 0.25 | 1675 | 1675 | 0 | 0.0% | 49 | 34 | 6 | 9 | 20.5 | congestión casi nula |
| 0.15 | 1005 | 1005 | 0 | 0.0% | 2 | 1 | 0 | 1 | 21.2 | ≈ valle, sin pico |

0 colisiones / 0 emergency en TODOS los scales (red estructuralmente sana, igual que el gate).

## Criterios "funcional" (pico AM)
1. waiting < ~2% de la demanda atendida → 0.40 (2.0%, justo en la línea), claro ≤0.35.
2. jam un orden de magnitud bajo el placeholder (3313 → centenas bajas) → 0.40=339 (9.8×↓), 0.35=127 (26×↓).
3. velocidad entre gridlock (6) y valle (21.9) → 0.40=18.3, 0.35=19.2 (congestión presente, sin paralizar).

## Borde del cliff
**Highest functional = scale 0.40.** 0.45 ya falla (waiting 4.4%>2%, jam 709 sólo 4.7×↓).
**Matiz honesto:** NO es un escalón discontinuo único — es una RAMPA EMPINADA en
0.40→0.55 (cada +0.05 ≈ duplica jam y waiting). El colapso total (6 km/h, 33% sin
insertar) es el placeholder 1.0; el régimen funcional limpio empieza ≤0.40. 0.40 cae
JUSTO en el umbral del 2% (sensibilidad de filo de cuchillo ahí).

## Verificación VALLE (00-06h)
| scale | inserted | waiting | teleports | km/h |
|------:|-----:|----:|----:|----:|
| 1.00 | 1400 | 0 | 0 | 21.9 (= gate) |
| 0.40 | 560 | 0 | 0 | 22.7 |
| 0.35 | 490 | 0 | 0 | 22.6 |
Valle sano a ambos: 0 waiting, 0 teleports, free-flow. Escala uniforme (1400→560/490 = ×0.40/×0.35).
Más liviano (esperado: el valle ya fluía; bajarlo sólo lo aclara), no roto ni sobre-vacío.

## Recomendación
- **Borde literal del cliff:** 0.40 (el scale más alto que cumple los 3 criterios).
- **Pick operativo para el dataset:** **0.35** — un escalón por debajo del filo, margen
  seguro (waiting 0.6%, jam 127 = 26×↓, 19.2 km/h) MANTENIENDO congestión real y deseable.
  0.40 sirve si se quiere el pico más cargado posible, asumiendo que vive en el umbral.
- Decisión final del nivel: del usuario, antes de la corrida 24h.
