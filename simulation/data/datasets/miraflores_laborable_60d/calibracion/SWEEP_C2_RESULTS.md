# C2 — Re-calibración del scale sobre la 24h CONTINUA (post-carryover). RECOMENDADO: 0.20

**Fecha:** 2026-06-01 · scratch, NO conf/, NO commits. Semilla=42 (igual que la 24h fallida).
**Método:** barrido {0.25,0.20,0.15,0.10} corriendo la 24h COMPLETA continua cada uno
(headless, control fijo, sin TraCI, edgeData freq=60). Criterio = ¿la red DRENA?
(recupera velocidad entre fases / presencia y teleports no crecen monótonos), medido
sobre la corrida continua — NO sobre ventanas aisladas (eso fue el error de C1).

## Veredicto: HAY ventana usable, y es nítida. Recomendado = scale 0.20.
Entre 0.20 (drena limpio) y 0.25 (colapsa) hay un CLIFF agudo. 0.20 da congestión real
bimodal (picos AM y PM caen a ~9 km/h) que RECUPERA a 20+ entre picos y de noche.

## Totales 24h + veredicto de drenaje
| scale | inserted | waiting | jam | yield | wrongLane | col/em | km/h glob | h sub-8 (max racha) | drena? |
|------:|-----:|----:|----:|---:|---:|:--:|---:|:--:|:--:|
| 0.25 | 7630 | 0 | 599 | 55 | 73 | 0/0 | 18.4 | **15** | **NO (colapsa)** |
| **0.20** | **6100** | **0** | **25** | **0** | **16** | **0/0** | **21.1** | **1** | **SÍ ✓ (recomendado)** |
| 0.15 | 4576 | 0 | 4 | 0 | 2 | 0/0 | 21.6 | 0 | sí (señal débil) |
| 0.10 | 3051 | 0 | 0 | 0 | 0 | 0/0 | 22.2 | 0 | sí (trivial, plano) |
0 colisiones / 0 emergency en todos.

## Perfil temporal de velocidad por hora (ground-truth edgeData, ponderado sampledSeconds, km/h)
| h | 0.25 | **0.20** | 0.15 | 0.10 |   | h | 0.25 | **0.20** | 0.15 | 0.10 |
|--:|--:|--:|--:|--:|--|--:|--:|--:|--:|--:|
|00|21.3|20.6|21.2|23.5| |12| 4.4|20.1|21.0|22.1|
|01|21.3|21.3|22.9|19.7| |13| 4.2|19.1|20.6|22.8|
|02|19.8|23.6|21.1|22.4| |14| 3.8|20.5|20.6|20.9|
|03|23.2|20.0|23.0|20.4| |15| 4.2|20.8|20.5|21.1|
|04|21.4|21.2|20.4|21.5| |16| 4.3|20.8|20.1|20.3|
|05|23.3|22.0|22.1|21.4| |17| 4.2|20.5|21.0|21.4|
|06|20.4|21.2|21.9|21.8| |18| 4.2|19.9|**15.5**|20.7|
|07|11.0|19.8|20.5|20.8| |19| 3.3|**15.6**|19.2|21.8|
|08| 7.1|**15.4**|19.4|21.1| |20| 1.7|**9.0**|21.4|21.1|
|09| 3.6|**8.8**|21.3|21.3| |21| 2.5|21.4|21.9|22.6|
|10| 2.9|7.4|21.2|21.5| |22| 2.4|20.8|21.2|21.1|
|11| 3.7|16.6|21.0|21.0| |23| 8.2|22.6|21.8|21.3|

**0.20 = forma clara:** valle 20-23 → AM dip 15.4→8.8 (07-09h) → recupera 16-21 (11-17h) →
PM dip 15.6→9.0 (19-20h) → recupera 21-22 (21-23h). Única hora <8 es 10h (7.4), transitorio
post-AM. **0.25 = colapsa** (07h 11 → 3-4 km/h clavado 09-22h, 15h sub-8). 0.15/0.10 drenan
pero los picos casi no se marcan (señal débil/nula).

## Presencia sampledSeconds/h (miles) — forma vs monótona
0.20 BIMODAL: pico 09h=114k y 19h=90k, baja a 40k entre picos y a 10k de noche → drena.
0.25 MONÓTONA-ish: trepa a 525k@19h, recién baja al final → acumula.

## Teleports/hora (indicador limpio de drenaje)
- **0.20:** sólo en picos — 08-10h:4/13/9, 19-20h:5/8; **CERO entre picos (11-18h) y tras 21h.**
  La red se vacía entre fases. ~41 total.
- 0.25: cada hora todo el día (51@09…110@19…136@20), nunca se aquieta → no drena.

## Recomendación honesta
- **scale 0.20:** el más alto que DRENA. Congestión real bimodal (dips a ~9 km/h en AM y PM,
  contraste fuerte vs ~21 off-peak) con recuperación completa. jam 25, waiting 0. Buena señal
  para dataset. **Pick.**
- 0.25 está sobre el cliff (colapsa, milder que 0.35 pero igual no drena).
- 0.15 fallback conservador si 0.20 se siente al filo, pero pierde señal (AM casi no dipea).
- 0.10 trivial (plano).
- NO hay problema de "no existe ventana": la ventana existe y es 0.20. El cliff 0.20↔0.25
  es agudo (bimodal), consistente con el hallazgo previo.
