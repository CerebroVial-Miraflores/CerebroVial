# Corredor Larco — Etapa 1: cierre (rama + DHU de alcance + red base SUMO)

**Rama:** `feature/corredor-larco-max-pressure` (desde `master`). Sin push, sin PR, sin merge.
**Alcance:** documentación de alcance + red base de simulación. **No toca el motor**
(`core_management_api/` intacto). Solo `documentation/` y `simulation/`.

## Qué se entregó

1. **DHU-027** (`documentation/lean-inception/4-decisiones/DECISIONS_HU.md`): la validación
   cuantitativa del motor pasa de **intersección aislada a RED/corredor coordinado**. Motivo
   estructural: la ventaja de Max Pressure viene del término *downstream* `x_down`, que en nodo
   aislado es ≈0 y vuelve a MP indistinguible de un control fijo bien sintonizado. **IE05 se
   reformula a métrica de RED** (demora promedio de red), conservando el umbral **RD% ≥ 15%**.
   Nota-puntero agregada en `documentation/docs/CONTROL.md` (def. previa de IE05). Frontera
   explícita: geometría **REAL** (OSM); tiempos fijos y matriz OD **SUPUESTOS** (Webster por
   nodo + demanda sintética) con plan de reemplazo cuando la Subgerencia de Movilidad provea
   tiempos 2014 y conteos.

2. **Red base SUMO** (`simulation/conf/corredor_larco/`): `corredor_larco.osm` (snapshot OSM
   **2026-05-29**, API principal de OSM), `corredor_larco.net.xml` (UTM zona 18S),
   `README.md`. Generador reproducible: `simulation/scripts/build_corredor_larco.sh`. SUMO 1.26.0.

## Red resultante (post-recorte del gate)

- **Bounding box** (Óvalo/Parque Kennedy excluido): `-77.0302,-12.1254,-77.0282,-12.1220`
  (minlon,minlat,maxlon,maxlat).
- Importación cruda: **5 TLS** + geometría colgante. Tras revisión visual del gate se aplicó
  `--remove-edges.explicit 406008845#5,344159559#0,39441587` →
  **EXACTAMENTE 3 TLS**, 17 edges.

| Cruce | Junction id | links | Conflicto |
|---|---|---|---|
| Larco × Diez Canseco | `108178122` | 5 | ⚠️ sin transversal entrante (ver deuda) |
| Larco × Schell | `133925753` | 9 | ✅ Larco × Schell pasante |
| Larco × Benavides | `cluster_108178119_263630444_2673400749_3245705958_#6more` | 15 | ✅ fuerte (≈4 ramas) |

- **Sentido de Av. Larco: CONFIRMADO sur→norte** (OSM correcto; no se invierte).
- **Recorte:** los 2 TLS extra sobre Av. Benavides al este se eliminaron retirando las calzadas
  más allá de ellos (`406008845#5`, `344159559#0`); se conservan ~150 m de approach de
  Benavides. Se retiraron 313 m colgantes de Diez Canseco al este (`39441587`). **Cascada:**
  remover `39441587` dejó Pasaje Tello (`406010997`) como ramal sin junction → netconvert lo
  auto-podó (inseparable de remover `39441587`; conservarlo exigiría mantener los 313 m
  colgantes). Calle Cristóbal Colón (independiente) se conserva. Sin rotondas
  (`roundabouts detectados: 0`).

## Deudas y follow-ups

- **⚠️ Diez Canseco sin conflicto NS-EW.** Diez Canseco importó **sin approach transversal
  entrante** → su semáforo solo regula Larco, **sin conflicto NS-EW**; el acoplamiento
  *downstream* se ejercita en 1 link interno (Benavides→Schell). **Suficiente para validar
  correctitud del motor**; recuperar el conflicto (verificar realidad / extender) queda para el
  **experimento de magnitud sobre la red completa de Miraflores**.
- **Seed del core:** el mapeo de cada cruce a su `intersection_id` falta `larco_diezcanseco`
  (existen `larco_schell`, `larco_benavides`). Necesario cuando la etapa de motor lo requiera.
- **Fuera de alcance de esta etapa:** demanda (`.rou.xml`), `.sumocfg`, corrida de simulación,
  baseline Webster por nodo del corredor, integración con el motor.

## Ciclo de vida del `.net.xml`

`build_corredor_larco.sh` reproduce **solo la importación inicial** OSM → `.net.xml`. Si la red
se edita en netedit, el `.net.xml` versionado pasa a ser **fuente de verdad curada y NO
regenerable** desde el script (re-ejecutarlo la pisaría). El script queda como referencia de
procedencia.

## Reproducibilidad

```bash
cd simulation
bash scripts/build_corredor_larco.sh              # descarga OSM (API principal) + netconvert + recorte
bash scripts/build_corredor_larco.sh --no-download # reusa el .osm congelado
```

## Commits de la rama

- `docs(corredor-larco): DHU-027 — alcance de validación pasa de nodo único a red/corredor; IE05 a métrica de red`
- `feat(corredor-larco): red base SUMO del corredor (OSM crudo + net.xml + comandos reproducibles)`
- `feat(corredor-larco): recorte a 3 TLS del corredor (--remove-edges.explicit) + deuda Diez Canseco`
