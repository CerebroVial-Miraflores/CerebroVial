# Red base SUMO — Corredor Av. José Larco (DHU-027)

Red base de simulación de los **3 cruces semaforizados consecutivos** de Av. José Larco,
Miraflores, para la validación de Max Pressure como control de **red/corredor** (no nodo
aislado). Decisión y sustento en
[`DECISIONS_HU.md` § DHU-027](../../../documentation/lean-inception/4-decisiones/DECISIONS_HU.md).

> **Camino de build PARALELO al de TTH-07.** La red de TTH-07
> (`conf/network/miraflores_4way.net.xml`) es topología genérica sintética (sin OSM). Esta
> red usa geometría **REAL** de OpenStreetMap. Ambas conviven; el genérico **no** se reemplaza.

## Cruces objetivo

| Cruce | Latitud | Longitud |
|---|---|---|
| Larco × Diez Canseco | -12.12242 | -77.02911 |
| Larco × Schell | -12.12297 | -77.02917 |
| Larco × Benavides | -12.12454 | -77.02933 |

## Reproducibilidad

- **Snapshot OSM:** descargado el **2026-05-29** desde la API principal de OpenStreetMap.
  OSM es un mapa vivo; `corredor_larco.osm` (versionado) **congela la geometría exacta** del
  experimento. No re-descargar salvo que se quiera actualizar deliberadamente el snapshot.
- **Bounding box** (recortado — Óvalo/Parque Kennedy **excluido**, ver DHU-027 y nota abajo),
  formato `minlon,minlat,maxlon,maxlat` (oeste,sur,este,norte):

  ```
  -77.0302,-12.1254,-77.0282,-12.1220
  ```

- **Generación:** `bash scripts/build_corredor_larco.sh` (desde `simulation/`). El script
  descarga el OSM (API principal de OSM, con validación anti-respuesta-de-error), y corre
  `netconvert`. SUMO usado: Eclipse SUMO 1.26.0.

  Comando `netconvert` efectivo:
  ```bash
  netconvert \
    --osm-files conf/corredor_larco/corredor_larco.osm \
    --type-files "$SUMO_HOME/data/typemap/osmNetconvert.typ.xml" \
    --output-file conf/corredor_larco/corredor_larco.net.xml \
    --proj.utm \
    --geometry.remove --remove-edges.isolated \
    --keep-edges.by-vclass passenger \
    --tls.guess-signals --tls.join false \
    --junctions.join \
    --no-turnarounds true \
    --output.street-names true --output.original-names true \
    --remove-edges.explicit 406008845#5,344159559#0,39441587
  ```
  Razón de las opciones: `--tls.guess-signals` reubica los nodos OSM `traffic_signals` sobre
  las edges entrantes (**preserva** los semáforos reales, no los inventa); `--tls.join false`
  mantiene los TLS del corredor **independientes** (la coordinación la hace el motor, no un
  programa fusionado de SUMO); `--junctions.join` limpia la avenida dividida en cruces únicos;
  `--keep-edges.by-vclass passenger` deja la red vehicular; `--output.street-names` hace la red
  autodocumentada. **No** se usa `--tls.guess` (inventaría semáforos donde OSM no los tiene).
  `--remove-edges.explicit` aplica el **recorte de saneamiento** decidido tras la revisión
  visual del gate (ver sección de recorte abajo).

## ⚠️ Ciclo de vida del `.net.xml` — LEER ANTES DE RE-EJECUTAR EL SCRIPT

`build_corredor_larco.sh` reproduce **solo la importación inicial** OSM → `.net.xml`. Una vez
que la red se edite en **netedit** (correcciones tras la validación visual — ver hallazgos
abajo), **`corredor_larco.net.xml` pasa a ser FUENTE DE VERDAD CURADA y NO regenerable desde
el script**: re-ejecutarlo lo **pisaría** y borraría las correcciones manuales. El script queda
como **referencia de procedencia**, no como regenerador idempotente.

## Estado de la red (importación OSM + recorte de saneamiento del gate)

netconvert terminó con `Success`. Proyección **UTM zona 18S** aplicada (el warning
`Cannot find proj.db` no afectó: el `.net.xml` lleva `projParameter` UTM válido).

La importación cruda trajo **5 TLS** (los 3 del corredor + 2 extra sobre Av. Benavides al
este) y geometría colgante fuera del corredor. Tras la **revisión visual del gate** se aplicó
un recorte vía `--remove-edges.explicit` y la red quedó con **exactamente 3 TLS** — los 3
cruces objetivo, identificados por nombre de calle:

| Cruce (calles importadas) | Junction id | Links controlados |
|---|---|---|
| Avenida José Larco × Avenida Ernesto Diez Canseco | `108178122` | 5 |
| Avenida José Larco × Calle Schell | `133925753` | 9 |
| Avenida José Larco × Avenida Alfredo Benavides | `cluster_108178119_263630444_2673400749_3245705958_#6more` | 15 |

### Sentido de Av. Larco — CONFIRMADO

OSM etiqueta `Avenida José Larco` como `oneway=yes` sentido **sur→norte** (hacia el
Óvalo/Parque Kennedy), 3 carriles. **Confirmado correcto** (OSM fiel a la realidad); la red
**no** se invierte.

### Recorte aplicado (`--remove-edges.explicit`)

Edges removidas: `406008845#5`, `344159559#0`, `39441587`.

- `406008845#5` y `344159559#0`: las dos calzadas de Av. Benavides **más allá** de los 2 TLS
  extra al este (~160 m de Larco). Al quedar como dead-end, netconvert descartó esos 2
  semáforos (`cluster_263630443_3245705965`, `cluster_108191737_3245705968_3245705969`) →
  **5 TLS → 3 TLS**. Se conservan ~150 m de approach de Benavides al cruce con Larco
  (`406008845#1` = 151 m, `344159559#2` = 153 m).
- `39441587`: 313 m de Av. Ernesto Diez Canseco colgando hacia el este (fuera del corredor).
  El approach de Diez Canseco al cruce queda en `511823826` (62 m).
- **Cascada:** remover `39441587` dejó **Pasaje Tello** (`406010997`) como ramal sin junction
  (desembocaba en el mismo nodo que la extensión este de Diez Canseco), y netconvert lo
  auto-podó ("Removed a road without junctions"). Es inseparable de remover `39441587`
  (conservarlo exigiría mantener los 313 m colgantes). **Calle Cristóbal Colón** (`406007420`,
  independiente) **sí** se conserva.

Resultado: 17 edges, 3 TLS. Óvalo/Parque Kennedy **fuera** (`roundabouts detectados: 0`);
el cruce más al norte es Diez Canseco, con ~214 m de approach norte de Larco. Calles en la
red: Avenida José Larco, Avenida Ernesto Diez Canseco, Calle Schell, Avenida Alfredo
Benavides, Calle Cristóbal Colón, Calle Tarata.

### ⚠️ DEUDA — Diez Canseco sin conflicto NS-EW (NO corregir aquí)

Diez Canseco importó **sin approach transversal entrante** → su semáforo (`108178122`) solo
regula Larco, **sin conflicto NS-EW**; el acoplamiento *downstream* se ejercita en 1 link
interno (Benavides→Schell). **Suficiente para validar correctitud del motor**; recuperar el
conflicto (verificar realidad / extender) queda para el **experimento de magnitud sobre la red
completa de Miraflores**. No se corrige en esta etapa.

## Fuera de alcance de esta etapa

Sin demanda (`.rou.xml`), sin `.sumocfg`, sin corrida de simulación, sin integración con el
motor. El mapeo de cada cruce a su `intersection_id` del seed del core (falta `larco_diezcanseco`)
es trabajo de etapas posteriores. Esta etapa entrega **solo la red base** para validación visual.
