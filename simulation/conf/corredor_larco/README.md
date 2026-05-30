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
    --output.street-names true --output.original-names true
  ```
  Razón de las opciones: `--tls.guess-signals` reubica los nodos OSM `traffic_signals` sobre
  las edges entrantes (**preserva** los semáforos reales, no los inventa); `--tls.join false`
  mantiene los TLS del corredor **independientes** (la coordinación la hace el motor, no un
  programa fusionado de SUMO); `--junctions.join` limpia la avenida dividida en cruces únicos;
  `--keep-edges.by-vclass passenger` deja la red vehicular; `--output.street-names` hace la red
  autodocumentada. **No** se usa `--tls.guess` (inventaría semáforos donde OSM no los tiene).

## ⚠️ Ciclo de vida del `.net.xml` — LEER ANTES DE RE-EJECUTAR EL SCRIPT

`build_corredor_larco.sh` reproduce **solo la importación inicial** OSM → `.net.xml`. Una vez
que la red se edite en **netedit** (correcciones tras la validación visual — ver hallazgos
abajo), **`corredor_larco.net.xml` pasa a ser FUENTE DE VERDAD CURADA y NO regenerable desde
el script**: re-ejecutarlo lo **pisaría** y borraría las correcciones manuales. El script queda
como **referencia de procedencia**, no como regenerador idempotente.

## Qué importó OSM (estado al snapshot 2026-05-29 — pendiente de validación visual)

netconvert terminó con `Success`. Proyección **UTM zona 18S** aplicada (el warning
`Cannot find proj.db` no afectó: el `.net.xml` lleva `projParameter` UTM válido).

**Los 3 cruces objetivo están presentes y correctamente identificados por nombre de calle:**

| Cruce (calles importadas) | Junction id | Links controlados |
|---|---|---|
| Avenida José Larco × Avenida Ernesto Diez Canseco | `108178122` | 5 |
| Avenida José Larco × Calle Schell | `133925753` | 9 |
| Avenida José Larco × Avenida Alfredo Benavides | `cluster_108178119_263630444_2673400749_3245705958_#6more` | 15 |

**Hallazgos para revisión visual en netedit (NO corregidos — ver disciplina de DHU-027):**

1. **Sentido de Av. Larco — CONFLICTO a confirmar.** OSM etiqueta `Avenida José Larco` como
   `oneway=yes` en sentido **sur→norte** (hacia el Óvalo/Parque Kennedy), 3 carriles, y
   netconvert lo importó fielmente así. Esto **contradice** el supuesto de trabajo de que Larco
   es sentido único **norte→sur**. No se invirtió nada: hay que validar contra la Larco real
   (puede ser que OSM esté mal, o que el supuesto N→S lo esté). Es el punto #1 a resolver.
2. **2 TLS extra sobre Av. Benavides (no sobre Larco).** Además de los 3 cruces objetivo,
   quedaron 2 semáforos cuyas únicas calles incidentes son `Avenida Alfredo Benavides`
   (`cluster_263630443_3245705965` y `cluster_108191737_3245705968_3245705969`, ~160 m al
   **este** de la intersección Larco × Benavides). Aparecen porque la red se extiende a lo
   largo de Benavides más allá del bbox (OSM conserva ways completas). Decidir en netedit si se
   recortan (corredor estricto sobre Larco) o se conservan como contexto de approach de Benavides.
3. **3 nodos-semáforo descartados** por netconvert ("does not control any links"):
   `108183458`, `108191738`, `263630442`. Son los nodos de la **calzada opuesta** de avenidas
   divididas que `--junctions.join` fusionó en los clusters (sus IDs son adyacentes a los
   clusters supervivientes). El cruce **sobrevive** como cluster; no se perdió ninguna
   intersección real. Benigno; confirmar visualmente.
4. **Extensión de la red mayor al bbox.** `convBoundary` ≈ 566 m × 720 m (vs ~376 m N-S del
   bbox), por ways que cruzan el borde. El Óvalo/Parque Kennedy quedó **fuera** (sin rotonda:
   `roundabouts detectados: 0`); el cruce más al norte es Diez Canseco. Hay ~220 m de approach
   norte de Larco más allá de Diez Canseco, sin semáforos — aceptable.

Calles importadas en la red: Avenida José Larco, Avenida Ernesto Diez Canseco, Calle Schell,
Avenida Alfredo Benavides, Calle Cristóbal Colón, Calle Tarata, Pasaje Tello.

## Fuera de alcance de esta etapa

Sin demanda (`.rou.xml`), sin `.sumocfg`, sin corrida de simulación, sin integración con el
motor. El mapeo de cada cruce a su `intersection_id` del seed del core (falta `larco_diezcanseco`)
es trabajo de etapas posteriores. Esta etapa entrega **solo la red base** para validación visual.
