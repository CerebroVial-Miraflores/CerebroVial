# STGNN Corredor Larco — Fase 1: constructor de grafo dirigido (D-011)

**Rama:** `feature/stgnn-corredor-larco` (contiene D-011). Sin push, sin PR, sin merge.
**Alcance:** investigación. **No toca producción** — `core_management_api/`,
`graph_edges/scripts/seed.py` y el predictor D-006/D-010 quedan intactos. Solo se
agrega un módulo del track en `ia_prediction_service/src/data/` + su test + el
artefacto de mapeo. No se instaló `tsl`, `pytorch-lightning` ni `torch_geometric`
(el constructor es torch puro; el formato PyG `[2, N]` no requiere la librería).

## Qué se entregó

1. **`ia_prediction_service/src/data/corridor_graph_builder.py`** — modelo
   *edge-as-node*. Parsea `corredor_larco.net.xml` y `corredor_larco.det.add.xml`
   (IDs leídos verbatim del XML), filtra los 6 edges instrumentados y construye el
   grafo dirigido S→N (arista A→B sii `net[A].to == net[B].from`). Devuelve
   `edge_index` (`torch.long [2,5]`), `edge_weight` (`torch.float [5]`, default
   `inverse`) y el dict de mapeo. 4 esquemas de peso parametrizables (`uniform`,
   `length`, `inverse`, `inv_sqrt`), normalizados a media=1.0.
2. **`ia_prediction_service/src/data/artifacts/corridor_graph_mapping.json`** —
   puente nodo↔edge↔detectores que la Fase 2 usa para alinear cada serie de
   detector a su nodo.
3. **`ia_prediction_service/tests/test_corridor_graph_builder.py`** — 6 tests
   (6 nodos / 5 aristas / sentido / fuentes-sumidero / cobertura de detectores /
   4 esquemas). Verde con `.venv/bin/python` (torch 2.9.1).

## Topología resultante (6 nodos, 5 aristas)

```
edge_index = [[0, 1, 2, 3, 4],
              [2, 2, 3, 5, 5]]
```
Fuentes (sin predecesor): nodos 0, 1, 4. Sumidero (sin sucesor): nodo 5.

## Limitaciones de cobertura de datos (a citar en Fase 5)

Son **limitaciones de los datos disponibles**, no decisiones de diseño:

- **(a) Salida de Diez Canseco ciega.** Los edges aguas abajo del nodo 5
  (`279893875#5`, `511823826`) no tienen detector. El nodo 5 (`279893875#4`) es
  un sumidero del grafo: no hay propagación observable aguas abajo de Schell→Diez
  Canseco. El spillback que cruce esa salida no se mide.
- **(b) Segundo inflow transversal de Benavides sub-observado.** El inflow
  `406007422#0` de Benavides no tiene detector, por lo que el inflow transversal
  hacia el troncal está parcialmente sub-observado.
- **(c) Cobertura troncal completa, lateral parcial.** Se instrumentan 6 de los 17
  edges del corredor. El **acoplamiento inter-TL del eje troncal S→N está
  completamente cubierto** (las 5 aristas del grafo); lo no cubierto son ramas
  laterales/salidas, no el eje principal que la STGNN modela.
