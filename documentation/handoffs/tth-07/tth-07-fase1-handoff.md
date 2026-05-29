# TTH-07 — Cierre de Fase 1 (scaffold `simulation/` + CT-07.1 topología)

**Rama**: `feature/tth-07` (cortada desde `feature/tth-07-fase0-docs`).
**Fecha de cierre**: 2026-05-29.
**Estado al cierre**: F1 verde, 8/8 tests pasando. Hallazgo material: SUMO
1.26 arrow writer no soporta append multi-interval ⇒ adoptado fallback
XML para outputs `edgeData`/`laneArea` (documentado y planificado en
F1.bis del plan).

---

## 1. Lo que F1 entregó

### Scaffold `simulation/` materializado

```
simulation/
├── .env.example              SUMO_HOME (wheel) + ENGINE_URL=http://localhost:8001/control/recommend
├── .gitignore                .venv, *.parquet, intermediarios _generated_*.xml
├── pyproject.toml            pin: eclipse-sumo==1.26.0, traci==1.26.0; sumolib>=1.26.0 (transitivo 1.27); pyarrow; pyyaml; requests
├── README.md                 esbozo F1 (completado en F6)
├── .venv/                    venv aislado (gitignored)
├── conf/network/
│   ├── network_params.yaml   parámetros legibles aprobados (NS 3 carriles × 300 m, EW 2 × 200 m, vmax 13.89 m/s, 2 fases NS/EW lefts permissive)
│   ├── linkstates.json       derivado por sumolib introspección — 6 sub-fases
│   ├── miraflores_4way.net.xml         red 4-vías (netconvert desde nodes/edges)
│   ├── miraflores_4way.tllogic.add.xml programa "baseline" con 6 sub-fases SUMO
│   ├── edgedata.add.xml      meandata por edge, freq=60, output XML (fallback)
│   └── lanearea.add.xml      10 detectores E2 (3+3+2+2 lanes), freq=60, output XML
├── src/cerebrovial_simulation/
│   ├── __init__.py
│   └── jam_level.py          D-009 canónico (transcripción literal de DECISIONS.md:211-225)
├── scripts/
│   └── build_network.py      genera red + linkstates derivados via sumolib
├── tests/
│   ├── conftest.py           setup SUMO_HOME + PATH para tests
│   ├── fixtures/test_smoke.rou.xml  20 vehículos, 4 direcciones, 60s
│   └── test_network_loads.py 8 tests (CT-07.8.a + corrección 4 + corrección 2)
└── data/.gitkeep
```

### Parámetros F1 aprobados (verificados en tests)

| Parámetro | Valor | Test verificador |
|-----------|-------|------------------|
| Carriles N-S/aproche | 3 | `test_a3_lane_counts` |
| Carriles E-W/aproche | 2 | `test_a3_lane_counts` |
| Largo efectivo N-S | ≈ 290 m (netconvert trim de ~10m) | `test_a5_approach_lengths` |
| Largo efectivo E-W | ≈ 186 m | `test_a5_approach_lengths` |
| vmax | 13.89 m/s | `test_a4_vmax_uniform` |
| tlLogic Option A | 2 fases NS+EW × 3 sub-fases (g/y/r) = 6 sub-fases SUMO | `test_a2_tls_program_baseline_has_6_subphases` |
| Lefts permissive | linkstates `G/g/r` derivados por sumolib introspección — `g` minúscula en lefts | `test_a8_linkstates_json_has_6_keys` |

### Linkstates derivados (corrección 4)

`conf/network/linkstates.json`:

```json
{
  "NS_g": "GGGGgrrrrGGGGgrrrr",
  "NS_y": "yyyyyrrrryyyyyrrrr",
  "NS_r": "rrrrrrrrrrrrrrrrrr",
  "EW_g": "rrrrrGGGgrrrrrGGGg",
  "EW_y": "rrrrryyyyrrrrryyyy",
  "EW_r": "rrrrrrrrrrrrrrrrrr"
}
```

18 chars = 5 links (N) + 4 links (E) + 5 links (S) + 4 links (W), ordenados
por TL link index. `g` minúscula identifica los lefts permissive (cede al
opuesto). Verificado por behavior en `test_a6_baseline_behavior` (no por
inspección de strings — el test corre la sim y comprueba que vehículos
NS y EW cruzan según sus fases).

---

## 2. Hallazgo material — fallback XML para outputs multi-interval

### Síntoma

SUMO 1.26 falla con `arrow/result.cc:28: ValueOrDie called on an error:
IOError: Appending to file not implemented` cuando intenta escribir
Parquet a un output multi-interval (`edgeData` o `laneArea` con
`freq < end`). Reproducido con:

```bash
sumo -n net.net.xml -r routes.rou.xml \
  --additional-files edgedata.add.xml,lanearea.add.xml \
  --end 120 \
  --output.format parquet
# arrow/result.cc: IOError: Appending to file not implemented.
```

Idem si los add-files declaran `file="edgedata.parquet"` directamente —
el sufijo o `--output.format` global activan el writer Parquet, que no
soporta append entre intervals.

### Causa raíz

Arrow Parquet writer cierra el archivo al primer flush — no soporta
append. SUMO 1.26 escribe un intervalo, cierra, y al siguiente intervalo
intenta reabrir-y-append ⇒ falla.

### Decisión — fallback XML documentado (plan F1.bis adoptado)

**Outputs single-write** (un solo flush al cierre de la simulación) →
**Parquet directo**:
- `--summary-output X.parquet` ✅ verificado.
- `--tripinfo-output X.parquet` ✅ verificado.

**Outputs multi-interval** (un flush por `freq`) → **XML**, pipeline
posterior convierte a Parquet con pyarrow:
- `edgedata.add.xml` declara `file="edgedata.xml"`.
- `lanearea.add.xml` declara `file="lanearea.xml"` compartido por los 10
  detectores (XML sí soporta multi-detector single-file).

**Implicancia para F2/F3/F5**:
- `coverage_check.py` (F2): parsea `edgedata.xml` y `lanearea.xml` con
  `xml.etree.ElementTree` o `sumolib.xml.parse`. La conversión a
  estructuras tabulares es rápida (tamaños esperados < 10 MB por
  corrida de 600s).
- `dataset.generate.py` (F3): mismo patrón. Convierte intermediarios XML
  a Parquet final del dataset (que sí se persiste como Parquet — esa es
  la única escritura, single-write, sin problema de append).
- `kpis.collect.py` (F5): lee `tripinfo.parquet` y `summary.parquet`
  (Parquet directo) — los outputs que F5 necesita son single-write.
  No depende del XML fallback.

**Confirma decisión opción C de F0**: el toolchain sigue siendo
`eclipse-sumo==1.26.0 + traci==1.26.0` sin libsumo. El fallback XML es
del lado SUMO, no del Python pin.

---

## 3. Tests CT-07.8.a verdes

```
$ cd simulation && .venv/bin/pytest tests/test_network_loads.py -v
tests/test_network_loads.py::test_a1_net_loads PASSED
tests/test_network_loads.py::test_a2_tls_program_baseline_has_6_subphases PASSED
tests/test_network_loads.py::test_a3_lane_counts PASSED
tests/test_network_loads.py::test_a4_vmax_uniform PASSED
tests/test_network_loads.py::test_a5_approach_lengths PASSED
tests/test_network_loads.py::test_a6_baseline_behavior PASSED
tests/test_network_loads.py::test_a7_outputs_parquet_and_xml_fallback PASSED
tests/test_network_loads.py::test_a8_linkstates_json_has_6_keys PASSED
============================== 8 passed in 2.85s ===============================
```

`test_a6_baseline_behavior` ejecuta 120s sumo + tripinfo y verifica que
vehículos NS y EW completan ruta — confirmación behavioral de que los
linkstates derivados son correctos (corrección 4 satisfecha).

`test_a7_outputs_parquet_and_xml_fallback` confirma que: (i)
summary/tripinfo escriben Parquet legible, (ii) edgeData/laneArea
escriben XML con los 10 detectores E2 presentes (intervalos cerrados
correctamente).

---

## 4. Restricciones honradas

- **Cero código en `core_management_api/`**. Regla anti-deps respetada.
- **`pin eclipse-sumo==1.26.0 + traci==1.26.0` exacto** en
  `pyproject.toml`. `sumolib==1.27.0` declarado como transitivo
  (`>=1.26.0`).
- **No push, no PR**. Commit local en `feature/tth-07`.
- **`scratch/sumo_s0/` removido** (migrado a `simulation/.venv/`).

---

## 5. Próximo paso — F2

F2 construye los 4 patrones de demanda (AM/PM peak, offpeak, weekend) y
verifica cobertura jam level vía `coverage_check.py`, que lee
`edgedata.xml` (velocidad por edge) + `lanearea.xml` (jamLengthInMeters)
y mapea a D-009. Calibra flujos iterativamente hasta cumplir CT-07.2.
