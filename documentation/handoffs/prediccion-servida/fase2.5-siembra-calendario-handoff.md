# Fase 2.5 — Siembra del calendario de congestión (histórico HU-23)

**Rama:** `feature/prediccion-servida` · **Cierre:** 2026-06-04 · **Sin push/PR/merge.**

## Qué entregó

Parametriza la fecha del sembrador de congestión y agrega un wrapper de loop para sembrar
`waze_jams` con un calendario de fechas reales (1 semana), habilitando el histórico observado del
mapa (HU-23). Antes, el sembrador fechaba **todo** con la constante hardcodeada
`DAY_EPOCH = datetime(2025, 1, 6)`, de modo que un segundo día colisionaba en la PK
`(event_uuid, snapshot_timestamp)` y era no-op.

- **`core_management_api/src/congestion/infrastructure/sumo_replay_adapter.py`** — parámetro
  keyword-only `day_epoch: datetime = DAY_EPOCH` en `__init__`, usado en `_row_at` en vez de la
  constante global. La constante se conserva como default. Al variar `day_epoch` cambian tanto
  `snapshot_timestamp` **como** `event_uuid` (uuid5 sobre el `ts`) → ambos componentes de la PK
  difieren entre fechas → sin colisión. Sin `day_epoch` explícito el comportamiento es byte-idéntico.
- **`scripts/replay_congestion.py`** — flag `--date YYYY-MM-DD`; solo pasa `day_epoch` al adapter si
  se provee (sin `--date`, idéntico a hoy).
- **`scripts/seed_congestion_calendar.py`** (nuevo) — wrapper de loop sobre el sembrador existente
  (subproceso por día, NO lo extiende). Dict `CALENDAR` explícito de 8 días, gate `--execute`
  (sin él = dry-run), flags `--only`/`--limit`, log por día + `master.log`. `logs/` en `.gitignore`.

### Mapeo fecha→seed (8 días, 1 jun → 8 jun 2026)
Secuencial desde seed042 por orden de calendario (1–7 jun → 042–048), con **seed051 reservado para
el 8-jun** (día vivo + día de predicción de la fase siguiente; seed051 ∈ TEST/no-visto). 8 fechas,
8 seeds distintos `{042..048, 051}`, sin repetición, todos en disco.

| Fecha | Seed | | Fecha | Seed |
|---|---|---|---|---|
| 2026-06-01 | seed042 | | 2026-06-05 | seed046 |
| 2026-06-02 | seed043 | | 2026-06-06 | seed047 (sáb, placeholder laborable) |
| 2026-06-03 | seed044 | | 2026-06-07 | seed048 (dom, placeholder laborable) |
| 2026-06-04 | seed045 | | 2026-06-08 | **seed051** (lun, día vivo) |

### Stage-gate (un día medido en la BD local)
2026-06-01 → seed042: **231 s**, **2,390,400 filas**, **+1648 MB** (hypertable 1647→3295 MB).
- **PK desambiguada:** 2025-01-06 y 2026-06-01 conviven (count 2.39M→4.78M, no fue no-op).
- **Geom:** 0 nulls en el día nuevo (poblada por el UPDATE-join del `preseed`).
- **Idempotencia:** re-corrida del mismo día → `geom_updated=0`, count sin cambios → no-op limpio.
- **Extrapolación 8 días:** ~31 min (worst case, geom-join por día) · ~19M filas · **~13 GB**.
- Verificación: ruff limpio; **159 tests** del core pasan; sin `--date` comportamiento idéntico.

## Abierto / diferido (no se pierde)

1. **Alcance reducido a 1 semana (decisión explícita).** El histórico de la demo es **8 días**, no
   un mes. El plan original eran **39 días** (1 may → 8 jun); se recortó por el costo de disco
   (~64 GB para los 39, ~35 GB de ellos índices). El **calendario completo de un mes** (mapeo de 39
   fechas) + una micro-auditoría de resolución/índices quedan como **retrabajo posterior** si alguna
   vez se quiere el mes entero. El wrapper escala sin cambios estructurales: solo crece el dict
   `CALENDAR`.
2. **Día huérfano 2025-01-06 (decisión diferida a la Fase 4 de prediccion-servida).** Es el seed051
   fechado en enero 2025, residuo de la siembra original de TTH-12. No es parte del calendario de la
   demo y queda medio año antes de la semana de junio. Si el read-path del histórico
   (`series_for_day`) lista días por fecha, podría aparecer suelto en el slider. **Decisión a tomar
   cuando se sepa cómo el frontend pide los días:** dejarlo (probablemente el front pide por rango
   junio y nunca lo muestra) o limpiarlo antes de la demo para una BD coherente. No se resuelve en
   esta fase; queda a la vista en la BD.
3. **Findes con placeholder laborable.** 6–7 jun usan perfil laborable (seed047/048) porque no hay
   perfil finde/feriado calibrado. **Deuda ya registrada** en
   `documentation/ESTADO_Y_PROXIMOS_PASOS.md` (§ "Perfiles de día sin calibrar", líneas ~230–242) —
   no se duplica aquí, solo se referencia.

## Cómo disparar la semana completa

```
.venv/bin/python scripts/seed_congestion_calendar.py --execute
```

2026-06-01 ya está sembrado (stage-gate, día real del calendario → no se revirtió): en la corrida
completa será un no-op rápido y se sembrarán los **7 días restantes** (~27 min, ~11.5 GB). Vista
previa sin tocar la BD: el mismo comando sin `--execute`.
