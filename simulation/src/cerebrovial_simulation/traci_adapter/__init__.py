"""F4 — Adaptador TraCI ↔ motor adaptativo via HTTP externo.

Cierra CT-07.5: integración bidireccional con SUMO. El motor se invoca
vía HTTP (`POST /control/recommend` lockeado en puerto 8001, contrato
en `documentation/contracts/engine_recommend_contract.md`); el plan
emitido se aplica al tlLogic SUMO en **borde de ciclo** (Catch A) con
expansión a 6 sub-fases SUMO (corrección 3).
"""
