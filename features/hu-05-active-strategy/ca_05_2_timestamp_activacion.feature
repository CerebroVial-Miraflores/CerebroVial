# language: es
Característica: Timestamp de activación de la estrategia vigente (CA-05.2)
  Como Operador
  Quiero conocer el timestamp en el que la estrategia vigente se activó
  Para saber cuánto tiempo lleva aplicándose la decisión actual

  # DHU-021 #13: decided_at (motor_decisions) y activated_at (engine_active_state)
  # son eventos distintos. La respuesta de HU-05 expone ambos para que la UI pueda
  # diferenciar "cuándo se calculó" de "cuándo entró en vigor".

  Antecedentes:
    Dado existe una decisión "max_pressure" registrada para "larco_schell"
    Y esa decisión fue activada como estrategia vigente para "larco_schell"

  Escenario: La respuesta incluye activated_at y decided_at como timestamps ISO-8601
    Dado un usuario autenticado con rol "operator"
    Cuando ese usuario hace "GET /control/active-state/larco_schell"
    Entonces la respuesta tiene código HTTP 200
    Y la respuesta contiene el campo "activated_at" con un timestamp ISO-8601
    Y la respuesta contiene el campo "decided_at" con un timestamp ISO-8601
