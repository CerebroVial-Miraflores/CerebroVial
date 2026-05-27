# language: es
Característica: Visualización pasiva de la estrategia de control activa (CA-05.1)
  Como Operador de Tráfico Municipal autenticado
  Quiero ver el nombre de la estrategia vigente y los tiempos de verde por acceso de mi intersección
  Para entender qué decisión de control automático está aplicándose en este momento

  Antecedentes:
    Dado existe una decisión "webster" registrada para "larco_schell"
    Y esa decisión fue activada como estrategia vigente para "larco_schell"

  Escenario: El Operador obtiene la estrategia vigente y los tiempos de verde por fase
    Dado un usuario autenticado con rol "operator"
    Cuando ese usuario hace "GET /control/active-state/larco_schell"
    Entonces la respuesta tiene código HTTP 200
    Y la respuesta contiene el campo "strategy_mode" con valor "webster"
    Y la respuesta contiene el campo "cycle_seconds" con un número
    Y la respuesta contiene una lista no vacía de "phase_timings"
    Y cada elemento de "phase_timings" tiene los campos "phase_id", "green", "yellow", "all_red"
