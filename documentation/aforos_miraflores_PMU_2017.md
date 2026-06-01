# Aforos vehiculares Miraflores — insumo para OD del STGNN

**Fuente:** Plan de Movilidad Urbana de Miraflores 2017–2020, Gerencia de Desarrollo Urbano y Medio Ambiente, Municipalidad de Miraflores. Datos de aforo: P.U.D. Miraflores (base ~2016). Matriz OD distrital: Convenio Telefónica S.A.

**Advertencia de uso:** son **volúmenes totales por intersección en hora pico (veh/hr)**, NO flujos direccionales por arista. La magnitud la ancla este documento; el reparto por movimiento/sentido lo aporta el conocimiento de dominio. Dato de 2017 → citar como orden de magnitud y estructura, no como tráfico actual.

---

## 1. Matriz OD distrital (para inyección de tráfico en el borde de la red)

Miraflores: ~54.000 viajes internos/día generados; ~299.664 viajes externos/día recibidos (3.5 visitantes por habitante; pob. ~85 mil).

### Viajes internos por zona de origen (veh/día)
| Zona | Avenidas características | Viajes/día |
|---|---|---|
| Centro y Sur | Larco, 28 de Julio | 20.566 |
| Centro Norte | Pardo, Espinar | 12.177 |
| Sur Este | Aurora | 8.533 |
| Nor Oeste | Santa Cruz | 7.619 |
| Norte | Arequipa, Angamos | 5.086 |
| **Total interno** | | **53.981** |

### Viajes externos — top distritos de origen (define sentidos de entrada dominantes)
| Distrito de origen | Viajes/día | Borde de entrada probable |
|---|---|---|
| Santiago de Surco | 34.744 | SE / E (Benavides, Rep. Panamá) |
| Chorrillos | 27.034 | S (Costa Verde, Reducto, Benavides) |
| San Juan de Miraflores | 25.947 | SE (Benavides, Tomás Marsano) |
| Surquillo | 20.279 | E / NE (Angamos, Rep. Panamá, Aramburú) |
| Barranco | 13.342 | S (Larco/Diagonal, Costa Verde) |
| San Borja | 12.212 | E (Angamos, Aramburú) |
| San Isidro | 8.258 | N / NE (Pardo, Santa Cruz, Arequipa) |

*Reparto del borde es interpretación geográfica — confirmar contra los edges de entrada reales de la red.*

---

## 2. Aforos por intersección en hora pico (veh/hr)

Columna **¿En red Miraflores?**: clasificación tentativa según ubicación geográfica. `SÍ` = dentro del distrito; `BORDE` = límite distrital, verificar; `FUERA` = probablemente fuera del polígono (Surco/Santiago de Surco), filtrar salvo que la red los incluya como nodos de borde.

| Intersección | Veh/hr | Período | Nivel servicio | Saturación | ¿En red Miraflores? |
|---|---|---|---|---|---|
| Tomás Marsano / Villarán | 7.407 | PM | F | H | FUERA (Surco) |
| Paseo de la República / Angamos | 5.931 | AM | E | H | SÍ |
| Tomás Marsano / Roque y Boloña | 4.805 | PM | F | H | FUERA (Surco) |
| Cmte. Espinar / Pardo | 4.377 | AM | F | H | SÍ |
| Aramburú / Arequipa | 4.338 | AM | F | H | SÍ |
| Rep. Panamá / Ricardo Palma | 4.333 | AM | F | H | SÍ |
| Santa Cruz / Cmte. Espinar | 5.611 | AM | F | F | SÍ |
| Benavides / La Merced | 4.545–4.554 | AM | F | G | SÍ |
| Benavides / Panamá | 4.358 | AM | F | G | SÍ |
| 28 de Julio / Reducto | 4.271 | PM | E | H | SÍ |
| Rep. Panamá / Roque y Boloña | 3.686 | AM | F | H | BORDE |
| Benavides / Óvalo Higuereta | 3.993 | PM | F | G | FUERA (Surco) |
| Paseo de la República / 28 de Julio | 3.978 | AM | E | H | SÍ |
| Arequipa / Angamos | 3.408 | AM | E | H | SÍ |
| Puente Villena | 2.874 | AM | F | H | BORDE (Costa Verde) |
| Cmte. Espinar / Angamos | 3.905 | AM | F | F | SÍ |
| Paseo de la República / González Prada | 1.833 | AM | F | H | SÍ |
| Armendáriz / Vasco Núñez | 2.956 | PM | F | F | BORDE (límite Barranco) |
| Aramburú / Paseo de la República | 5.090 | AM | D | D | SÍ |
| Paseo de la República / Ricardo Palma | 3.818 | PM | D | F | SÍ |
| 28 de Julio / Panamá | 2.731 | AM | D | G | SÍ |
| Paseo de la República / Benavides | 3.778 | AM | D | E | SÍ |
| 28 de Julio / La Paz | 2.209 | AM | D | F | SÍ |
| Óvalo (Larco/Pardo/Arequipa) | 5.329 | AM | C-D-F | A-B | SÍ |
| Paseo de la República / Diez Canseco | 4.101 | AM | B | E | SÍ |
| Benavides / Larco | 3.390–3.805 | AM | C | D | SÍ |
| Pardo / Del Ejército | 3.850 | — | F | — | SÍ |
| Pardo / Santa Cruz | 3.850 | AM | — | — | SÍ |
| 28 de Julio / Larco | 2.255 | AM | C | B | SÍ |
| Aramburú / Petit Thouars | 3.174 | PM | B | A | SÍ |
| Paseo de la República / Schell | 2.659 | AM | B-C | A-B | SÍ |
| Ricardo Palma / La Paz | 3.796 | AM | — | — | SÍ |
| Puente Armendáriz | 5.987 | AM | — | — | BORDE (Costa Verde / Barranco) |
| Pérez Araníbar / Bajada San Martín | 5.830 | AM | — | — | BORDE (límite San Isidro) |
| Armendáriz / La Paz | 3.266 | AM | — | — | SÍ |

---

## 3. Notas para el recalibrado de perfil_dia_corredor.yaml

- **Avenidas piloto ya identificadas (Arequipa/Angamos, Larco/Benavides, Óvalo Gutiérrez, Pardo/Espinar):** todas tienen aforo aquí salvo Óvalo Gutiérrez (es límite con San Isidro, puede no estar en la tabla con ese nombre — verificar Santa Cruz/Cmte. Espinar como proxy de esa zona).
- **Pico AM domina** en la mayoría de ejes de entrada al distrito (viajes hacia el centro financiero). Reducto, Petit Thouars, Villarán y Roque y Boloña marcan PM. Esto sugiere al menos dos formas de perfil laborable (AM-dominante vs PM-dominante) si se quiere fidelidad.
- **Free-flow por edge:** mantener `min(lane.speed, vType.maxSpeed=13.89)` como se acordó; el PDF no aporta velocidades de free-flow utilizables.
- **Filtrado:** descartar las filas `FUERA` salvo que `miraflores.net.xml` las incluya. Confirmar las `BORDE` contra los edges de frontera reales: son buenos candidatos a puntos de inyección de demanda externa.
- **Reparto direccional:** ninguno de estos números es por sentido. Hay que dividir el volumen del cruce entre movimientos según geometría + dominancia de avenida.
