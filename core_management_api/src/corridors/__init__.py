"""Dominio de corredores TomTom (track feature/tomtom, Fase B-2).

Dado un corredor (cadena ordenada de aristas de ``graph_edges``) y las respuestas de
``flowSegmentData`` que el FRONTEND ya obtuvo de TomTom, decide para cada arista a qué
segmento OpenLR de TomTom pertenece (matching geométrico + desambiguación por sentido) y
persiste el mapping en ``corridor_edges``.

Frontera dura (ToS TomTom 11.4 / 11.6.1): el backend NUNCA consulta TomTom y la GEOMETRÍA
de TomTom es INPUT EFÍMERO — entra al endpoint, se usa para el cálculo, se descarta. Lo
único que se persiste es ``tomtom_openlr`` (ID, string) + ``edge_id`` + ``sequence``.
"""
