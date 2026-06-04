"""Registry string→adaptador. ``ckpt["model"]`` decide qué adaptador construir.

Diseñado para que sumar un alias sea TRIVIAL: el rename futuro
``TimeThenSpaceModel → MirafloresSTGNNModel`` se resuelve agregando UNA línea
(``"MirafloresSTGNNModel": StgnnAdapter``) — ambos strings apuntan al mismo adaptador.
El remapeo de ``arch`` y la decisión grafo/no-grafo viven en el adaptador, no acá, así que
el alias no toca nada más.
"""
from __future__ import annotations

from .adapters import GruAdapter, InferenceAdapter, StgnnAdapter

# ckpt["model"] (string persistido en el .pt) -> clase de adaptador.
_REGISTRY: dict[str, type[InferenceAdapter]] = {
    "MirafloresGRUBaseline": GruAdapter,
    "TimeThenSpaceModel": StgnnAdapter,
    # Alias futuro (rename TimeThenSpaceModel -> MirafloresSTGNNModel): agregar
    #   "MirafloresSTGNNModel": StgnnAdapter,
    # y ambos checkpoints (viejo string y nuevo) resuelven al mismo adaptador.
}


def resolve(model_name: str) -> type[InferenceAdapter]:
    """Devuelve la clase de adaptador para ``ckpt["model"]``; error claro si no está.

    No adivina: un modelo no registrado es un error explícito (no un fallback silencioso).
    """
    if model_name not in _REGISTRY:
        raise ValueError(
            f"modelo desconocido en ckpt['model']={model_name!r}; "
            f"registrados: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[model_name]
