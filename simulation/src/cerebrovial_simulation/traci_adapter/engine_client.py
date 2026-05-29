"""Cliente HTTP para POST /control/recommend.

Contrato vivo: documentation/contracts/engine_recommend_contract.md
"""
from __future__ import annotations

import os
from typing import Optional

import requests


DEFAULT_ENGINE_URL = "http://localhost:8001/control/recommend"
DEFAULT_TIMEOUT_S = 2.0


class EngineUnavailable(RuntimeError):
    pass


def recommend(
    *,
    intersection_id: str,
    timestamp: str,
    phase_flows: list[dict],
    lost_time: float = 10.0,
    engine_url: Optional[str] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """Invoca el motor; devuelve el body del ControlRecommendation.

    Retorna el dict bajo la clave "data" (i.e. el ControlRecommendation
    directo, sin el wrapper RecommendResponse).
    """
    url = engine_url or os.environ.get("ENGINE_URL", DEFAULT_ENGINE_URL)
    payload = {
        "intersection_id": intersection_id,
        "timestamp": timestamp,
        "phases": phase_flows,
        "lost_time": lost_time,
        "predicted_demand": None,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
    except requests.RequestException as exc:
        raise EngineUnavailable(f"motor no responde en {url}: {exc}") from exc

    if resp.status_code != 200:
        raise EngineUnavailable(
            f"motor respondió {resp.status_code}: {resp.text[:300]}"
        )
    body = resp.json()
    # Contrato: RecommendResponse envuelve ControlRecommendation en "data".
    return body.get("data", body)
