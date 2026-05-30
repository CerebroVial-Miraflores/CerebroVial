"""Aplica el plan recibido del motor al tlLogic SUMO via TraCI.

Estrategia (Catch A + corrección 3 + Catch 3):
- **Sensado**: independiente del applier. Cuando el e2e tiene un plan
  nuevo, lo entrega via ``update_pending(rec)``.
- **Actuación**: el applier aplica el plan SOLO en el **borde de ciclo**
  (último sub-fase = EW_r idx 5, justo al wrappear a NS_g idx 0).
  Garantiza una aplicación por ciclo, modelando adaptación realista, y
  evita reiniciar el programa mid-fase con ``currentPhaseIndex=0`` que
  hambrearía EW.
- **Expansión PhaseTiming → 3 sub-fases SUMO** (corrección 3): cada
  ``PhaseTiming(green, yellow, all_red)`` del motor se expande en
  ``[Phase(green, G/g), Phase(yellow, y), Phase(all_red, r)]`` con los
  linkstates derivados en F1 (``conf/network/linkstates.json``).
  2 timings motor (NS, EW) ⇒ 6 sub-fases SUMO.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import traci


# Idx de la última sub-fase del programa baseline/adaptive (EW_r).
LAST_SUBPHASE_IDX = 5


def load_linkstates(linkstates_path: Path) -> dict[str, str]:
    return json.loads(linkstates_path.read_text())


def expand_timings_to_sumo_phases(
    timings: list[dict],
    linkstates: dict[str, str],
) -> list[traci.trafficlight.Phase]:
    """Expande la lista de PhaseTiming del motor a la lista plana de fases SUMO.

    Cada PhaseTiming = (green, yellow, all_red) ⇒ 3 Phases SUMO.
    Para 2 timings (NS, EW) ⇒ 6 Phases.
    """
    sumo_phases: list[traci.trafficlight.Phase] = []
    for t in timings:
        phase_id = t["phase_id"]
        for sub_name, dur in [("g", t["green"]), ("y", t["yellow"]), ("r", t["all_red"])]:
            key = f"{phase_id}_{sub_name}"
            if key not in linkstates:
                raise ValueError(f"linkstates faltante para {key!r}")
            sumo_phases.append(
                traci.trafficlight.Phase(duration=float(dur), state=linkstates[key])
            )
    return sumo_phases


class TllogicApplier:
    """Mantiene cache `pending_plan` y aplica en borde de ciclo."""

    def __init__(
        self,
        tls_id: str,
        linkstates: dict[str, str],
        program_id: str = "adaptive",
        last_subphase_idx: int = LAST_SUBPHASE_IDX,
    ) -> None:
        self.tls_id = tls_id
        self.linkstates = linkstates
        self.program_id = program_id
        self.last_subphase_idx = last_subphase_idx
        self.pending_plan: Optional[dict] = None
        self.applied_count = 0

    def update_pending(self, recommendation: dict) -> None:
        """Llamado tras cada consulta al motor (cada 30s sim)."""
        self.pending_plan = recommendation

    def maybe_apply(self, sim_time: float) -> bool:
        """Verifica si este step es borde de ciclo y aplica si hay plan pendiente.

        Retorna True si aplicó este step.
        """
        if self.pending_plan is None:
            return False

        current_idx = traci.trafficlight.getPhase(self.tls_id)
        if current_idx != self.last_subphase_idx:
            return False

        next_switch = traci.trafficlight.getNextSwitch(self.tls_id)
        # next_switch es el sim_time del próximo cambio. Si está
        # justo por arriba del sim_time actual (<= 1 step), estamos en
        # el step que cierra el ciclo.
        if not (sim_time <= next_switch <= sim_time + 1.001):
            return False

        # Aplica.
        sumo_phases = expand_timings_to_sumo_phases(
            self.pending_plan["phase_timings"], self.linkstates
        )
        logic = traci.trafficlight.Logic(
            programID=self.program_id,
            type=0,
            currentPhaseIndex=0,
            phases=sumo_phases,
        )
        traci.trafficlight.setProgramLogic(self.tls_id, logic)
        # setProgramLogic NO reinicia el reloj de la fase 0 en SUMO 1.26: retiene el
        # next-switch agendado del EW_r saliente (observado: nextSwitch queda en
        # now+0.1–0.7s en vez de now+green_NS) → NS_g se trunca a ~1 step. Forzar el
        # arranque limpio de la fase 0 con su verde completo.
        traci.trafficlight.setPhase(self.tls_id, 0)
        traci.trafficlight.setPhaseDuration(self.tls_id, sumo_phases[0].duration)
        self.pending_plan = None
        self.applied_count += 1
        return True
