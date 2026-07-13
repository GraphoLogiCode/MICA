"""B6: one gate decision — which interaction mode the agent is in, and why.

Schema only, in this phase: the enum, the inputs snapshot, and the decision record,
exactly as D0's B6 box writes them. The finite-state machine that PRODUCES these
records — the hysteresis, the idle/proximity veto, the safety lattice — is D5's own
deliverable (Phase G part 2) and wraps the commit gate built in mica/gate/. Putting
the schema here first means the gate code, the FSM, and the proof log all speak one
type from day one.

The six inputs_snapshot fields are D0's canonical feature set, used verbatim so the
contract has one home (D5 §2). EXECUTE_CHUNK stays in the enum but is disabled by
config in v1 — the first live demo runs advisory + preview only, with even
PLACE_LOW_RISK config-disabled until the pinned conditions hold (D5 §9).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GateState(Enum):
    OBSERVE = "observe"
    SUGGEST = "suggest"
    PREVIEW = "preview"                  # ghost hologram; nothing placed
    PLACE_LOW_RISK = "place_low_risk"    # reversible blocks only
    GATHER = "gather"                    # fetch whitelisted materials (D5 §4 amendment
                                         # 2026-07-10); config-disabled by default
    EXECUTE_CHUNK = "execute_chunk"      # in the enum, config-disabled in v1
    YIELD = "yield"                      # step back from the build region


@dataclass(frozen=True)
class InputsSnapshot:
    """The belief and safety features the decision was made from (D0's six)."""

    top_goal: str | None
    p_top_goal: float
    belief_entropy: float                # nats — the tracker's entropy() convention
    p_z1: float
    proximity: float | None              # blocks from human to the agent's target;
                                         # None when no target exists this read
    reversibility: bool | None           # committed prefix entirely reversible?
                                         # None when nothing is committed


@dataclass(frozen=True)
class GateDecision:
    """One gate read's outcome. Every decision explains itself (the reason field)."""

    state: GateState
    inputs_snapshot: InputsSnapshot
    reason: str
