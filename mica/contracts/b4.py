"""B4: the control context c_k — the ONE thing the intent module hands the decoder.

This is D0's arm-swap point. The four experiment arms differ only in what fills the
intent slot of this record; everything else — the evidence vector, the field types,
the dimensions — is identical by construction, so the decoder and the gate cannot
tell arms apart except through intent content. That is what makes the comparison
causal (D0's arm table).

Information flow is strictly one-way (D4): the belief flows INTO this record; nothing
the decoder produces ever flows back into the belief. And per contract C4, no human
evidence reaches the decoder except through this record — the agent's own action
history is the only other input, and that is the agent conditioning on itself, never
scoring the human (the do-operator rule).

The intent slot is deliberately explicit AND maskable:
  slot_kind      which mechanism filled it (arm0_zero / arm1_dense / arm2_llm / arm3)
  goal_marginal  the mechanism's goal distribution (all zeros for arm0_zero)
  top_goal, p_top, entropy_nats, p_z1   the named symbolic reads the D4 design pins
                 (entropy in NATS, the tracker's entropy() convention; p_z1 is the
                 neutral 0.5 for mechanisms with no mode estimate)
The slot is intervenable: context.py can rebuild the whole record from a point-mass
posterior on a chosen goal, which is what the S2 verifier and the intervenability
proof-test require — a real recompute, never an in-place overwrite.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .goals import GOALS

SLOT_KINDS = ("arm0_zero", "arm1_dense", "arm2_llm", "arm3")

# Mechanisms with no reasoning-mode estimate get this constant, the mode kernel's
# stationary point — "no information either way", not "confidently deliberate".
NEUTRAL_P_Z1 = 0.5


@dataclass(frozen=True)
class ControlContext:
    """One correction step's decoder input. Same shape in every arm."""

    tick: int
    k: int                                # evidence-step index within the session
    belief_snapshot_id: int               # := the correction tick (readiness review F2)
    slot_kind: str
    goal_marginal: tuple[float, ...]      # |G| values; all zeros only for arm0_zero
    top_goal: str | None                  # None only for arm0_zero
    p_top: float
    entropy_nats: float
    p_z1: float
    evidence: tuple[float, ...]           # the arm-invariant dense evidence vector
    history: tuple[int, ...] = ()         # the agent's OWN prior action tokens (C4)

    def __post_init__(self):
        if self.slot_kind not in SLOT_KINDS:
            raise ValueError(f"unknown slot kind {self.slot_kind!r}")
        if len(self.goal_marginal) != len(GOALS):
            raise ValueError(f"goal_marginal must have {len(GOALS)} entries")
        total = sum(self.goal_marginal)
        if self.slot_kind == "arm0_zero":
            if total != 0.0 or self.top_goal is not None:
                raise ValueError("arm0_zero must carry an all-zero, goal-less slot")
        # tolerance sized for the corpus's 6-decimal slot storage, not for live
        # float error alone — a marginal off by 1e-4 is storage rounding, not a bug
        elif not math.isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError(f"goal_marginal sums to {total}, not 1")
        for name, value in (("p_top", self.p_top), ("p_z1", self.p_z1)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value} outside [0, 1]")
        if self.entropy_nats < 0.0:
            raise ValueError("entropy cannot be negative")
        if not all(math.isfinite(v) for v in self.evidence):
            raise ValueError("evidence vector carries a non-finite value")
