"""C_eta: builds the B4 control context from fused evidence and an intent read.

One module, four fill styles. The EVIDENCE half of the context is identical in every
arm — the same dense channels the likelihood heads read (via mica/intent/features.py,
the single feature definition), the frozen Phase-E adapter embedding e_k, and two
small origin-anchored spatial reads (where the crosshair rests and where the build's
center sits, relative to the build origin). The INTENT slot is the only part that
varies, and each mechanism's mapping into it is written here once and frozen (C5/C6):

  arm3        the tracker's belief: category marginal, top goal, full-belief entropy
              in nats, P(z=1)
  arm1_dense  the implicit classifier's distribution; entropy of that distribution;
              neutral P(z=1) (no mode estimate exists)
  arm2_llm    the LLM's stated-confidence distribution (D4-S1 mapping, computed in
              mica/intent/arm2.py); entropy of it; neutral P(z=1)
  arm0_zero   all zeros — no intent mechanism at all

The BUILD ORIGIN rule is pinned here because every consumer must use the same one:
the origin is the rounded centroid of the EARLIEST evidence record with a built cell.
That is observable from banked evidence alone (no raw capture needed), goal-free, and
arm-invariant. All grammar coordinates are offsets from this origin.

intervened_belief is the S2 door and the intervenability proof-test's subject: a
point-mass posterior on a forced goal, with the mode marginal kept — the full context
is then REBUILT from it (never patched in place), so forced contexts go through the
exact code real ones do.
"""
from __future__ import annotations

import math

from ..contracts.b1 import GOALS
from ..contracts.b3 import FusedEvidence
from ..contracts.b4 import NEUTRAL_P_Z1, ControlContext
from ..intent import features, heads_v1
from ..intent.tracker import Belief, category_marginal, entropy, mode_marginal
from .grammar import COORD_RANGE

ADAPTER_DIM = 64      # the Phase-E adapter's e_k width (models/heads_v1 trunk output)
# shared(14) + global(9) + per-goal blocks(5*7) + e_k + flag + focus(3+1) + centroid(3+1)
EVIDENCE_DIM = (features.SHARED_DIM + features.GLOBAL_DIM
                + len(GOALS) * features.GOAL_DIM + ADAPTER_DIM + 1 + 8)


def build_origin(fused_records) -> tuple[int, int, int] | None:
    """The session's coordinate anchor: where building started.

    Rounded centroid of the earliest record that has anything built — with one cell
    down, the centroid IS that cell. None until something is built."""
    for fused in fused_records:
        g = fused.global_feats
        if g.built_count > 0 and g.centroid is not None:
            return (round(g.centroid[0]), round(g.centroid[1]), round(g.centroid[2]))
    return None


def _offset_channel(point, origin) -> list[float]:
    """A world point as a squashed offset from the origin, plus a presence flag."""
    if point is None or origin is None:
        return [0.0, 0.0, 0.0, 0.0]
    return [max(-1.0, min(1.0, (p - o) / COORD_RANGE)) for p, o in zip(point, origin)] + [1.0]


def evidence_vector(fused: FusedEvidence, origin: tuple[int, int, int] | None) -> tuple[float, ...]:
    """The arm-invariant evidence half of c_k. Fixed width EVIDENCE_DIM."""
    goal_blocks = [v for goal in GOALS for v in features.goal_dense(fused, goal)]
    if heads_v1.available():
        e_k, e_flag = [float(v) for v in heads_v1.embed(fused)], 1.0
    else:
        e_k, e_flag = [0.0] * ADAPTER_DIM, 0.0
    focus_point = ((fused.focus.block.x, fused.focus.block.y, fused.focus.block.z)
                   if fused.focus.block else None)
    vector = (features.shared_dense(fused) + features.global_dense(fused) + goal_blocks
              + e_k + [e_flag]
              + _offset_channel(focus_point, origin)
              + _offset_channel(fused.global_feats.centroid, origin))
    assert len(vector) == EVIDENCE_DIM
    return tuple(vector)


def _dist_entropy(dist: dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in dist.values() if p > 0.0)


def _slot_from_distribution(dist: dict[str, float]) -> dict:
    top = max(dist, key=dist.get)
    return {"goal_marginal": tuple(dist[g] for g in GOALS), "top_goal": top,
            "p_top": dist[top], "entropy_nats": _dist_entropy(dist),
            "p_z1": NEUTRAL_P_Z1}


def arm3_slot(belief: Belief) -> dict:
    """The tracker's belief as slot content. Entropy is the FULL (goal, mode) belief's,
    in nats — the D4 pin; the marginal alone would understate the uncertainty."""
    marginal = category_marginal(belief)
    top = max(marginal, key=marginal.get)
    return {"goal_marginal": tuple(marginal[g] for g in GOALS), "top_goal": top,
            "p_top": marginal[top], "entropy_nats": entropy(belief),
            "p_z1": mode_marginal(belief)[1]}


def arm1_slot(dist: dict[str, float]) -> dict:
    return _slot_from_distribution(dist)


def arm2_slot(dist: dict[str, float]) -> dict:
    return _slot_from_distribution(dist)


def arm0_slot() -> dict:
    return {"goal_marginal": tuple(0.0 for _ in GOALS), "top_goal": None,
            "p_top": 0.0, "entropy_nats": 0.0, "p_z1": NEUTRAL_P_Z1}


def intervened_belief(belief: Belief, goal: str) -> Belief:
    """do(goal := g) at the belief level: all goal mass onto the forced goal, the
    mode marginal preserved. The caller rebuilds the whole context from this."""
    modes = mode_marginal(belief)
    return {(g, z): (modes[z] if g == goal else 0.0)
            for (g, z) in belief}


def control_context(fused: FusedEvidence, k: int, slot_kind: str, slot: dict,
                    origin: tuple[int, int, int] | None,
                    evidence: tuple[float, ...] | None = None,
                    history: tuple[int, ...] = ()) -> ControlContext:
    """Assemble one B4 record. `evidence` may be passed in when the caller already
    computed it (the corpus builder does, once per record)."""
    return ControlContext(
        tick=fused.tick, k=k, belief_snapshot_id=fused.tick,
        slot_kind=slot_kind,
        goal_marginal=tuple(slot["goal_marginal"]),
        top_goal=slot["top_goal"], p_top=slot["p_top"],
        entropy_nats=slot["entropy_nats"], p_z1=slot["p_z1"],
        evidence=evidence if evidence is not None else evidence_vector(fused, origin),
        history=history,
    )
