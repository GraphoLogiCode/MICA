"""B3: one correction's fused evidence — the verified join of a B1 and a B2 record.

This is the D3 boundary from D0's B3 box, as a contract instead of an ad-hoc dict.
The constructor CHECKS the handoff: the two records must describe the same correction
(same action tick, same consumed event ids) or fusion refuses — a mismatched pair is a
pipeline bug, never something to paper over silently.

What flows through, and to whom (the goal-symmetry rule is part of the schema):
  shared (adapter + BOTH heads):  state_feats, focus, global_feats, h2d, h3d
  DELIBERATIVE HEAD ONLY:         s_goal, per_goal   (g-indexed — feeding them to the
                                  heuristic head would break mode identifiability)
  TARGET, NEVER A FEATURE:        a_hat and event_ids are the scored action and its
                                  provenance — readable for indexing and labels, never
                                  as head or adapter input (the head must PREDICT a_k,
                                  not read it; 09-F1). idle is defined FROM a_hat
                                  (idle == a_hat is IDLE, validator-enforced), so it
                                  is one bit of the target: the FSM may read it (the
                                  gate predicts nothing), the featurizer may not —
                                  it sat in shared_dense until 2026-07-16 and let both
                                  trained heads read the answer (review 13 F1).

The dense fused vector e_k = A_phi(h2d, state_feats-embedding, h3d) is the TRAINED
adapter's output and does not exist until Phase E trains it; in the hand-coded phase
the heads read these symbolic channels directly. Nothing here may ever carry belief.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .b0 import BlockPos
from .b1 import Evidence2D, Focus, MacroAction, StateFeats
from .b2 import Evidence3D, GlobalStructure, PerGoalStructure, Pose


@dataclass(frozen=True)
class FusedEvidence:
    """Everything the intent module may read for one correction step."""

    tick: int
    event_ids: tuple[int, ...]
    a_hat: MacroAction
    idle: bool
    state_feats: StateFeats
    focus: Focus
    global_feats: GlobalStructure
    s_goal: tuple[float, ...] | None                 # deliberative head only
    per_goal: Mapping[str, PerGoalStructure]         # deliberative head only
    h2d: tuple[float, ...] | None = None
    h3d: tuple[float, ...] | None = None


def fuse(b1: Evidence2D, b2: Evidence3D) -> FusedEvidence:
    """Join one scored B1 record with its B2 record, verifying they are the same step."""
    if not b1.scored:
        raise ValueError("fusion consumes scored corrections only; got a context record")
    action_tick = b1.tick_range[1] + 1
    if action_tick != b2.tick:
        raise ValueError(f"join mismatch: B1 action tick {action_tick} vs B2 tick {b2.tick}")
    if tuple(b1.event_ids) != tuple(b2.event_ids):
        raise ValueError(f"join mismatch at tick {b2.tick}: "
                         f"B1 events {b1.event_ids} vs B2 events {b2.event_ids}")
    return FusedEvidence(
        tick=b2.tick,
        event_ids=tuple(b1.event_ids),
        a_hat=b1.a_hat,
        idle=b1.idle,
        state_feats=b1.state_feats,
        focus=b1.focus,
        global_feats=b2.global_feats,
        s_goal=b1.s_goal,
        per_goal=b2.per_goal,
        h2d=b1.h2d,
        h3d=b2.h3d,
    )


def fuse_dicts(b1: dict, b2: dict) -> FusedEvidence:
    """The same verified join, from the serialized jsonl forms (offline replays)."""
    sf = b1["state_feats"]
    fo = b1["focus"]
    typed_b1 = Evidence2D(
        tick_range=tuple(b1["tick_range"]),
        a_hat=MacroAction(b1["a_hat"]),
        a_hat_conf=b1["a_hat_conf"],
        idle=b1["idle"],
        state_feats=StateFeats(
            held_item=sf["held_item"], hotbar=tuple(sf["hotbar"]),
            pos_delta=tuple(sf["pos_delta"]), yaw_delta=sf["yaw_delta"],
            pitch_delta=sf["pitch_delta"], recent_actions=tuple(sf["recent_actions"]),
            inventory=(tuple((item, count) for item, count in sf["inventory"])
                       if sf.get("inventory") is not None else None),
            inventory_tick=sf.get("inventory_tick"),
        ),
        focus=Focus(block=None if fo["block"] is None else BlockPos(*fo["block"]),
                    dwell_ticks=fo["dwell_ticks"]),
        scored=b1["scored"],
        event_ids=tuple(b1["event_ids"]),
        h2d=tuple(b1["h2d"]) if b1.get("h2d") else None,
        s_goal=tuple(b1["s_goal"]) if b1.get("s_goal") else None,
    )
    g = b2["global"]
    typed_b2 = Evidence3D(
        tick=b2["tick"],
        event_ids=tuple(b2["event_ids"]),
        scored=b2["scored"],
        per_goal={
            goal: PerGoalStructure(
                comp=f["comp"], edit_distance=f["edit_distance"], fit=f["fit"],
                pose=Pose(**f["pose"]), subtype=f["subtype"], delta_comp=f["delta_comp"],
            )
            for goal, f in b2["per_goal"].items()
        },
        global_feats=GlobalStructure(
            built_count=g["built_count"],
            bbox=tuple(g["bbox"]) if g["bbox"] else None,
            centroid=tuple(g["centroid"]) if g["centroid"] else None,
            planar_runs=g["planar_runs"], has_enclosure=g["has_enclosure"],
            symmetry=g["symmetry"], symmetry_support=g["symmetry_support"],
            edit_locality=g["edit_locality"],
        ),
        h3d=tuple(b2["h3d"]) if b2.get("h3d") else None,
    )
    return fuse(typed_b1, typed_b2)


def fuse_streams(b1_dicts: list[dict], b2_dicts: list[dict],
                 session_id: str = "?") -> list[FusedEvidence]:
    """Fuse a whole session's serialized streams: the scored B1 records joined 1:1
    with the B2 records, COUNTS ASSERTED FIRST. Every offline eval used to pair
    these with a bare zip, which silently truncates on a mismatch — session
    210003's broken evidence hid inside every eval for a week that way (found
    2026-07-12). A mismatch now names the session and both counts, loudly."""
    scored = [record for record in b1_dicts if record.get("scored")]
    if len(scored) != len(b2_dicts):
        raise ValueError(
            f"{session_id}: {len(scored)} scored B1 vs {len(b2_dicts)} B2 records — "
            "the banked evidence is inconsistent; regenerate it (--redo-evidence) "
            "instead of evaluating a truncated stream")
    return [fuse_dicts(a, b) for a, b in zip(scored, b2_dicts)]
