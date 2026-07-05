"""Source B — completed-build hindsight labeling (D3's data recipe, implemented).

The idea: a FINISHED structure is easy to recognize even when the same structure
half-built is ambiguous. So we take a finished session, fix its goal label once from
the completed build, and then attach that one label to every step of the build's
replay — turning one finished build into many (early partial evidence -> next action)
training pairs. The early pairs are the valuable ones: they teach a head what pursuing
a goal looks like before the structure gives it away.

The anti-circularity rule this module exists to protect: the LABEL comes from the
100%-complete structure (a state no likelihood head ever sees at inference time),
recognized by D2's finished-structure matcher — while the PAIRS carry only partial,
strictly pre-action evidence. Labeler signal and inference signal never coincide.

Keep/discard: a finished build whose best category match is weak, or whose top two
categories are nearly tied, is DISCARDED rather than labeled — one wrong label poisons
every pair replayed from that build. The two constants below define "weak" and
"nearly tied"; the labeling report prints them and the kept/discarded fraction, so the
discard rate is visible, never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..contracts.b0 import is_agent_actor
from ..contracts.b3 import fuse_dicts
from ..perception.evidence3d import read_finished_build
from ..perception.voxel_replay import Region, ReplayWorld

# The confidence rule, pinned (and printed into every report):
#   score  = fit x comp of the winning instance (the settled D2 ranking key —
#            the shape is present AND the player put it there)
#   kept   = score >= MIN_SCORE and (score - runner_up_score) >= MIN_MARGIN
# Set against the scripted corpus (labels known by construction) so that every kept
# label there is correct; the real free build (treehouse) is judged by the same bar.
MIN_SCORE = 0.20
MIN_MARGIN = 0.05


@dataclass(frozen=True)
class FinishedLabel:
    """What the recognizer says a finished build is — and whether to trust it."""

    goal: str                 # winning category
    subtype: str              # winning template instance (the style read)
    score: float              # fit x comp of the winner
    fit: float
    comp: float
    margin: float             # winner's score minus the runner-up category's
    runner_up: str
    kept: bool
    reason: str               # "" when kept; why it was discarded otherwise


def finished_world(packets, region: Region, base: dict) -> ReplayWorld:
    """Replay EVERY human event to the end — the completed structure the label reads.

    Agent blocks are dropped here (A7): the label must describe what the HUMAN built.
    The world object reports events outside the region (escaped) — callers should treat
    a session with escapes as unlabelable, since the finished build is partly invisible.
    """
    world = ReplayWorld(region, dict(base))
    for packet in sorted(packets, key=lambda p: p.tick):
        for event in packet.server.block_events:
            if not is_agent_actor(event.actor):
                world.apply(event)
    return world


def label_finished_build(world: ReplayWorld) -> FinishedLabel:
    """One label for one completed world, or a reasoned refusal."""
    readings = read_finished_build(world)
    ranked = sorted(
        ((reading.fit * reading.comp, goal, reading, subtype)
         for goal, (reading, subtype) in readings.items()),
        key=lambda entry: entry[0], reverse=True)
    best_score, goal, reading, subtype = ranked[0]
    runner_score, runner_goal = ranked[1][0], ranked[1][1]
    margin = best_score - runner_score
    if world.escaped:
        # checked before "nothing built": escapes are usually WHY nothing is visible
        kept, reason = False, f"{len(world.escaped)} events escaped the region — build partly invisible"
    elif not world.built():
        kept, reason = False, "nothing built"
    elif best_score < MIN_SCORE:
        kept, reason = False, f"best match {best_score:.3f} below threshold {MIN_SCORE}"
    elif margin < MIN_MARGIN:
        kept, reason = False, f"near tie with {runner_goal} (margin {margin:.3f} < {MIN_MARGIN})"
    else:
        kept, reason = True, ""
    return FinishedLabel(goal=goal, subtype=subtype, score=round(best_score, 4),
                         fit=round(reading.fit, 4), comp=round(reading.comp, 4),
                         margin=round(margin, 4), runner_up=runner_goal,
                         kept=kept, reason=reason)


def build_pairs(b1_dicts: list[dict], b2_dicts: list[dict],
                label: FinishedLabel, session_id: str) -> list[dict]:
    """The replay pairs for one KEPT session: each scored B1 record with its B2
    partner, tagged with the finished-build label and the build-progress fraction.

    The evidence in each pair is pre-action by construction — the records come from
    the same streams the runtime uses, whose windows end strictly before the scored
    action. The join is VERIFIED per pair (fuse_dicts refuses a tick or event-id
    mismatch), so a mislabeled pairing cannot slip through silently.
    """
    scored = [record for record in b1_dicts if record.get("scored")]
    if len(scored) != len(b2_dicts):
        raise ValueError(f"{session_id}: {len(scored)} scored B1 vs {len(b2_dicts)} B2 records")
    total_events = sum(len(record["event_ids"]) for record in scored) or 1
    pairs = []
    consumed = 0
    for b1, b2 in zip(scored, b2_dicts):
        fuse_dicts(b1, b2)                      # the verified join; raises on mismatch
        pairs.append({
            "session": session_id,
            "label": {"goal": label.goal, "subtype": label.subtype,
                      "score": label.score, "margin": label.margin},
            "progress": round(consumed / total_events, 4),   # fraction built BEFORE this action
            "next_action": b1["a_hat"],
            "b1": b1,
            "b2": b2,
        })
        consumed += len(b1["event_ids"])
    return pairs
