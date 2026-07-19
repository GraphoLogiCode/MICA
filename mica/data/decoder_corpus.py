"""Training samples for the D4 action decoder: (control context -> helper trace).

The decoder learns one thing: given the belief-conditioned context at a correction
step, propose the actions that would usefully continue the build. The v1 corpus is
LOCAL-ONLY by pinned decision (2026-07-05): scripted sessions whose plans the
generator wrote, so the "right thing to do next" is known exactly — the remaining
plan cells — and the belief slot comes from a real tracker replay (trained v1 heads,
their jointly-fitted knobs) over the same evidence every other consumer reads.

The helper trace at step k is the plan's future placements in the plan's own order,
with two honesty rules: the helper never re-makes the builder's mistakes (mistake
placements are filtered out by pairing each break with the placement it corrects),
and a mistake block STANDING at step k becomes a break action (the helper cleans up).

DECODER-CORPUS HYGIENE (the D4 rule, this corpus's 09-F1): the context must end
strictly before the first predicted action's effects render. Asserted per sample:
  1. the evidence join is verified (fuse refuses mismatched records);
  2. every target place names a cell NOT yet standing at step k — a cell already in
     the world means the context contains its own prediction;
  3. every target break names a cell that IS standing — breaking nothing is a trace
     bug, not a training example.
A violation raises CorpusHygieneViolation and the sample never enters training.
"""
from __future__ import annotations

import json
import os

from ..capture.scripted_goals import BuildPlan, build_from_plan
from ..capture.synthetic import ScriptedPlacement
from ..contracts.b0 import BlockOp
from ..contracts.b3 import FusedEvidence
from ..decoder import context as context_builder
from ..decoder.grammar import COORD_RANGE, Action, Break, Place
from ..intent import arm1, heads_v1
from ..intent.tracker import correct, predict, uniform_belief

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORPUS_DIR = os.path.join(_ROOT, "capture", "decoder")

TARGET_ACTIONS = 12    # how far ahead one sample's helper trace looks
ARM2_IDLE_EVERY = 10   # the pinned Phase-F cadence, reused unchanged


class CorpusHygieneViolation(AssertionError):
    """A decoder sample whose context is not strictly ahead of its own targets."""


def plan_from_label(label: dict) -> BuildPlan:
    """Rebuild the exact BuildPlan a banked session was generated from.

    Some plan fields (like base_gap_ticks) were drawn from the per-goal Random
    stream, so the only faithful reconstruction is to re-run plan_variants with the
    same goal seed and pick the plan whose seed matches. The caller then verifies
    the regenerated event count against the banked label — a mismatch refuses the
    session rather than training on wrong targets."""
    from ..capture.scripted_goals import plan_variants

    seed = label["seed"]
    goal_seed, index = divmod(seed, 1000)
    for plan in plan_variants(label["goal"], index + 1, goal_seed):
        if plan.seed == seed:
            return plan
    raise ValueError(f"no plan with seed {seed} under goal {label['goal']}")


def skipped_place_ids(placements: tuple[ScriptedPlacement, ...]) -> set[int]:
    """Indices of placements a helper must never repeat. Two kinds, found in one
    replay of the plan:

      mistakes    each break pairs with the most recent live placement at its cell;
                  that placement is the mistake the break corrects
      redundant   a placement whose cell is ALREADY standing when it happens — the
                  generator's wrong-cell draw collided with an existing block (or a
                  scheduled break never fired before session end); in the real game
                  that click would not have landed, so a helper must not copy it

    When a break lands on a cell whose only live placement was real (its mistake
    twin was redundant), the real placement joins the skip set too: the plan's net
    outcome is that the cell ends empty, and the trace follows the plan's outcome."""
    open_places: dict[tuple[int, int, int], list[int]] = {}
    standing: set[tuple[int, int, int]] = set()
    skips: set[int] = set()
    for index, op in enumerate(placements):
        cell = (op.pos.x, op.pos.y, op.pos.z)
        if op.op is BlockOp.PLACE:
            if cell in standing:
                skips.add(index)
            else:
                standing.add(cell)
                open_places.setdefault(cell, []).append(index)
        else:
            standing.discard(cell)
            if open_places.get(cell):
                skips.add(open_places[cell].pop())
    return skips


def _standing_at(placements, action_tick: int) -> dict[tuple[int, int, int], str]:
    """Cells standing strictly BEFORE the action at action_tick (the pre-action world,
    matching what the evidence may contain)."""
    standing: dict[tuple[int, int, int], str] = {}
    for op in placements:
        if op.tick >= action_tick:
            break
        cell = (op.pos.x, op.pos.y, op.pos.z)
        if op.op is BlockOp.PLACE:
            standing[cell] = op.block_type
        else:
            standing.pop(cell, None)
    return standing


def helper_actions(placements: tuple[ScriptedPlacement, ...], skips: set[int],
                   action_tick: int, origin: tuple[int, int, int],
                   limit: int = TARGET_ACTIONS) -> list[Action]:
    """The helper trace at one step: what a helper should do from this moment on."""
    standing = _standing_at(placements, action_tick)
    actions: list[Action] = []
    for index, op in enumerate(placements):
        if op.tick < action_tick or len(actions) >= limit:
            continue
        cell = (op.pos.x, op.pos.y, op.pos.z)
        offset = (cell[0] - origin[0], cell[1] - origin[1], cell[2] - origin[2])
        if any(abs(v) > COORD_RANGE for v in offset):
            raise CorpusHygieneViolation(
                f"target cell {cell} is {offset} from the origin — outside the "
                f"grammar's +/-{COORD_RANGE} range")
        if op.op is BlockOp.PLACE:
            if index in skips:
                continue        # mistakes and redundant clicks are never re-made
            if cell in standing:
                raise CorpusHygieneViolation(
                    f"target place at {cell} is already standing at tick "
                    f"{action_tick} — the context would contain its own prediction")
            actions.append(Place(*offset, block=op.block_type))
            standing[cell] = op.block_type
        else:
            if cell not in standing:
                continue                       # its mistake was filtered out above
            actions.append(Break(*offset))
            standing.pop(cell)
    return actions


def replay_beliefs(fused_records: list[FusedEvidence]) -> list[dict]:
    """The belief after each correction: the exact filter loop, trained v1 heads,
    their jointly-fitted knobs. Returns the full Belief objects — the commit gate
    needs them for kernel propagation, not just the slot read-offs."""
    params = heads_v1.tracker_params()
    belief = uniform_belief()
    previous_tick = 0
    beliefs = []
    for fused in fused_records:
        gap_seconds = max(fused.tick - previous_tick, 1) / 20.0
        previous_tick = fused.tick
        belief = predict(belief, gap_seconds, params)
        belief, _ = correct(belief, heads_v1.likelihood(fused, fused.a_hat), params)
        beliefs.append(belief)
    return beliefs


def replay_belief_slots(fused_records: list[FusedEvidence]) -> list[dict]:
    """The arm3 slot per step — the belief replay folded to slot content."""
    return [context_builder.arm3_slot(belief)
            for belief in replay_beliefs(fused_records)]


def session_rows(session_id: str, label: dict, fused_records: list[FusedEvidence],
                 placements: tuple[ScriptedPlacement, ...],
                 arm2_reader=None) -> list[dict]:
    """Every decoder sample one session yields, hygiene-asserted, ready to serialize."""
    origin = context_builder.build_origin(fused_records)
    if origin is None:
        return []
    skips = skipped_place_ids(placements)
    arm3_slots = replay_belief_slots(fused_records)
    arm1_ready = arm1.available()

    total_events = sum(len(f.event_ids) for f in fused_records) or 1
    consumed = 0
    held_arm2: dict | None = None
    since_query = 0
    rows = []
    for k, fused in enumerate(fused_records):
        progress = consumed / total_events
        consumed += len(fused.event_ids)
        slots: dict = {"arm3": arm3_slots[k],
                       "arm1": context_builder.arm1_slot(arm1.distribution(fused))
                               if arm1_ready else None,
                       "arm2": None}
        if arm2_reader is not None:
            since_query += 1
            if fused.event_ids or held_arm2 is None or since_query >= ARM2_IDLE_EVERY:
                held_arm2 = context_builder.arm2_slot(arm2_reader.read(fused))
                since_query = 0
            slots["arm2"] = held_arm2
        targets = helper_actions(placements, skips, fused.tick, origin)
        rows.append({
            "session": session_id, "k": k, "tick": fused.tick,
            "goal": label["goal"], "subtype": label["subtype"], "mode": label["mode"],
            "progress": round(progress, 4), "origin": list(origin),
            "evidence": [round(float(v), 6)
                         for v in context_builder.evidence_vector(fused, origin)],
            "slots": {name: (None if slot is None else _slot_to_json(slot))
                      for name, slot in slots.items()},
            "target": [_action_to_json(a) for a in targets],
        })
    return rows


def _slot_to_json(slot: dict) -> dict:
    return {"goal_marginal": [round(float(v), 6) for v in slot["goal_marginal"]],
            "top_goal": slot["top_goal"], "p_top": round(float(slot["p_top"]), 6),
            "entropy_nats": round(float(slot["entropy_nats"]), 6),
            "p_z1": round(float(slot["p_z1"]), 6)}


def _action_to_json(action: Action) -> dict:
    from ..decoder.grammar import action_to_json

    return action_to_json(action)


def real_session_placements(session) -> tuple[tuple[ScriptedPlacement, ...], int]:
    """A REAL session's human block events in the plan-shaped stream the target
    builder eats — (placements, agent_events_dropped).

    This is the label-free door (pre-registered 2026-07-19): the targets are what
    the human ACTUALLY did next, so contested and discarded sessions train the
    token stream with no goal label anywhere. The A7 rule applies at the source —
    agent-placed blocks are dropped and counted, exactly as they are excluded
    from evidence everywhere else. Downstream, mistake-pairing and every corpus
    hygiene assert run unchanged: a human's own break-corrects-place pattern is
    the same shape as a scripted plan's."""
    from ..contracts.b0 import is_agent_actor

    placements = []
    dropped = 0
    for packet in session.packets:
        for event in packet.server.block_events:
            if is_agent_actor(event.actor):
                dropped += 1
                continue
            placements.append(ScriptedPlacement(
                tick=packet.tick, pos=event.pos, block_type=event.block_type,
                op=event.op, actor=event.actor))
    return tuple(placements), dropped


def in_range_placements(placements: tuple[ScriptedPlacement, ...],
                        origin: tuple[int, int, int]
                        ) -> tuple[tuple[ScriptedPlacement, ...], int]:
    """Drop events outside the grammar's +/-COORD_RANGE of the origin — free
    builds occasionally wander; a scripted plan never does. Dropping (and
    counting) keeps the hygiene asserts as asserts instead of session-killers."""
    kept, dropped = [], 0
    for op in placements:
        offset = (op.pos.x - origin[0], op.pos.y - origin[1], op.pos.z - origin[2])
        if any(abs(v) > COORD_RANGE for v in offset):
            dropped += 1
            continue
        kept.append(op)
    return tuple(kept), dropped


REAL_LABEL = {"goal": None, "subtype": None, "mode": None}
# Real rows carry goal None: the rationale head never trains on them and the
# counterfactual sharpening never draws them — no label exists to leak.


def regenerate_placements(label: dict) -> tuple[ScriptedPlacement, ...]:
    """A banked session's placements, rebuilt from its label — verified by count."""
    build, rebuilt_label = build_from_plan(plan_from_label(label))
    if rebuilt_label["events"] != label["events"]:
        raise ValueError(f"{label.get('seed')}: reconstruction drifted — "
                         f"{rebuilt_label['events']} events vs banked {label['events']}")
    return build.placements


def load_rows(directory: str = CORPUS_DIR) -> tuple[list[dict], list[dict], dict]:
    """(train rows, eval rows, labels) from a written corpus directory.

    Only group == "train" may ever train: BOTH eval groups — the fresh holdout and
    the banked transfer sessions — go to the eval side. Routing by "not holdout"
    here once put the banked transfer group into training, which would have made
    the transfer number a memorization number."""
    with open(os.path.join(directory, "decoder_labels.json"), encoding="utf-8") as handle:
        labels = json.load(handle)
    train, evaluation = [], []
    for session_id, label in sorted(labels.items()):
        if label["group"] in ("real_pretrain", "real_holdout"):
            # The pre-registered NTP experiment's rows (2026-07-19): loaded
            # explicitly via load_group by the flag that asked for them — never
            # folded into the standard scripted train/eval split.
            continue
        path = os.path.join(directory, f"{session_id}.decoder.jsonl")
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        (train if label["group"] == "train" else evaluation).extend(rows)
    return train, evaluation, labels


def load_group(group: str, directory: str = CORPUS_DIR) -> list[dict]:
    """Every row of one named session group (e.g. real_pretrain, real_holdout)."""
    with open(os.path.join(directory, "decoder_labels.json"), encoding="utf-8") as handle:
        labels = json.load(handle)
    rows: list[dict] = []
    for session_id, label in sorted(labels.items()):
        if label["group"] != group:
            continue
        path = os.path.join(directory, f"{session_id}.decoder.jsonl")
        rows.extend(json.loads(line)
                    for line in open(path, encoding="utf-8") if line.strip())
    return rows
