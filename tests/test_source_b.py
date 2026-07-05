"""Source B (D3's completed-build hindsight labeling), on synthetic fixtures.

What must hold: a finished template build labels correctly with confidence; builds
that cannot be trusted (nothing/too little built, near-ties, events outside the
region) are DISCARDED with the reason stated, never guessed; the agent's blocks
never reach the finished world the label reads (A7); and every replay pair is a
verified join carrying the one finished-build label plus a strictly-before-this-
action progress fraction.
"""
import json

from mica.capture.sample_builds import assisted_build, wall_row_build
from mica.capture.scripted_goals import build_from_plan, plan_variants
from mica.capture.synthetic import generate_session
from mica.contracts.b0 import is_agent_actor
from mica.contracts.serialize import evidence2d_to_dict, evidence3d_to_dict
from mica.data.source_b import (
    build_pairs, finished_world, label_finished_build,
)
from mica.perception.evidence2d import evidence_stream
from mica.perception.evidence3d import build_evidence3d
from mica.perception.voxel_replay import Region, ReplayWorld, region_around_events


def _scripted(goal="defense", seed=7):
    build, truth = build_from_plan(plan_variants(goal, 1, seed)[0])
    return build, truth, generate_session(build)


def test_finished_template_build_labels_correctly():
    build, truth, session = _scripted()
    world = finished_world(session.packets, region_around_events(session), {})
    label = label_finished_build(world)
    assert label.kept, label.reason
    assert label.goal == truth["goal"]
    assert label.subtype == truth["subtype"]
    assert label.margin >= 0.05


def test_too_little_built_is_discarded_not_guessed():
    build, _, session = _scripted()
    # keep only the first two placements' events: a couple of blocks say nothing
    world = ReplayWorld(region_around_events(session), {})
    events = sorted((e for p in session.packets for e in p.server.block_events),
                    key=lambda e: e.event_id)[:2]
    for event in events:
        world.apply(event)
    label = label_finished_build(world)
    assert not label.kept
    assert "below threshold" in label.reason or "near tie" in label.reason


def test_nothing_built_is_discarded():
    label = label_finished_build(ReplayWorld(Region(-8, 0, -8, 8, 8, 8), {}))
    assert not label.kept and label.reason == "nothing built"


def test_events_outside_the_region_make_the_build_unlabelable():
    _, _, session = _scripted()
    # a region far from every placement: the finished build is partly invisible
    world = finished_world(session.packets, Region(1000, 0, 1000, 1020, 20, 1020), {})
    label = label_finished_build(world)
    assert not label.kept and "escaped" in label.reason


def test_agent_blocks_never_reach_the_labeled_world():
    session = generate_session(assisted_build())
    agent_cells = {(e.pos.x, e.pos.y, e.pos.z)
                   for p in session.packets for e in p.server.block_events
                   if is_agent_actor(e.actor)}
    assert agent_cells, "fixture must contain agent placements"
    world = finished_world(session.packets, region_around_events(session), {})
    assert not (set(world.built()) & agent_cells)


def test_pairs_carry_the_label_and_a_pre_action_progress_fraction():
    build, truth, session = _scripted()
    region = region_around_events(session)
    label = label_finished_build(finished_world(session.packets, region, {}))
    records = tuple(evidence_stream(session.packets))
    corrections = tuple(r for r in records if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    b2 = build_evidence3d(ReplayWorld(region, {}), corrections, events)
    pairs = build_pairs([evidence2d_to_dict(r) for r in records],
                        [evidence3d_to_dict(r) for r in b2], label, build.session_id)

    assert len(pairs) == len(corrections)
    assert all(p["label"]["goal"] == truth["goal"] for p in pairs)
    # progress counts events strictly BEFORE each pair's action: starts at zero and
    # never decreases; a pair that consumes events can never sit at 1.0 (its own
    # events are not yet counted), while a trailing non-build action may.
    fractions = [p["progress"] for p in pairs]
    assert fractions[0] == 0.0
    assert all(a <= b for a, b in zip(fractions, fractions[1:]))
    assert all(p["progress"] < 1.0 for p in pairs if p["b1"]["event_ids"])
    # each pair's action is the B1 record's own a_hat — the (evidence -> action) shape
    assert all(p["next_action"] == p["b1"]["a_hat"] for p in pairs)
    # and the whole thing survives a JSONL round trip
    assert json.loads(json.dumps(pairs[0]))["label"]["subtype"] == truth["subtype"]
