"""The B3 handoff: fusion joins exactly one B1 record with its own B2 record, carries
every channel across, and refuses mismatched pairs — nothing bypassed, nothing silent."""
import dataclasses

import pytest

from mica.capture.sample_builds import pen_build
from mica.capture.synthetic import generate_session
from mica.contracts.b3 import fuse
from mica.perception.evidence2d import evidence_stream
from mica.perception.evidence3d import build_evidence3d
from mica.perception.voxel_replay import ReplayWorld, region_around_events


def _pairs():
    session = generate_session(pen_build())
    corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    b2 = build_evidence3d(ReplayWorld(region_around_events(session), {}), corrections, events)
    return list(zip(corrections, b2))


def test_every_channel_arrives():
    b1, b2 = next((a, b) for a, b in _pairs() if a.event_ids)
    fused = fuse(b1, b2)
    # from D1: the whole behavior bundle, not just the action label
    assert fused.a_hat is b1.a_hat
    assert fused.state_feats is b1.state_feats
    assert fused.focus is b1.focus
    assert fused.idle is b1.idle
    # from D2: per-goal structure AND the global shape facts
    assert fused.per_goal is b2.per_goal
    assert fused.global_feats is b2.global_feats
    assert fused.event_ids == tuple(b1.event_ids) == tuple(b2.event_ids)


def test_held_item_reaches_the_evidence():
    # the synthetic builder equips what they are about to place, so the pre-action
    # window's held item is the fence — the D1 style cue the heads consume
    place_pairs = [(a, b) for a, b in _pairs() if a.event_ids]
    fused = fuse(*place_pairs[2])
    assert fused.state_feats.held_item == "minecraft:oak_fence"


def test_a_hat_conf_never_reaches_the_filter():
    # rule-certainty metadata may not multiply into likelihoods (D3 constraint N1).
    # The B3 contract enforces this by omission — fusion drops the field — so two B1
    # records differing only in a_hat_conf fuse identically and score identically,
    # and no head can ever read it (review 12-F5).
    from mica.intent.heads_v0 import likelihood

    b1, b2 = next((a, b) for a, b in _pairs() if a.event_ids)
    doctored = dataclasses.replace(b1, a_hat_conf=0.123)
    assert fuse(b1, b2) == fuse(doctored, b2)
    assert likelihood(fuse(b1, b2), b1.a_hat) == likelihood(fuse(doctored, b2), b1.a_hat)


def test_mismatched_pairs_are_refused():
    pairs = _pairs()
    (b1_a, b2_a), (_, b2_b) = pairs[0], pairs[1]
    with pytest.raises(ValueError):
        fuse(b1_a, b2_b)                                        # someone else's step
    with pytest.raises(ValueError):
        fuse(b1_a, dataclasses.replace(b2_a, tick=b2_a.tick + 1))   # shifted tick
    context = dataclasses.replace(b1_a, scored=False)
    with pytest.raises(ValueError):
        fuse(context, b2_a)                                     # context records never fuse
