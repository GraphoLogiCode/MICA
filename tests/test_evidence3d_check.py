"""The B2 contract validator accepts a clean Evidence3D stream and flags each kind of
violation — so a PASS on real D2 output means the records are safe for D3."""
import dataclasses

from mica.capture.sample_builds import pen_build
from mica.capture.synthetic import generate_session
from mica.perception.evidence2d import evidence_stream
from mica.perception.evidence3d import build_evidence3d
from mica.perception.voxel_replay import ReplayWorld, region_around_events
from mica.validation.evidence3d_check import check_join, check_record, check_stream


def _stream():
    session = generate_session(pen_build())
    corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    world = ReplayWorld(region_around_events(session), {})
    return corrections, build_evidence3d(world, corrections, events)


def test_clean_stream_passes():
    corrections, records = _stream()
    assert check_stream(records) == []
    assert check_join(list(corrections), records) == []


def test_out_of_range_feature_flagged():
    _, records = _stream()
    goal, feats = next(iter(records[-1].per_goal.items()))
    broken = dict(records[-1].per_goal)
    broken[goal] = dataclasses.replace(feats, comp=1.7)
    assert any("comp" in m for m in check_record(dataclasses.replace(records[-1], per_goal=broken)))


def test_unscored_record_flagged():
    _, records = _stream()
    assert any("scored" in m for m in check_record(dataclasses.replace(records[0], scored=False)))


def test_double_consumption_flagged():
    _, records = _stream()
    doubled = records + [next(r for r in records if r.event_ids)]
    assert any("consumption" in m for m in check_stream(doubled))


def test_join_mismatch_flagged():
    corrections, records = _stream()
    shifted = [dataclasses.replace(records[0], tick=records[0].tick + 1)] + records[1:]
    assert any("tick mismatch" in m for m in check_join(list(corrections), shifted))
