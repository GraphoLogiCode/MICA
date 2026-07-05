"""Golden equivalence: the streaming perception core must reproduce the batch driver
record for record, bit for bit, on every gap-free recording — sample builds, the whole
scripted-corpus plan set, and any gate-passing real capture on this machine. The
records are compared in their serialized (JSONL-line) form, so "equal" here means the
files on disk would be byte-identical. Also pins the streaming-only guarantees: records
leave in batch order with no future knowledge (prefix property), and a build run's
scored record waits for its last event id before leaving (deferred emission)."""
import glob
import os

import pytest

from mica.capture.jsonl_ingest import JsonlSource
from mica.capture.sample_builds import (
    bridge_deck_build,
    burst_place_build,
    mixed_build,
    pen_build,
    sparse_build,
    walk_only_build,
    wall_row_build,
)
from mica.capture.scripted_goals import build_from_plan, plan_variants
from mica.capture.synthetic import generate_session
from mica.contracts.b1 import GOALS, MacroAction
from mica.contracts.serialize import evidence2d_to_dict, evidence3d_to_dict
from mica.perception.evidence2d import Evidence2DStream, evidence_stream
from mica.perception.evidence3d import Evidence3DStream, build_evidence3d
from mica.perception.voxel_replay import ReplayWorld, region_around_events
from mica.validation.b0_gate import gate_checks

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW = os.path.join(_ROOT, "capture", "raw")


def _fixture_sessions():
    builds = [
        wall_row_build(), walk_only_build(), mixed_build(), sparse_build(),
        burst_place_build(), pen_build(),
        bridge_deck_build("x"), bridge_deck_build("z", session_id="synthetic-bridge-z"),
    ]
    return [(build.session_id, generate_session(build)) for build in builds]


def _scripted_sessions():
    sessions = []
    for goal in GOALS:
        for plan in plan_variants(goal, 4, 7):   # a corpus-sized sample of the plan space
            build, _ = build_from_plan(plan)
            sessions.append((build.session_id, generate_session(build)))
    return sessions


def _real_sessions():
    """Every raw capture next to a manifest that passes the B0 gate; [] elsewhere."""
    found = []
    for path in sorted(glob.glob(os.path.join(_RAW, "*.jsonl"))):
        if ".evidence" in os.path.basename(path):
            continue   # derived artifacts, not captures
        manifest = path.replace(".jsonl", ".manifest.json")
        if not os.path.exists(manifest):
            continue
        session = JsonlSource(path, manifest).load()
        if any(not ok for _, ok, _ in gate_checks(session, os.path.dirname(path))):
            continue
        found.append((os.path.basename(path), session))
    return found


def _batch(packets):
    return [evidence2d_to_dict(record) for record in evidence_stream(packets)]


def _streamed(packets):
    core = Evidence2DStream()
    out = []
    for packet in sorted(packets, key=lambda p: p.tick):
        out.extend(evidence2d_to_dict(record) for record in core.feed(packet))
    out.extend(evidence2d_to_dict(record) for record in core.finish())
    return out


def test_stream_matches_batch_on_every_sample_build():
    for name, session in _fixture_sessions():
        assert _streamed(session.packets) == _batch(session.packets), name


def test_stream_matches_batch_on_every_scripted_corpus_plan():
    for name, session in _scripted_sessions():
        assert _streamed(session.packets) == _batch(session.packets), name


def test_stream_matches_batch_on_real_captures():
    sessions = _real_sessions()
    if not sessions:
        pytest.skip("no gate-passing real captures on this machine")
    for name, session in sessions:
        assert _streamed(session.packets) == _batch(session.packets), name


def test_records_release_in_batch_order_with_no_future_knowledge():
    # The prefix property: at any moment mid-stream, everything released so far must be
    # exactly the front of the batch output — a record may never depend on a packet
    # that hasn't been fed yet, and never jump the queue.
    for name, session in _fixture_sessions():
        expected = _batch(session.packets)
        core = Evidence2DStream()
        seen = 0
        for packet in sorted(session.packets, key=lambda p: p.tick):
            for record in core.feed(packet):
                assert evidence2d_to_dict(record) == expected[seen], (name, seen)
                seen += 1
        for record in core.finish():
            assert evidence2d_to_dict(record) == expected[seen], (name, seen)
            seen += 1
        assert seen == len(expected), name


def _batch_b2(session):
    corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    world = ReplayWorld(region_around_events(session), {})
    return [evidence3d_to_dict(r) for r in build_evidence3d(world, corrections, events)]


def _live_style_b2(session):
    # The real live interleaving: every tick's events are harvested as the tick arrives,
    # and each scored record hits D2 the moment D1 releases it.
    d1 = Evidence2DStream()
    d2 = Evidence3DStream(ReplayWorld(region_around_events(session), {}))
    out = []

    def consume(records):
        for record in records:
            b2 = d2.on_correction(record)
            if b2 is not None:
                out.append(evidence3d_to_dict(b2))

    for packet in sorted(session.packets, key=lambda p: p.tick):
        d2.harvest(packet)
        consume(d1.feed(packet))
    consume(d1.finish())
    return out


def test_structure_stream_matches_batch_when_fed_live_style():
    for name, session in _fixture_sessions() + _scripted_sessions():
        assert _live_style_b2(session) == _batch_b2(session), name


def test_burst_run_defers_scored_until_the_run_closes():
    # 25 places on consecutive ticks (5..29): the scored PLACE record cannot leave
    # until the run closes at tick 30, because its event ids keep growing; the context
    # record due at tick 25 queues behind it so the batch order survives.
    packets = generate_session(burst_place_build()).packets
    core = Evidence2DStream()
    releases = {}
    for packet in sorted(packets, key=lambda p: p.tick):
        out = core.feed(packet)
        if out:
            releases[packet.tick] = out
    assert 25 not in releases                      # the due context was held, not released
    closing = releases[30]                         # first non-PLACE tick closes the run
    assert [(r.a_hat, r.scored) for r in closing[:2]] == [
        (MacroAction.PLACE, True), (MacroAction.PLACE, False)]
    assert closing[0].event_ids == tuple(range(25))   # every event of the burst, in order


def test_feed_refuses_non_increasing_ticks():
    packets = sorted(generate_session(wall_row_build()).packets, key=lambda p: p.tick)
    core = Evidence2DStream()
    core.feed(packets[0])
    core.feed(packets[1])
    with pytest.raises(ValueError):
        core.feed(packets[1])
