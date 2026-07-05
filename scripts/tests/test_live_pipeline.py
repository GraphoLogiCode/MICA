"""The live pipeline end to end, without a game: synthetic session -> mod-shaped wire
bytes -> socket reader -> reorderer -> LivePipeline. The four proof logs it writes
must be byte-identical to what the offline wrappers produce for the same session —
including the belief at every correction — and that must survive out-of-order arrival.
Also pins the failure modes: a join mismatch raises, and a mid-stream stop leaves
valid, flushed, prefix-exact logs (crash-safety)."""
import dataclasses
import io
import json

import pytest

from mica.capture.live_ingest import ordered_packets
from mica.capture.live_stream import read_moments
from mica.capture.sample_builds import burst_place_build, pen_build, sparse_build, wall_row_build
from mica.capture.scripted_goals import build_from_plan, plan_variants
from mica.capture.synthetic import generate_session
from mica.capture.wire_synthetic import session_wire
from mica.contracts.b1 import GOALS
from mica.contracts.b3 import fuse
from mica.contracts.serialize import (
    belief_to_dict, evidence2d_to_dict, evidence3d_to_dict, fused_to_dict,
)
from mica.intent.heads_v0 import likelihood
from mica.intent.tracker import TrackerParams, correct, predict, uniform_belief
from mica.live_pipeline import LivePipeline, PipelineConfig, RecordSinks
from mica.perception.evidence2d import evidence_stream
from mica.perception.evidence3d import Evidence3DStream, build_evidence3d
from mica.perception.voxel_replay import ReplayWorld, region_around_events


def _sessions():
    builds = [wall_row_build(), sparse_build(), burst_place_build(), pen_build(),
              build_from_plan(plan_variants(GOALS[0], 1, 7)[0])[0]]
    return [(b.session_id, generate_session(b)) for b in builds]


def _expected_logs(session):
    """What the offline wrappers write for this session, as serialized lines."""
    params = TrackerParams()
    b1 = list(evidence_stream(session.packets))
    scored = [r for r in b1 if r.scored]
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    b2 = build_evidence3d(ReplayWorld(region_around_events(session), {}), tuple(scored), events)

    belief, prev = uniform_belief(), 0
    fused_lines, belief_lines = [], []
    b2_iter = iter(b2)
    for record in b1:
        if record.scored:
            fused = fuse(record, next(b2_iter))
            dt = max(fused.tick - prev, 1) / 20.0
            prev = fused.tick
            belief = predict(belief, dt, params)
            belief, norm = correct(belief, likelihood(fused, fused.a_hat), params)
            fused_lines.append(json.dumps(fused_to_dict(fused)))
            belief_lines.append(json.dumps(belief_to_dict(fused.tick, "correction", belief, norm)))
        else:
            tick = record.tick_range[1]
            drifted = predict(belief, max(tick - prev, 1) / 20.0, params)
            belief_lines.append(json.dumps(belief_to_dict(tick, "drift", drifted)))
    return {
        "evidence2d": [json.dumps(evidence2d_to_dict(r)) for r in b1],
        "evidence3d": [json.dumps(evidence3d_to_dict(r)) for r in b2],
        "fused": fused_lines,
        "belief": belief_lines,
    }


def _fresh_pipeline(session):
    sinks = RecordSinks(evidence2d=io.StringIO(), evidence3d=io.StringIO(),
                        fused=io.StringIO(), belief=io.StringIO())
    d2 = Evidence3DStream(ReplayWorld(region_around_events(session), {}))
    return LivePipeline(PipelineConfig(params=TrackerParams(), sinks=sinks, d2=d2)), sinks


def _sink_lines(sinks):
    return {name: getattr(sinks, name).getvalue().splitlines()
            for name in ("evidence2d", "evidence3d", "fused", "belief")}


def _run_wire(session, **wire_kwargs):
    pipeline, sinks = _fresh_pipeline(session)
    wire = session_wire(session, **wire_kwargs)
    for packet, frame in ordered_packets(read_moments(io.BytesIO(wire))):
        pipeline.on_moment(packet, frame)
    pipeline.finish()
    return pipeline, sinks


def test_wire_fed_pipeline_reproduces_the_offline_logs_byte_for_byte():
    for name, session in _sessions():
        _, sinks = _run_wire(session)
        assert _sink_lines(sinks) == _expected_logs(session), name


def test_out_of_order_arrival_changes_nothing():
    # frames delayed several messages -> frameless ticks overtake framed ones on the
    # wire -> the reorderer restores tick order -> every log still byte-identical
    for name, session in _sessions():
        ticks = sorted(p.tick for p in session.packets)
        _, sinks = _run_wire(session, framed_ticks=ticks[1:6], frame_delay=4)
        assert _sink_lines(sinks) == _expected_logs(session), name


def test_join_mismatch_raises_instead_of_logging_garbage():
    session = generate_session(wall_row_build())

    class _WrongTick:
        """A broken D2 that stamps every record one tick late."""

        def __init__(self, inner):
            self._inner = inner
            self.pending_event_ids = ()

        def harvest(self, packet):
            self._inner.harvest(packet)

        def on_correction(self, record):
            b2 = self._inner.on_correction(record)
            return dataclasses.replace(b2, tick=b2.tick + 1)

    sinks = RecordSinks(evidence2d=io.StringIO(), evidence3d=io.StringIO(),
                        fused=io.StringIO(), belief=io.StringIO())
    broken = _WrongTick(Evidence3DStream(ReplayWorld(region_around_events(session), {})))
    pipeline = LivePipeline(PipelineConfig(params=TrackerParams(), sinks=sinks, d2=broken))
    with pytest.raises(ValueError, match="join mismatch"):
        for packet, frame in ordered_packets(read_moments(io.BytesIO(session_wire(session)))):
            pipeline.on_moment(packet, frame)


def test_mid_stream_stop_leaves_valid_prefix_logs():
    # kill the loop halfway with no finish(): every sink must already hold complete,
    # parseable lines forming an exact prefix of the full run — per-record flushing
    # is what makes a live crash lose at most the record in flight
    session = generate_session(sparse_build())
    expected = _expected_logs(session)
    pipeline, sinks = _fresh_pipeline(session)
    moments = list(ordered_packets(read_moments(io.BytesIO(session_wire(session)))))
    for packet, frame in moments[: len(moments) // 2]:
        pipeline.on_moment(packet, frame)
    lines = _sink_lines(sinks)
    for name in lines:
        assert lines[name] == expected[name][: len(lines[name])], name
        for line in lines[name]:
            json.loads(line)   # every flushed line is complete JSON
    assert lines["evidence2d"]   # the stop was late enough that something was written


def test_d1_only_mode_writes_behavior_log_and_nothing_else():
    # no base snapshot yet -> no D2, no fusion, no belief; the B1 log still flows
    session = generate_session(wall_row_build())
    sinks = RecordSinks(evidence2d=io.StringIO(), evidence3d=io.StringIO(),
                        fused=io.StringIO(), belief=io.StringIO())
    pipeline = LivePipeline(PipelineConfig(params=TrackerParams(), sinks=sinks, d2=None))
    for packet, frame in ordered_packets(read_moments(io.BytesIO(session_wire(session)))):
        pipeline.on_moment(packet, frame)
    summary = pipeline.finish()
    assert sinks.evidence2d.getvalue().splitlines() == _expected_logs(session)["evidence2d"]
    assert sinks.evidence3d.getvalue() == ""
    assert sinks.fused.getvalue() == ""
    assert sinks.belief.getvalue() == ""
    assert summary["records"]["corrections"] == 0
