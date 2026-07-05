"""Late-attach catch-up: a pipeline that starts listening MID-session must still hold
the whole build. The real failure this guards against (2026-07-04): run_live was
relaunched partway into a session, seeded its world from a snapshot that already
CONTAINED the player's first 110 blocks, and the live structure read showed 22 of
100 built cells for the rest of the night. The fix replays the on-disk recording up
to the first live moment — human events into the feature world (A7), every event
into the monitor's shadow world — with each event applied exactly once."""
import io
import json
import os
import socket
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import run_live  # noqa: E402

from mica.capture.sample_builds import assisted_build, walk_only_build
from mica.capture.synthetic import generate_session
from mica.capture.wire_synthetic import packet_message
from mica.contracts.serialize import packet_to_dict
from mica.intent.tracker import TrackerParams
from mica.live_pipeline import LivePipeline, PipelineConfig, RecordSinks
from mica.perception.evidence3d import Evidence3DStream
from mica.perception.voxel_replay import Region, ReplayWorld, SnapshotMonitor

_SID = "latetest-0001"
_REGION = (-8, 56, -8, 12, 72, 8)
_PLANKS = "minecraft:oak_planks"
_HUMAN_CELLS = {(i, 64, 0): _PLANKS for i in range(5)}          # what the human builds
_AGENT_CELLS = {(x, 64, 3): "minecraft:stone" for x in (0, 1, 2)}  # what the agent places


def _session():
    return generate_session(assisted_build(session_id=_SID))


def _events_with_ticks(session):
    return [(p.tick, e) for p in sorted(session.packets, key=lambda p: p.tick)
            for e in p.server.block_events]


def _snapshot_runs(region, cells):
    """Palette + run-length encode a sparse cell map, scan order y -> z -> x —
    the exact format the mod writes, so load_snapshot can decode it back."""
    x0, y0, z0, x1, y1, z1 = region
    palette = ["minecraft:air"] + sorted(set(cells.values()))
    index_of = {block: i for i, block in enumerate(palette)}
    runs = []
    for y in range(y0, y1 + 1):
        for z in range(z0, z1 + 1):
            for x in range(x0, x1 + 1):
                index = index_of.get(cells.get((x, y, z), "minecraft:air"), 0)
                if runs and runs[-1][1] == index:
                    runs[-1][0] += 1
                else:
                    runs.append([1, index])
    return palette, runs


def _write_snapshot(snap_dir, tick, cells):
    palette, runs = _snapshot_runs(_REGION, cells)
    path = os.path.join(str(snap_dir), f"{tick}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"tick": tick, "region": list(_REGION), "palette": palette, "runs": runs},
                  handle)


def _lay_out(tmp_path, session, snapshots=((0, {}),), framed_ticks=()):
    """A finished-so-far capture dir: jsonl + manifest + snapshot files."""
    jsonl = tmp_path / f"{_SID}.jsonl"
    dicts = [packet_to_dict(p) for p in sorted(session.packets, key=lambda p: p.tick)]
    for raw in dicts:
        if raw["tick"] in framed_ticks:    # non-square frames keep the disk gate happy
            raw["client"]["pov_frame"] = {"path": f"frames/{raw['tick']}.png",
                                          "width": 4, "height": 2}
    jsonl.write_text("".join(json.dumps(d) + "\n" for d in dicts), encoding="utf-8")
    manifest = {
        "session_id": _SID, "session_start_ms": 0,
        "mc_version": "1.16.5", "mod_version": "fabric-b0-0.0.4",
        "event_schema_version": "1", "frame_every": 1, "frame_width_px": 4,
        "snapshot_quiet_ticks": 40, "snapshot_region": list(_REGION),
        "declared_event_count": session.declared_event_count,
    }
    (tmp_path / f"{_SID}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    snap_dir = tmp_path / _SID / "snapshots"
    snap_dir.mkdir(parents=True)
    for tick, cells in snapshots:
        _write_snapshot(snap_dir, tick, cells)
    return str(jsonl)


def test_seed_history_applies_human_events_only():
    session = _session()
    d2 = Evidence3DStream(ReplayWorld(Region(*_REGION), {}))
    applied = d2.seed_history(event for _, event in _events_with_ticks(session))
    assert applied == 5                                # the agent's 3 were dropped (A7)
    assert d2._world.built() == _HUMAN_CELLS
    assert d2.pending_event_ids == ()                  # seeded ids never enter the pool
    assert d2._cached is None                          # first correction computes fresh


def test_seeded_events_are_never_double_applied():
    session = _session()
    packets = sorted(session.packets, key=lambda p: p.tick)
    d2 = Evidence3DStream(ReplayWorld(Region(*_REGION), {}))
    d2.seed_history(event for _, event in _events_with_ticks(session))
    monitor = SnapshotMonitor(Region(*_REGION), {})
    monitor.seed_history(_events_with_ticks(session), [])

    pipeline = LivePipeline(PipelineConfig(
        params=TrackerParams(),
        sinks=RecordSinks(evidence2d=io.StringIO(), evidence3d=io.StringIO(),
                          fused=io.StringIO(), belief=io.StringIO()),
        d2=d2, monitor=monitor))
    # Reseeding before anything was consumed must still be allowed: catch-up
    # seeding is not consumption, so the guard's assertion may not trip here.
    assert pipeline.events_consumed == 0
    pipeline.reseed_structure(d2, monitor)

    # The wire redelivers the ENTIRE session (a full-overlap rehearsal): every
    # event id is already seeded, so nothing may apply twice anywhere.
    for packet in packets:
        pipeline.on_moment(packet, None)
    summary = pipeline.finish()
    assert d2._world.built() == _HUMAN_CELLS           # unchanged: no double placement
    assert monitor.world.built() == {**_HUMAN_CELLS, **_AGENT_CELLS}
    assert summary["crop_escapes"] == 0 and summary["quarantined"] is False
    assert summary["consumed_event_ids"] == 5          # D1 still consumed them (behavior)
    assert summary["unconsumed_event_ids"] == []


def test_monitor_seed_history_interleaves_snapshots():
    session = _session()
    events = _events_with_ticks(session)
    # the world as it truly stood at tick 12: human blocks @5,@10 plus agent's @7
    mid_cells = {(0, 64, 0): _PLANKS, (1, 64, 0): _PLANKS, (0, 64, 3): "minecraft:stone"}

    clean = SnapshotMonitor(Region(*_REGION), {})
    clean.seed_history(events, [(12, dict(mid_cells), Region(*_REGION))])
    assert clean.divergence_count == 0 and not clean.quarantined
    assert clean.world.built() == {**_HUMAN_CELLS, **_AGENT_CELLS}

    # the same snapshot MISSING a block the replay placed = a phantom, class c
    broken = SnapshotMonitor(Region(*_REGION), {})
    wrong = dict(mid_cells)
    del wrong[(0, 64, 0)]
    broken.seed_history(events, [(12, wrong, Region(*_REGION))])
    assert broken.class_counts["c"] > 0 and broken.quarantined

    # redelivering a seeded event afterwards changes nothing at all
    before = (dict(clean.world.changed), clean.divergence_count, len(clean.world.escaped))
    for packet in sorted(session.packets, key=lambda p: p.tick):
        clean.apply_packet(packet)
    assert (dict(clean.world.changed), clean.divergence_count,
            len(clean.world.escaped)) == before


def test_seed_structure_late_attach_reconstructs_full_build(tmp_path):
    session = _session()
    mid_cells = {(0, 64, 0): _PLANKS, (1, 64, 0): _PLANKS, (0, 64, 3): "minecraft:stone"}
    jsonl = _lay_out(tmp_path, session, snapshots=((0, {}), (12, mid_cells)))

    d2, monitor, seed_tick = run_live._seed_structure(jsonl)
    assert seed_tick == 0                              # events on disk -> pre-build base
    last_tick = max(p.tick for p in session.packets)
    run_live._catch_up_structure(d2, monitor, jsonl, _SID, cutoff_tick=last_tick + 1)
    assert d2._world.built() == _HUMAN_CELLS           # the whole build, human only
    assert monitor.world.built() == {**_HUMAN_CELLS, **_AGENT_CELLS}
    assert monitor.divergence_count == 0 and not monitor.quarantined


def test_seed_structure_without_disk_events_matches_today(tmp_path):
    # no events on disk = the normal rig flow: base from the NEWEST snapshot,
    # nothing caught up — byte-identical to the pre-catch-up behavior
    session = generate_session(walk_only_build(session_id=_SID))
    terrain = {(0, 60, 0): "minecraft:stone"}
    jsonl = _lay_out(tmp_path, session, snapshots=((0, {}), (40, terrain)))

    d2, monitor, seed_tick = run_live._seed_structure(jsonl)
    assert seed_tick == 40                             # the newest provisional base
    assert d2._world.base == terrain and d2._world.changed == {}
    run_live._catch_up_structure(d2, monitor, jsonl, _SID, cutoff_tick=10 ** 9)
    assert d2._world.changed == {}                     # a no-op with nothing recorded


def test_live_late_attach_end_to_end_over_a_socket(tmp_path, monkeypatch):
    # the full runner against a wire that serves only the TAIL of the session
    # while the complete recording sits on disk — tonight's recovery, in a test
    session = _session()
    jsonl = _lay_out(tmp_path, session, framed_ticks=(2, 3, 4))
    cut = 12                                           # attach here: after 3 of 8 events
    tail = [packet_to_dict(p) for p in sorted(session.packets, key=lambda p: p.tick)
            if p.tick >= cut]
    wire = b"".join(packet_message(raw) for raw in tail)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def _run():
        conn, _ = server.accept()
        conn.sendall(wire)
        conn.close()
        server.close()

    threading.Thread(target=_run, daemon=True).start()
    monkeypatch.setattr(run_live, "_RAW", str(tmp_path))
    monkeypatch.delenv("MICA_PIXELS", raising=False)
    monkeypatch.delenv("MICA_H3D", raising=False)

    assert run_live._live("127.0.0.1", port, session_arg=jsonl) == 0

    status = json.loads(open(tmp_path / "live_status.json", encoding="utf-8").read())
    assert status["built_count"] == 5                  # the WHOLE build, not the tail's 3
    assert {tuple(c) for c in status["built_cells"]} == set(_HUMAN_CELLS)
    report = json.loads(open(jsonl[: -len(".jsonl")] + ".live_run.json",
                             encoding="utf-8").read())
    assert report["ended"] == "eof"
    assert report["quarantined"] is False
    assert report["gate_on_disk_pass"] is True


def test_live_status_built_count_is_uncapped(tmp_path):
    from mica.contracts.b0 import BlockEvent, BlockOp, BlockPos

    region = Region(0, 56, 0, 29, 72, 29)
    world = ReplayWorld(region, {})
    cells = [(x, 64, z) for x in range(25) for z in range(25)][:600]
    for eid, (x, y, z) in enumerate(cells):
        world.apply(BlockEvent(event_id=eid, pos=BlockPos(x, y, z),
                               block_type=_PLANKS, op=BlockOp.PLACE, actor="tester"))
    d2 = Evidence3DStream(world)
    pipeline = LivePipeline(PipelineConfig(params=TrackerParams(), sinks=RecordSinks(),
                                           d2=d2))
    path = str(tmp_path / "live_status.json")
    run_live._write_live_status(path, _SID, pipeline, None, d2=d2)
    status = json.loads(open(path, encoding="utf-8").read())
    assert status["built_count"] == 600                # the truth, uncapped
    assert len(status["built_cells"]) == 512           # the display list stays bounded
