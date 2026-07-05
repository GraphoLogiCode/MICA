"""run_live.py live mode, end to end, without a game: a thread plays the mod (serving
wire bytes on a real localhost socket, with the session's disk copy + manifest +
snapshot laid out in a temp capture dir), and the runner does everything it would do
in production — discover the active session, seed structure from the snapshot, stream,
write all four proof logs + provenance + live_run.json, then run the disk gate. The
logs must equal the offline wrappers' output and the run must exit 0."""
import dataclasses
import json
import os
import socket
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import run_live  # noqa: E402

from mica.capture.jsonl_ingest import JsonlSource
from mica.capture.sample_builds import wall_row_build
from mica.capture.synthetic import generate_session
from mica.capture.wire_synthetic import frame_message, packet_message
from mica.contracts.serialize import packet_to_dict
from mica.perception.voxel_replay import Region, ReplayWorld

_SID = "livetest-0001"
_REGION = (-8, 56, -8, 12, 72, 8)
_FRAMED_TICKS = (2, 3, 4)          # non-square frames make the disk gate's frame checks pass


def _packet_dicts():
    session = generate_session(wall_row_build(session_id=_SID))
    dicts = []
    for packet in sorted(session.packets, key=lambda p: p.tick):
        raw = packet_to_dict(packet)
        if packet.tick in _FRAMED_TICKS:
            raw["client"]["pov_frame"] = {"path": f"frames/{packet.tick}.png", "width": 4, "height": 2}
        dicts.append(raw)
    return dicts, session.declared_event_count


def _lay_out_capture_dir(tmp_path, dicts, declared):
    jsonl = tmp_path / f"{_SID}.jsonl"
    jsonl.write_text("".join(json.dumps(d) + "\n" for d in dicts), encoding="utf-8")
    manifest = {
        "session_id": _SID, "session_start_ms": 0,
        "mc_version": "1.16.5", "mod_version": "fabric-b0-0.0.3",
        "event_schema_version": "1", "frame_every": 1, "frame_width_px": 4,
        "snapshot_quiet_ticks": 40, "snapshot_region": list(_REGION),
        "declared_event_count": declared,
    }
    (tmp_path / f"{_SID}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    snap_dir = tmp_path / _SID / "snapshots"
    snap_dir.mkdir(parents=True)
    region = Region(*_REGION)
    volume = ((region.x1 - region.x0 + 1) * (region.y1 - region.y0 + 1)
              * (region.z1 - region.z0 + 1))
    snapshot = {"tick": 0, "region": list(_REGION),
                "palette": ["minecraft:air"], "runs": [[volume, 0]]}   # empty flat world
    (snap_dir / "0.json").write_text(json.dumps(snapshot), encoding="utf-8")
    return str(jsonl)


def _serve_wire(dicts) -> int:
    """Speak the mod's wire protocol on an ephemeral port from a thread; return the port."""
    rgba = bytes(4 * 2 * 4)
    messages = []
    for raw in dicts:
        messages.append(packet_message(raw))
        if raw["client"]["pov_frame"] is not None:
            messages.append(frame_message(raw["tick"], rgba))
    wire = b"".join(messages)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def _run():
        conn, _ = server.accept()
        conn.sendall(wire)
        conn.close()               # EOF = clean session end
        server.close()

    threading.Thread(target=_run, daemon=True).start()
    return port


def test_live_mode_end_to_end_over_a_real_socket(tmp_path, monkeypatch):
    dicts, declared = _packet_dicts()
    jsonl = _lay_out_capture_dir(tmp_path, dicts, declared)
    port = _serve_wire(dicts)
    monkeypatch.setattr(run_live, "_RAW", str(tmp_path))
    monkeypatch.delenv("MICA_PIXELS", raising=False)   # this test is the symbolic run

    assert run_live._live("127.0.0.1", port) == 0

    # the agent bridge: the runner leaves a parseable status snapshot behind
    status = json.loads(open(tmp_path / "live_status.json", encoding="utf-8").read())
    assert status["session_id"] == _SID
    assert "top_goal" in status and "p_top_goal" in status
    assert "player_pos" in status and "current_behavior" in status
    assert "pixels" not in status                       # symbolic run: no pixel block

    base = jsonl[: -len(".jsonl")]
    report = json.loads(open(f"{base}.live_run.json", encoding="utf-8").read())
    assert report["ended"] == "eof"
    assert report["gate_on_disk_pass"] is True
    assert report["quarantined"] is False
    assert report["gate_live"]["clean"] is True
    assert report["ingest"]["gap_ticks"] == 0 and report["ingest"]["duplicates"] == 0
    assert report["records"]["corrections"] == report["records"]["scored"] > 0
    assert os.path.exists(f"{base}.d2_provenance.json")

    # the live-written logs must be exactly what the offline wrappers produce
    disk = JsonlSource(jsonl, jsonl.replace(".jsonl", ".manifest.json")).load()
    base_cells = run_live.load_snapshot(os.path.join(str(tmp_path), _SID, "snapshots", "0.json"))[2]
    expected = run_live._expected_lines(disk, ReplayWorld(Region(*_REGION), dict(base_cells)))
    for name in ("evidence2d", "evidence3d", "fused", "belief"):
        written = open(f"{base}.{name}.jsonl", encoding="utf-8").read().splitlines()
        assert written == expected[name], name


def test_live_mode_attaches_to_a_named_session(tmp_path, monkeypatch):
    # --session skips freshness discovery entirely: the rehearsal path, where the
    # attached session is a staged COPY whose mtime is old by construction.
    dicts, declared = _packet_dicts()
    jsonl = _lay_out_capture_dir(tmp_path, dicts, declared)
    port = _serve_wire(dicts)
    monkeypatch.setattr(run_live, "_RAW", str(tmp_path / "somewhere-else"))
    monkeypatch.delenv("MICA_PIXELS", raising=False)
    monkeypatch.delenv("MICA_H3D", raising=False)

    assert run_live._live("127.0.0.1", port, session_arg=jsonl) == 0
    assert os.path.exists(jsonl[: -len(".jsonl")] + ".live_run.json")


def test_feeder_stages_a_complete_clone_and_never_touches_the_source(tmp_path, monkeypatch):
    import replay_live_feed

    dicts, declared = _packet_dicts()
    jsonl = _lay_out_capture_dir(tmp_path, dicts, declared)
    before = {path: os.path.getmtime(path) for path in
              [jsonl, jsonl.replace(".jsonl", ".manifest.json")]}
    monkeypatch.setattr(replay_live_feed, "_REHEARSAL", str(tmp_path / "rehearsal"))

    staged, pending = replay_live_feed._stage(jsonl)
    stage_dir = os.path.dirname(staged)
    assert staged != jsonl and str(tmp_path / "rehearsal") in staged
    assert os.path.exists(os.path.join(stage_dir, f"{_SID}.manifest.json"))
    # only the BASE snapshot is staged up front — later ones drip in mid-feed, so
    # run_live seeds exactly the world a real attach would see
    assert os.path.exists(os.path.join(stage_dir, _SID, "snapshots", "0.json"))
    assert pending == []                      # this fixture has just the base snapshot
    for path, mtime in before.items():        # the raw source is never written
        assert os.path.getmtime(path) == mtime
    assert replay_live_feed._stage(jsonl)[0] == staged   # restages clean to the same path


def test_feeder_grow_mode_stages_an_empty_jsonl(tmp_path, monkeypatch):
    # --grow-jsonl rehearses the mod's parallel disk write: the staged session file
    # starts EMPTY and fills as the feed plays, so a mid-feed attach catches up from
    # a true prefix instead of reading the whole finished recording.
    import replay_live_feed

    dicts, declared = _packet_dicts()
    jsonl = _lay_out_capture_dir(tmp_path, dicts, declared)
    monkeypatch.setattr(replay_live_feed, "_REHEARSAL", str(tmp_path / "rehearsal"))

    staged, _ = replay_live_feed._stage(jsonl, grow=True)
    assert os.path.getsize(staged) == 0
    # the manifest still stages whole — run_live needs its region to seed
    manifest = json.loads(open(os.path.join(os.path.dirname(staged),
                                            f"{_SID}.manifest.json"), encoding="utf-8").read())
    assert manifest["snapshot_region"] == list(_REGION)


def test_feeder_grow_mode_stages_an_empty_jsonl(tmp_path, monkeypatch):
    # --grow-jsonl rehearses the mod's parallel disk write: the staged session file
    # starts EMPTY and fills as the feed plays, so a mid-feed attach catches up from
    # a true prefix instead of reading the whole finished recording.
    import replay_live_feed

    dicts, declared = _packet_dicts()
    jsonl = _lay_out_capture_dir(tmp_path, dicts, declared)
    monkeypatch.setattr(replay_live_feed, "_REHEARSAL", str(tmp_path / "rehearsal"))

    staged, _ = replay_live_feed._stage(jsonl, grow=True)
    assert os.path.getsize(staged) == 0
    # the manifest still stages whole — run_live needs its region to seed
    manifest = json.loads(open(os.path.join(os.path.dirname(staged),
                                            f"{_SID}.manifest.json"), encoding="utf-8").read())
    assert manifest["snapshot_region"] == list(_REGION)


def test_live_mode_attaches_to_a_named_session(tmp_path, monkeypatch):
    # --session skips freshness discovery entirely: the rehearsal path, where the
    # attached session is a staged COPY whose mtime is old by construction.
    dicts, declared = _packet_dicts()
    jsonl = _lay_out_capture_dir(tmp_path, dicts, declared)
    port = _serve_wire(dicts)
    monkeypatch.setattr(run_live, "_RAW", str(tmp_path / "somewhere-else"))
    monkeypatch.delenv("MICA_PIXELS", raising=False)
    monkeypatch.delenv("MICA_H3D", raising=False)

    assert run_live._live("127.0.0.1", port, session_arg=jsonl) == 0
    assert os.path.exists(jsonl[: -len(".jsonl")] + ".live_run.json")


def test_feeder_stages_a_complete_clone_and_never_touches_the_source(tmp_path, monkeypatch):
    import replay_live_feed

    dicts, declared = _packet_dicts()
    jsonl = _lay_out_capture_dir(tmp_path, dicts, declared)
    before = {path: os.path.getmtime(path) for path in
              [jsonl, jsonl.replace(".jsonl", ".manifest.json")]}
    monkeypatch.setattr(replay_live_feed, "_REHEARSAL", str(tmp_path / "rehearsal"))

    staged, pending = replay_live_feed._stage(jsonl)
    stage_dir = os.path.dirname(staged)
    assert staged != jsonl and str(tmp_path / "rehearsal") in staged
    assert os.path.exists(os.path.join(stage_dir, f"{_SID}.manifest.json"))
    # only the BASE snapshot is staged up front — later ones drip in mid-feed, so
    # run_live seeds exactly the world a real attach would see
    assert os.path.exists(os.path.join(stage_dir, _SID, "snapshots", "0.json"))
    assert pending == []                      # this fixture has just the base snapshot
    for path, mtime in before.items():        # the raw source is never written
        assert os.path.getmtime(path) == mtime
    assert replay_live_feed._stage(jsonl)[0] == staged   # restages clean to the same path


def test_live_mode_attaches_to_a_named_session(tmp_path, monkeypatch):
    # --session skips freshness discovery entirely: the rehearsal path, where the
    # attached session is a staged COPY whose mtime is old by construction.
    dicts, declared = _packet_dicts()
    jsonl = _lay_out_capture_dir(tmp_path, dicts, declared)
    port = _serve_wire(dicts)
    monkeypatch.setattr(run_live, "_RAW", str(tmp_path / "somewhere-else"))
    monkeypatch.delenv("MICA_PIXELS", raising=False)
    monkeypatch.delenv("MICA_H3D", raising=False)

    assert run_live._live("127.0.0.1", port, session_arg=jsonl) == 0
    assert os.path.exists(jsonl[: -len(".jsonl")] + ".live_run.json")


def test_feeder_stages_a_complete_clone_and_never_touches_the_source(tmp_path, monkeypatch):
    import replay_live_feed

    dicts, declared = _packet_dicts()
    jsonl = _lay_out_capture_dir(tmp_path, dicts, declared)
    before = {path: os.path.getmtime(path) for path in
              [jsonl, jsonl.replace(".jsonl", ".manifest.json")]}
    monkeypatch.setattr(replay_live_feed, "_REHEARSAL", str(tmp_path / "rehearsal"))

    staged, pending = replay_live_feed._stage(jsonl)
    stage_dir = os.path.dirname(staged)
    assert staged != jsonl and str(tmp_path / "rehearsal") in staged
    assert os.path.exists(os.path.join(stage_dir, f"{_SID}.manifest.json"))
    # only the BASE snapshot is staged up front — later ones drip in mid-feed, so
    # run_live seeds exactly the world a real attach would see
    assert os.path.exists(os.path.join(stage_dir, _SID, "snapshots", "0.json"))
    assert pending == []                      # this fixture has just the base snapshot
    for path, mtime in before.items():        # the raw source is never written
        assert os.path.getmtime(path) == mtime
    assert replay_live_feed._stage(jsonl)[0] == staged   # restages clean to the same path


def test_reseed_structure_recovers_from_a_moved_provisional_region():
    # The 193059 false quarantine: mod 0.0.4's region is provisional until the first
    # block event, so run_live can seed on a region the mod then abandons. Re-seeding
    # BEFORE any event is consumed must leave the pipeline exactly as if it had seeded
    # right the first time: no escapes, the build visible.
    import io

    from mica.intent.tracker import TrackerParams
    from mica.live_pipeline import LivePipeline, PipelineConfig, RecordSinks
    from mica.perception.evidence3d import Evidence3DStream
    from mica.perception.voxel_replay import Region, ReplayWorld, SnapshotMonitor

    session = generate_session(wall_row_build(session_id=_SID))
    packets = sorted(session.packets, key=lambda p: p.tick)
    first_event_tick = min(p.tick for p in packets if p.server.block_events)

    provisional = Region(1000, 0, 1000, 1020, 20, 1020)      # the abandoned anchor
    final = Region(*_REGION)                                  # where the build happens
    pipeline = LivePipeline(PipelineConfig(
        params=TrackerParams(),
        sinks=RecordSinks(evidence2d=io.StringIO(), evidence3d=io.StringIO(),
                          fused=io.StringIO(), belief=io.StringIO()),
        d2=Evidence3DStream(ReplayWorld(provisional, {})),
        monitor=SnapshotMonitor(provisional, {})))

    for packet in packets:
        if packet.tick == first_event_tick:                  # the region froze; we notice
            assert pipeline.events_consumed == 0
            pipeline.reseed_structure(Evidence3DStream(ReplayWorld(final, {})),
                                      SnapshotMonitor(final, {}))
        pipeline.on_moment(packet, None)
    summary = pipeline.finish()

    assert summary["crop_escapes"] == 0                      # nothing escaped the frame
    assert summary["quarantined"] is False
    assert summary["consumed_event_ids"] == session.declared_event_count
    # and reseeding AFTER evidence exists must refuse loudly
    try:
        pipeline.reseed_structure(Evidence3DStream(ReplayWorld(final, {})),
                                  SnapshotMonitor(final, {}))
        assert False, "reseed after consumption must raise"
    except AssertionError as error:
        assert "consumed" in str(error)


def test_live_mode_survives_a_dropped_moment(tmp_path, monkeypatch):
    # the mod drops one moment under backpressure: the runner keeps going, counts the
    # gap, stays un-quarantined (no events lost), and says the run is not proof-grade
    dicts, declared = _packet_dicts()
    jsonl = _lay_out_capture_dir(tmp_path, dicts, declared)
    dropped = [d for d in dicts if d["tick"] != 3]     # tick 3 carries no block event
    port = _serve_wire(dropped)
    monkeypatch.setattr(run_live, "_RAW", str(tmp_path))

    assert run_live._live("127.0.0.1", port) == 0      # session still usable, disk complete
    report = json.loads(open(jsonl[: -len(".jsonl")] + ".live_run.json", encoding="utf-8").read())
    assert report["gate_live"]["tick_gaps"]["missing_ticks"] == 1
    assert report["gate_live"]["clean"] is False       # honest: not proof-grade
    assert report["gate_on_disk_pass"] is True         # the disk copy is complete
    assert report["quarantined"] is False
