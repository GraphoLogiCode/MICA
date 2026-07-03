"""Verifies the B0 gate correctly accepts a good capture and rejects bad ones, so its
PASS on a real mod capture can be trusted."""
import dataclasses

from mica.capture.sample_builds import walk_only_build, wall_row_build
from mica.capture.synthetic import generate_session
from mica.contracts.b0 import FrameRef
from mica.validation.b0_gate import gate_checks, gate_passes


def _with_frames(session, width, height):
    frame = FrameRef(path="f.png", width=width, height=height)
    packets = tuple(
        dataclasses.replace(packet, client=dataclasses.replace(packet.client, pov_frame=frame))
        for packet in session.packets
    )
    return dataclasses.replace(session, packets=packets)


def _failed(session):
    return {name for name, ok, _ in gate_checks(session) if not ok}


def test_good_capture_passes():
    # a clean build session (block events) with undistorted (non-square) frames
    session = _with_frames(generate_session(wall_row_build()), 256, 144)
    assert gate_passes(session)


def test_missing_frames_fail():
    session = generate_session(wall_row_build())   # synthetic carries no pov_frame
    assert "frames present" in _failed(session)


def test_square_frames_fail_as_distorted():
    session = _with_frames(generate_session(wall_row_build()), 128, 128)
    assert "frames undistorted (non-square)" in _failed(session)


def test_walk_only_fails_for_no_block_events():
    session = _with_frames(generate_session(walk_only_build()), 256, 144)
    assert "block events exercised" in _failed(session)


def test_provisional_count_is_hard_rejected():
    # A crashed recording may have lost its end with no way to tell, so a passing
    # gate on it would certify completeness it cannot check.
    session = _with_frames(generate_session(wall_row_build()), 256, 144)
    crashed = dataclasses.replace(session, declared_is_provisional=True)
    assert gate_passes(session)
    assert "manifest finalized (clean stop)" in _failed(crashed)
    assert not gate_passes(crashed)


def test_snapshots_required_only_when_declared(tmp_path):
    session = _with_frames(generate_session(wall_row_build()), 256, 144)
    declared = dataclasses.replace(
        session, manifest=dataclasses.replace(session.manifest, snapshot_quiet_ticks=40)
    )
    # declares snapshots but has none on disk -> the capture is not D2-replayable, fail
    failing = {name for name, ok, _ in gate_checks(declared, str(tmp_path)) if not ok}
    assert "region snapshots (D2)" in failing
    # the initial snapshot appears -> pass
    snap_dir = tmp_path / declared.manifest.session_id / "snapshots"
    snap_dir.mkdir(parents=True)
    (snap_dir / "0.json").write_text("{}", encoding="utf-8")
    assert gate_passes(declared, str(tmp_path))
    # an older capture that never declared snapshots must not fail retroactively
    assert gate_passes(session, str(tmp_path))
