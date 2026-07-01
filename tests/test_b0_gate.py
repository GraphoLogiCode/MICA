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
