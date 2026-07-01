import dataclasses

from mica.capture.sample_builds import wall_row_build
from mica.capture.synthetic import generate_session
from mica.contracts.b0 import FrameRef
from mica.validation.coverage import evaluate_coverage, gating_gaps


def _by_reader(results):
    return {result.reader: result for result in results}


def test_synthetic_satisfies_d2_and_d1_symbolic():
    session = generate_session(wall_row_build())
    results = _by_reader(evaluate_coverage(session.packets))
    assert results["D2 3D structure stream"].satisfied
    assert results["D1 2D symbolic half"].satisfied


def test_synthetic_reports_pixel_half_as_the_open_gap():
    session = generate_session(wall_row_build())
    results = _by_reader(evaluate_coverage(session.packets))
    pixel = results["D1 2D pixel half (VPT/MineCLIP)"]
    assert not pixel.satisfied
    assert pixel.missing == ("pov_frame",)
    # Missing the screen image is fine here — the recording still has everything it must carry.
    assert gating_gaps(session.packets) == ()


def test_frames_close_the_pixel_gap():
    session = generate_session(wall_row_build())
    frame = FrameRef(path="frame.png", width=128, height=128)
    packets = tuple(
        dataclasses.replace(packet, client=dataclasses.replace(packet.client, pov_frame=frame))
        for packet in session.packets
    )
    results = _by_reader(evaluate_coverage(packets))
    assert results["D1 2D pixel half (VPT/MineCLIP)"].satisfied


def test_pov_frame_present_only_sometimes_still_satisfies():
    # Frames are throttled and skipped in menus, so most ticks have none. The pixel
    # half counts as covered as long as frames appear at least sometimes.
    session = generate_session(wall_row_build())
    frame = FrameRef(path="f.png", width=128, height=128)
    packets = tuple(
        dataclasses.replace(
            packet,
            client=dataclasses.replace(packet.client, pov_frame=(frame if packet.tick % 2 == 0 else None)),
        )
        for packet in session.packets
    )
    assert _by_reader(evaluate_coverage(packets))["D1 2D pixel half (VPT/MineCLIP)"].satisfied


def test_crosshair_flagged_only_when_never_present():
    # A capture that has the crosshair on some ticks (the mod / synthetic) is covered;
    # a server-only capture that never has it (the bot) is flagged.
    session = generate_session(wall_row_build())
    assert _by_reader(evaluate_coverage(session.packets))["D1 2D symbolic half"].satisfied

    blanked = tuple(
        dataclasses.replace(packet, client=dataclasses.replace(packet.client, crosshair_target=None))
        for packet in session.packets
    )
    symbolic = _by_reader(evaluate_coverage(blanked))["D1 2D symbolic half"]
    assert "crosshair" in symbolic.missing
