import dataclasses

from mica.capture.sample_builds import mixed_build, wall_row_build, walk_only_build
from mica.capture.synthetic import generate_session
from mica.validation.sync_report import build_sync_report


def _strip_event(session, event_id):
    def strip(packet):
        if any(event.event_id == event_id for event in packet.server.block_events):
            return dataclasses.replace(packet, server=dataclasses.replace(packet.server, block_events=()))
        return packet

    return dataclasses.replace(session, packets=tuple(strip(packet) for packet in session.packets))


def test_clean_session_passes():
    report = build_sync_report(generate_session(wall_row_build()))
    assert report.passed
    assert report.ticks.contiguous
    assert report.alignment.within_bound
    assert report.block_events.complete
    assert report.actions_have_prior_moment


def test_mixed_session_with_break_and_multi_event_passes():
    report = build_sync_report(generate_session(mixed_build()))
    assert report.passed
    assert report.block_events.complete


def test_missing_tick_breaks_continuity():
    session = generate_session(wall_row_build())
    # Remove an empty moment (no block change) so only the 'complete moments' check fails.
    kept = tuple(packet for packet in session.packets if packet.tick != 12)
    report = build_sync_report(dataclasses.replace(session, packets=kept))
    assert not report.ticks.contiguous
    assert not report.passed


def test_dropped_middle_block_change_is_detected():
    report = build_sync_report(_strip_event(generate_session(wall_row_build()), event_id=2))
    assert report.block_events.dropped == 1
    assert not report.block_events.complete
    assert not report.passed


def test_dropped_last_block_change_is_detected():
    # Losing the LAST change used to be invisible; the declared total now catches it.
    report = build_sync_report(_strip_event(generate_session(wall_row_build()), event_id=4))
    assert report.block_events.dropped == 1
    assert not report.passed


def test_player_clock_offset_beyond_bound_fails():
    build = dataclasses.replace(wall_row_build(), alignment_jitter_ms=100.0)
    report = build_sync_report(generate_session(build))
    assert not report.alignment.within_bound
    assert not report.passed


def test_slow_server_does_not_look_like_a_sync_problem():
    # A slow server makes both clocks late together; that must NOT fail the clock check.
    build = dataclasses.replace(wall_row_build(), server_lag_ms=200.0)
    report = build_sync_report(generate_session(build))
    assert report.alignment.within_bound
    assert report.passed


def test_session_with_no_building_passes():
    report = build_sync_report(generate_session(walk_only_build()))
    assert report.block_events.count == 0
    assert report.passed


def test_empty_recording_fails_without_crashing():
    session = generate_session(wall_row_build())
    empty = dataclasses.replace(session, packets=(), declared_event_count=0)
    report = build_sync_report(empty)
    assert not report.passed
