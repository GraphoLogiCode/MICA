from mica.capture.sample_builds import mixed_build, wall_row_build
from mica.capture.synthetic import generate_session
from mica.contracts.b0 import BlockOp
from mica.validation.sync_report import _collect_block_events


def test_wall_row_ticks_are_contiguous_from_zero():
    session = generate_session(wall_row_build())
    ticks = [packet.tick for packet in session.packets]
    assert ticks == list(range(ticks[-1] + 1))


def test_wall_row_emits_five_events_with_monotonic_ids():
    session = generate_session(wall_row_build())
    events = _collect_block_events(session.packets)
    assert [event.event_id for event in events] == [0, 1, 2, 3, 4]
    assert session.declared_event_count == 5


def test_wall_row_places_one_block_every_five_ticks():
    session = generate_session(wall_row_build())
    action_ticks = sorted(
        packet.tick for packet in session.packets if packet.server.block_events
    )
    assert action_ticks == [5, 10, 15, 20, 25]


def test_mixed_build_has_a_break_and_a_multi_change_moment():
    session = generate_session(mixed_build())
    events = _collect_block_events(session.packets)
    assert [event.event_id for event in events] == [0, 1, 2, 3]
    assert any(event.op is BlockOp.BREAK for event in events)
    moments_with_two_changes = [p for p in session.packets if len(p.server.block_events) == 2]
    assert len(moments_with_two_changes) == 1
