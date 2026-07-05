"""D1 (2D behavior perception) on synthetic builds: the macro-action segmentation is
correct, every scored record's evidence ends strictly before the action it scores
(snapshot rule), each block event is consumed exactly once, and long runs emit
unscored ~1 Hz context records."""
from mica.capture.sample_builds import mixed_build, sparse_build, wall_row_build
from mica.capture.synthetic import generate_session
from mica.contracts.b1 import MacroAction
from mica.perception.evidence2d import evidence_stream


def _records(build):
    return list(evidence_stream(generate_session(build).packets))


def _event_tick(session, event_id):
    for packet in session.packets:
        for event in packet.server.block_events:
            if event.event_id == event_id:
                return packet.tick
    raise KeyError(event_id)


def test_place_actions_match_the_scripted_build():
    # five blocks placed (ticks 5,10,15,20,25) -> five PLACE macro-actions, each carrying
    # exactly the one event it consumed, ids 0..4.
    places = [r for r in _records(wall_row_build()) if r.a_hat is MacroAction.PLACE]
    assert len(places) == 5
    assert all(len(r.event_ids) == 1 for r in places)
    assert sorted(eid for r in places for eid in r.event_ids) == [0, 1, 2, 3, 4]


def test_break_and_multi_place_recognized():
    records = _records(mixed_build())
    assert any(r.a_hat is MacroAction.BREAK for r in records)
    # the tick that places two blocks at once is ONE place action consuming both events
    multi = [r for r in records if r.a_hat is MacroAction.PLACE and len(r.event_ids) == 2]
    assert len(multi) == 1


def test_snapshot_rule_evidence_ends_before_the_action():
    session = generate_session(mixed_build())
    for record in evidence_stream(session.packets):
        for event_id in record.event_ids:
            assert record.tick_range[1] < _event_tick(session, event_id)


def test_long_pause_emits_unscored_context_records():
    # a ~64-tick pause between two placements -> three ~1 Hz context records inside it,
    # none scored, none consuming an event, all labeled with the ongoing run (idle)
    records = _records(sparse_build())
    context = [r for r in records if not r.scored]
    assert len(context) == 3
    assert all(r.event_ids == () for r in context)
    assert all(r.a_hat is MacroAction.IDLE and r.idle for r in context)


def test_context_records_do_not_disturb_consumption():
    records = _records(sparse_build())
    consumed = [eid for r in records for eid in r.event_ids]
    assert sorted(consumed) == [0, 1]           # both placements, exactly once
    scored = [r for r in records if r.scored]
    assert all(r.scored for r in scored) and len(scored) < len(records)


def test_single_consumption_no_event_counted_twice():
    consumed = [eid for r in _records(mixed_build()) for eid in r.event_ids]
    assert len(consumed) == len(set(consumed))
    assert sorted(consumed) == [0, 1, 2, 3]   # every event consumed exactly once


def test_state_feats_carries_the_held_item():
    places = [r for r in _records(wall_row_build()) if r.a_hat is MacroAction.PLACE]
    assert places and all(r.state_feats.held_item == "minecraft:oak_planks" for r in places)


def test_focus_blank_before_a_synthetic_place():
    # synthetic aims the crosshair only on the place tick itself (the action); the
    # pre-action window sees open air -> focus blank. This is the snapshot rule at work.
    places = [r for r in _records(wall_row_build()) if r.a_hat is MacroAction.PLACE]
    assert all(r.focus.block is None for r in places)


def test_idle_runs_emit_once_with_action_history():
    records = _records(wall_row_build())
    idles = [r for r in records if r.a_hat is MacroAction.IDLE]
    assert idles and all(r.idle for r in idles)
    # an idle right after a placement carries that placement in its action history
    assert any("place" in r.state_feats.recent_actions for r in idles)
