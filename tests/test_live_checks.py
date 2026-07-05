"""LiveGate: the per-packet half of the B0 gate. A clean synthetic session raises no
problems; each visible fault (tick hole, clock drift, lost event id, action with no
prior moment) is flagged on the exact packet that reveals it and lands in summary()."""
import dataclasses

from mica.capture.sample_builds import wall_row_build
from mica.capture.synthetic import ScriptedBuild, generate_session
from mica.validation.live_checks import LiveGate


def _packets(build=None):
    session = generate_session(build or wall_row_build())
    return sorted(session.packets, key=lambda p: p.tick)


def _run(gate, packets):
    problems = []
    for packet in packets:
        problems.extend(gate.check(packet))
    return problems


def test_clean_session_raises_no_problems():
    gate = LiveGate()
    assert _run(gate, _packets()) == []
    summary = gate.summary()
    assert summary["clean"] is True
    assert summary["tick_gaps"]["missing_ticks"] == 0
    assert summary["block_events"]["seen"] == 5
    assert summary["fields_never_seen"] == []


def test_tick_hole_is_counted_and_named():
    packets = _packets()
    thinned = [p for p in packets if p.tick != 3]
    gate = LiveGate()
    problems = _run(gate, thinned)
    assert any("tick gap" in problem for problem in problems)
    assert gate.summary()["tick_gaps"] == {"missing_ticks": 1, "stretches": 1}
    assert gate.summary()["clean"] is False


def test_lost_event_id_is_caught_on_the_next_event():
    # drop the packet carrying event id 1 (tick 10): ids then run 0, 2, ... and the
    # jump must be flagged the moment id 2 appears — this is how a dropped
    # event-carrying moment becomes visible instead of silently corrupting the world
    packets = [p for p in _packets() if p.tick != 10]
    gate = LiveGate()
    problems = _run(gate, packets)
    assert any("event id jumped 1 -> 2" in problem for problem in problems)
    assert gate.summary()["block_events"]["ids_skipped"] == 1


def test_action_right_after_a_hole_has_no_prior_moment():
    # remove tick 4: the placement at tick 5 has no tick-4 packet to read its
    # pre-action window's end from — flagged, because its evidence is damaged
    packets = [p for p in _packets() if p.tick != 4]
    gate = LiveGate()
    problems = _run(gate, packets)
    assert any("no prior moment" in problem for problem in problems)
    assert gate.summary()["actions_without_prior_moment"] == 1


def test_clock_drift_beyond_the_bound_is_flagged():
    build = wall_row_build()
    drifting = dataclasses.replace(build, alignment_jitter_ms=10_000.0)
    gate = LiveGate()
    problems = _run(gate, _packets(drifting))
    assert any("alignment residual" in problem for problem in problems)
    assert gate.summary()["alignment"]["violations"] > 0
    assert gate.summary()["clean"] is False


def test_summary_reports_a_field_that_never_appears():
    packets = _packets()
    blank_crosshair = [
        dataclasses.replace(
            p, client=dataclasses.replace(p.client, crosshair_target=None))
        for p in packets
    ]
    gate = LiveGate()
    _run(gate, blank_crosshair)
    assert "crosshair_target" in gate.summary()["fields_never_seen"]
    assert gate.summary()["clean"] is False
