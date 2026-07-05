"""The choreographer: natural, physically plausible body and gaze around a frozen
placement plan — deterministic, opt-in (the static default stays byte-identical),
and honest to the D1 thresholds (aim lands before the action, flicks only when the
schedule forces them)."""
import math
from statistics import mean

from mica.capture.motion import DELIBERATE_GAZE, choreograph
from mica.capture.sample_builds import wall_row_build
from mica.capture.scripted_goals import build_from_plan, plan_variants
from mica.capture.synthetic import ScriptedBuild, generate_session
from mica.contracts.b0 import BlockOp
from mica.contracts.b1 import GOALS, MacroAction
from mica.perception.evidence2d import evidence_stream
from mica.perception.evidence3d import build_evidence3d
from mica.perception.voxel_replay import ReplayWorld, region_around_events
from mica.validation.evidence2d_check import check_stream as check_b1
from mica.validation.evidence3d_check import check_join, check_stream as check_b2

_EYE_HEIGHT = 1.62
_REACH = 4.5
_SPRINT_STEP = 0.281       # vanilla sprint speed per tick, plus float headroom
_FLICK_LIMIT = 180.001     # one mouse yank is the hard bound on any single turn
_CACHE: dict = {}


def _corpus():
    """Every session of the default scripted corpus, generated once."""
    if "sessions" not in _CACHE:
        _CACHE["sessions"] = [
            (build, generate_session(build))
            for goal in GOALS
            for build, _ in (build_from_plan(plan) for plan in plan_variants(goal, 4, seed=7))
        ]
    return _CACHE["sessions"]


def _action_gaps(build):
    """tick -> distance to the next action tick, for every action but the last."""
    ticks = sorted({placement.tick for placement in build.placements})
    return {a: b - a for a, b in zip(ticks, ticks[1:])}


def _confirm_run(session, placement):
    """How many ticks right after a placement keep the crosshair on the new block."""
    by_tick = {packet.tick: packet for packet in session.packets}
    run, tick = 0, placement.tick + 1
    while tick in by_tick and by_tick[tick].client.crosshair_target.block_pos == placement.pos:
        run += 1
        tick += 1
    return run


def test_choreography_is_deterministic():
    plan = plan_variants("habitation", 3, seed=7)[0]
    build_a, _ = build_from_plan(plan)
    build_b, _ = build_from_plan(plan)
    assert choreograph(build_a) == choreograph(build_b)
    assert generate_session(build_a).packets == generate_session(build_b).packets


def test_static_default_pose_is_unchanged():
    build = wall_row_build()
    assert build.motion is None
    for packet in generate_session(build).packets:
        pos = packet.server.player_pos
        assert (pos.x, pos.y, pos.z) == (0.0, 64.0, 0.0)
        assert packet.client.yaw == 0.0
        assert packet.client.pitch == (45.0 if packet.server.block_events else 0.0)


def test_builder_acts_within_reach():
    for _, session in _corpus():
        for packet in session.packets:
            for event in packet.server.block_events:
                pos = packet.server.player_pos
                eye = (pos.x, pos.y + _EYE_HEIGHT, pos.z)
                center = (event.pos.x + 0.5, event.pos.y + 0.5, event.pos.z + 0.5)
                assert math.dist(eye, center) <= _REACH


def test_walking_is_physical():
    # Reach is the hard invariant, so a frozen schedule that outruns vanilla sprint
    # costs speed instead: the careful builder stays under sprint, the hasty one is
    # allowed its declared scramble — and never more.
    for build, session in _corpus():
        cap = (_SPRINT_STEP if build.motion is DELIBERATE_GAZE
               else build.motion.dash_cap + 0.001)
        previous = None
        for packet in session.packets:
            pos = packet.server.player_pos
            if previous is not None:
                step = math.hypot(pos.x - previous.x, pos.z - previous.z)
                assert step <= cap
                if step > 0.05:
                    assert "forward" in packet.client.input_state.keys
                if packet.server.block_events:
                    assert step == 0.0
                    assert packet.client.input_state.keys == ()
            previous = pos


def test_head_turns_bounded_and_flicks_only_just_before_actions():
    for build, session in _corpus():
        action_ticks = {placement.tick for placement in build.placements}
        previous = None
        for packet in session.packets:
            if previous is not None:
                turn = max(abs(packet.client.yaw - previous.client.yaw),
                           abs(packet.client.pitch - previous.client.pitch))
                assert turn <= _FLICK_LIMIT
                if turn > build.motion.turn_cap_deg + 1e-9:
                    assert packet.tick + 1 in action_ticks
            previous = packet


def test_every_action_is_aimed_at_before_it_happens():
    for _, session in _corpus():
        events = {e.event_id: e for p in session.packets for e in p.server.block_events}
        for record in evidence_stream(session.packets):
            if record.scored and record.a_hat in (MacroAction.PLACE, MacroAction.BREAK):
                assert record.focus.block == events[record.event_ids[0]].pos
                assert record.focus.dwell_ticks >= 1


def test_places_get_a_confirming_look():
    runs = []
    for build, session in _corpus():
        if build.motion is not DELIBERATE_GAZE:
            continue
        gaps = _action_gaps(build)
        runs.extend(_confirm_run(session, placement)
                    for placement in build.placements
                    if placement.op is BlockOp.PLACE and gaps.get(placement.tick, 99) >= 12)
    assert runs
    assert sum(1 for run in runs if run >= 1) / len(runs) >= 0.8
    assert max(runs) >= 4


def test_breaks_get_a_verifying_stare():
    stared = False
    for build, session in _corpus():
        by_tick = {packet.tick: packet for packet in session.packets}
        gaps = _action_gaps(build)
        for placement in build.placements:
            if placement.op is not BlockOp.BREAK or gaps.get(placement.tick, 99) < 8:
                continue
            first, second = by_tick[placement.tick + 1], by_tick[placement.tick + 2]
            still = (abs(second.client.yaw - first.client.yaw) <= 1.0
                     and abs(second.client.pitch - first.client.pitch) <= 1.0)
            blank = (first.client.crosshair_target.block_pos is None
                     and second.client.crosshair_target.block_pos is None)
            stared = stared or (still and blank)
    assert stared


def test_thinking_pauses_sweep_the_structure():
    swept = False
    for build, session in _corpus():
        if build.motion is not DELIBERATE_GAZE:
            continue
        by_tick = {packet.tick: packet for packet in session.packets}
        changes_at = {}
        for placement in build.placements:
            changes_at.setdefault(placement.tick, []).append(placement)
        ticks = sorted(changes_at)
        placed = set()
        for a, b in zip(ticks, ticks[1:]):
            for placement in changes_at[a]:
                cell = (placement.pos.x, placement.pos.y, placement.pos.z)
                placed.add(cell) if placement.op is BlockOp.PLACE else placed.discard(cell)
            if b - a < 60:
                continue
            edges = (changes_at[a][0].pos, changes_at[b][0].pos)
            for tick in range(a + 1, b):
                block = by_tick[tick].client.crosshair_target.block_pos
                if (block is not None and block not in edges
                        and (block.x, block.y, block.z) in placed):
                    swept = True
    assert swept


def test_deliberate_lingers_longer_than_shortcut():
    runs: dict[bool, list[int]] = {True: [], False: []}
    for build, session in _corpus():
        gaps = _action_gaps(build)
        runs[build.motion is DELIBERATE_GAZE].extend(
            _confirm_run(session, placement)
            for placement in build.placements
            if placement.op is BlockOp.PLACE and gaps.get(placement.tick, 99) >= 12)
    assert runs[True] and runs[False]
    assert mean(runs[True]) > mean(runs[False])


def test_companion_glances_appear_off_action_ticks():
    seen = False
    for _, session in _corpus():
        for packet in session.packets:
            target = packet.client.crosshair_target
            if target.entity == "minecraft:player":
                assert target.block_pos is None
                assert not packet.server.block_events
                seen = True
    assert seen


def test_choreographed_session_passes_the_chain_contracts():
    plan = next(p for p in plan_variants("habitation", 4, seed=7) if p.mode == "deliberate")
    build, _ = build_from_plan(plan)
    session = generate_session(build)
    b1 = list(evidence_stream(session.packets))
    corrections = [record for record in b1 if record.scored]
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    b2 = build_evidence3d(ReplayWorld(region_around_events(session), {}), tuple(corrections), events)
    assert check_b1(b1) == []
    assert check_b2(b2) == []
    assert check_join(corrections, b2) == []
    consumed = sorted(eid for record in b1 for eid in record.event_ids)
    assert consumed == sorted(events)
