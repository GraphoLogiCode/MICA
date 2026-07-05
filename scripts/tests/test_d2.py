"""D2 on synthetic builds: features are pre-action (snapshot rule), completion tracks
the scripted build, registration finds the build's rotation and style, and every B2
record joins one-to-one with its B1 correction."""
from mica.capture.sample_builds import bridge_deck_build, pen_build
from mica.capture.synthetic import generate_session
from mica.perception.evidence2d import evidence_stream
from mica.perception.evidence3d import build_evidence3d
from mica.perception.voxel_replay import ReplayWorld, region_around_events
from mica.validation.evidence3d_check import check_join, check_stream


def _run(build):
    session = generate_session(build)
    corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    world = ReplayWorld(region_around_events(session), {})
    return corrections, build_evidence3d(world, corrections, events)


def test_pen_completion_rises_and_stays_pre_action():
    corrections, records = _run(pen_build())
    pen_comp = [r.per_goal["production"].comp for r in records if r.event_ids]
    # each record is computed BEFORE its own placement: the first sees an empty world,
    # the last sees 15 of 16 posts — never its own block (the snapshot rule, on data)
    assert pen_comp[0] == 0.0
    assert all(a <= b for a, b in zip(pen_comp, pen_comp[1:]))
    assert 0.9 < pen_comp[-1] < 1.0


def test_pen_wins_the_category_contest_and_names_its_style():
    _, records = _run(pen_build())
    last_place = [r for r in records if r.event_ids][-1]
    trailing_idle = records[-1]
    top_goal = max(last_place.per_goal, key=lambda g: last_place.per_goal[g].comp)
    assert top_goal == "production"
    assert last_place.per_goal["production"].subtype == "fence pen"   # the fine layer
    # a half-built ring encloses nothing (the missing corner at the very end is a
    # different story: a corner gap blocks 4-connected passage, so 15 posts DO enclose)
    halfway = [r for r in records if r.event_ids][8]
    assert not halfway.global_feats.has_enclosure
    assert last_place.per_goal["production"].comp < 1.0
    # ...while the trailing idle correction, scored AFTER the final post, rightly sees
    # the finished pen — earlier actions are legitimate evidence for later ones
    assert trailing_idle.global_feats.has_enclosure
    assert trailing_idle.per_goal["production"].comp == 1.0
    assert trailing_idle.per_goal["production"].fit == 1.0


def test_finished_pen_closes_a_loop():
    from mica.perception.evidence3d import _global_feats

    session = generate_session(pen_build())
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    world = ReplayWorld(region_around_events(session), {})
    for event_id in sorted(events):
        world.apply(events[event_id])
    finished = _global_feats(world, ())
    assert finished.built_count == 16
    assert finished.has_enclosure
    assert finished.symmetry > 0.9   # a square ring mirrors onto itself


def test_bridge_registration_finds_rotation_and_style():
    # the fixture builds the railed-bridge instance exactly (deck plus both rail rows),
    # once along each axis — registration must recover the axis AND name the style
    _, along_x = _run(bridge_deck_build(axis="x"))
    _, along_z = _run(bridge_deck_build(axis="z"))
    fit_x = along_x[-1].per_goal["infrastructure"]
    fit_z = along_z[-1].per_goal["infrastructure"]
    assert fit_x.fit > 0.9 and fit_x.pose.rot in (0, 180) and fit_x.subtype == "railed bridge"
    assert fit_z.fit > 0.9 and fit_z.pose.rot in (90, 270) and fit_z.subtype == "railed bridge"


def test_join_is_one_to_one_and_contract_clean():
    corrections, records = _run(pen_build())
    assert check_stream(records) == []
    assert check_join(list(corrections), records) == []
    consumed = [eid for r in records for eid in r.event_ids]
    assert sorted(consumed) == list(range(16))   # every fence post, exactly once
