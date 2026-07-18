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
    assert last_place.per_goal["production"].subtype == "animal_husbandry"   # the fine layer (v3 style read)
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


def test_symmetry_exact_on_rectangles_and_translation_invariant():
    # Perception review 2026-07-16 F1: the old axis-mirror formulas computed the
    # negated centroid offset instead of the reflection, so a NON-SQUARE symmetric
    # shape (the diagonal mirrors can't rescue it) scored 0.33-0.5, and the same
    # shape scored differently at different world positions. Both properties are
    # pinned here: exact 1.0 for a mirror-symmetric rectangle ring, at any offset.
    from mica.perception.evidence3d import _symmetry

    ring = list({(x, 0, 0) for x in range(5)} | {(x, 0, 2) for x in range(5)}
                | {(0, 0, z) for z in range(3)} | {(4, 0, z) for z in range(3)})
    scores = set()
    for ox, oz in ((0, 0), (200, 300), (7, 13), (-40, 999)):   # even and odd offsets
        score, support = _symmetry([(x + ox, 0, z + oz) for x, _, z in ring])
        scores.add(score)
        assert support == 12
    assert scores == {1.0}

    # a centroid BETWEEN cells (2x2 block) must still mirror exactly
    assert _symmetry([(10, 0, 10), (11, 0, 10), (10, 0, 11), (11, 0, 11)])[0] == 1.0

    # Shapes symmetric under exactly ONE mirror each, so every mirror formula is
    # pinned individually — max() over mirrors means a broken formula hides behind
    # a correct one on any doubly-symmetric shape (how the original bug shipped).
    t_shape = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0),
               (2, 0, 1), (2, 0, 2)]                       # x-plane mirror only
    assert _symmetry(t_shape)[0] == 1.0
    t_rot = [(z, 0, x) for x, _, z in t_shape]             # z-plane mirror only
    assert _symmetry(t_rot)[0] == 1.0
    l_diag = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 0, 1), (0, 0, 2)]   # diagonal only
    assert _symmetry(l_diag)[0] == 1.0

    # and a genuinely asymmetric shape must not read symmetric
    j = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 0, 1), (3, 0, 2)]
    assert _symmetry(j)[0] < 0.75


def test_bridge_registration_finds_rotation_and_style():
    # the fixture builds the railed-bridge instance exactly (deck plus both rail rows),
    # once along each axis — registration must recover the axis AND name the style
    _, along_x = _run(bridge_deck_build(axis="x"))
    _, along_z = _run(bridge_deck_build(axis="z"))
    fit_x = along_x[-1].per_goal["infrastructure"]
    fit_z = along_z[-1].per_goal["infrastructure"]
    assert fit_x.fit > 0.9 and fit_x.pose.rot in (0, 180) and fit_x.subtype == "bridge"
    assert fit_z.fit > 0.9 and fit_z.pose.rot in (90, 270) and fit_z.subtype == "bridge"


def test_stray_low_block_does_not_sink_the_template():
    # Perception review 2026-07-16 F2: the vertical anchor was the lowest built cell
    # anywhere, so ONE stray block below the build sank the whole template into the
    # terrain — underground stone satisfied its #solid cells (comp stayed 1.0 for the
    # wrong reason) and every real cabin block became "surplus" (edit 0 -> 57). The
    # anchor is now registered by built-cell alignment, so the stray must not move it.
    from mica.perception.evidence3d import _read_goal
    from mica.perception.templates import instance
    from mica.perception.voxel_replay import Region, ReplayWorld

    cabin = instance("cabin")
    region = Region(-50, 0, -50, 80, 100, 80)
    stone = {(x, y, z): "minecraft:stone"
             for x in range(-20, 40) for z in range(-20, 40) for y in range(55, 64)}
    world = ReplayWorld(region, dict(stone))
    for (dx, dy, dz) in cabin.cells:
        world.changed[(dx, 64 + dy, dz)] = "minecraft:oak_planks"

    clean = _read_goal(world, cabin, world.built())
    assert (clean.comp, clean.edit_distance) == (1.0, 0)
    assert clean.pose.dy == 64

    world.changed[(30, 60, 30)] = "minecraft:cobblestone"   # one stray, 4 below, far away
    stray = _read_goal(world, cabin, world.built())
    assert stray.pose.dy == 64                               # anchor did not move
    assert stray.comp == 1.0
    assert stray.edit_distance <= 1                          # at most the stray itself


def test_high_start_anchors_to_the_built_layer_not_min_y():
    # The other side of F2: a builder who starts HIGH (a treehouse platform before
    # its trunk) was unreadable under min-y — the anchor sat at the platform, pushing
    # the template's own platform a further 5 blocks up. Alignment voting anchors the
    # template so its slab layer meets the built slab.
    from mica.perception.evidence3d import _read_goal
    from mica.perception.templates import instance
    from mica.perception.voxel_replay import Region, ReplayWorld

    treehouse = instance("treehouse")
    world = ReplayWorld(Region(-50, 0, -50, 80, 100, 80), {})
    for dx in range(5):                                      # the platform only, at y=70
        for dz in range(5):
            world.changed[(dx, 70, dz)] = "minecraft:oak_planks"

    reading = _read_goal(world, treehouse, world.built())
    assert reading.pose.dy == 65                             # slab layer (dy=5) meets y=70
    assert reading.comp == 25 / 30                           # platform done, trunk missing


def test_join_is_one_to_one_and_contract_clean():
    corrections, records = _run(pen_build())
    assert check_stream(records) == []
    assert check_join(list(corrections), records) == []
    consumed = [eid for r in records for eid in r.event_ids]
    assert sorted(consumed) == list(range(16))   # every fence post, exactly once
