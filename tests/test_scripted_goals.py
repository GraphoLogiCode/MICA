"""The Source-A generator: seeded determinism, real variance between sessions, and
self-validation — a scripted build's own label must be what D2 reads back off it."""
from mica.capture.scripted_goals import build_from_plan, plan_variants
from mica.capture.synthetic import generate_session
from mica.contracts.b0 import BlockOp
from mica.contracts.b1 import GOALS
from mica.perception.evidence2d import evidence_stream
from mica.perception.evidence3d import build_evidence3d
from mica.perception.voxel_replay import ReplayWorld, region_around_events


def _records(build):
    session = generate_session(build)
    corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    return build_evidence3d(ReplayWorld(region_around_events(session), {}), corrections, events)


def test_same_seed_same_corpus():
    a, _ = build_from_plan(plan_variants("habitation", 3, seed=7)[1])
    b, _ = build_from_plan(plan_variants("habitation", 3, seed=7)[1])
    assert a.placements == b.placements


def test_variants_actually_vary():
    plans = plan_variants("defense", 4, seed=7)
    builds = [build_from_plan(p)[0].placements for p in plans]
    assert len({tuple((pl.pos.x, pl.pos.y, pl.pos.z) for pl in b) for b in builds}) == 4
    assert len({p.order for p in plans}) >= 3
    assert {p.mode for p in plans} == {"deliberate", "shortcut"}


def test_full_deliberate_build_reads_back_as_its_own_label():
    # completion 1.0, orderly, no rush: D2's final trailing reading must name both the
    # category AND the style the generator wrote — coarse-to-fine, self-validating
    for goal in GOALS:
        for plan in plan_variants(goal, 2, seed=7)[:2]:
            full = plan if plan.completion == 1.0 else None
            if full is None:
                continue
            build, label = build_from_plan(full)
            last = _records(build)[-1]
            top = max(last.per_goal, key=lambda g: (last.per_goal[g].fit, last.per_goal[g].comp))
            assert top == goal, f"{goal} session read back as {top}"
            assert last.per_goal[goal].subtype == label["subtype"], \
                f"{label['subtype']} read back as {last.per_goal[goal].subtype}"


def test_taxonomy_and_templates_stay_coupled():
    from mica.contracts.goals import TAXONOMY, category_of
    from mica.perception.templates import TEMPLATES

    for goal, subtypes in TAXONOMY.items():
        assert {t.name for t in TEMPLATES[goal]} == set(subtypes)
        for subtype in subtypes:
            assert category_of(subtype) == goal


def test_mistakes_become_break_events_off_the_shape():
    shortcut = next(p for p in plan_variants("infrastructure", 6, seed=11) if p.mistake_rate >= 0.05)
    build, _ = build_from_plan(shortcut)
    breaks = [p for p in build.placements if p.op is BlockOp.BREAK]
    places = {(p.pos.x, p.pos.y, p.pos.z) for p in build.placements if p.op is BlockOp.PLACE}
    assert all((b.pos.x, b.pos.y, b.pos.z) in places for b in breaks)   # only breaks own mistakes
