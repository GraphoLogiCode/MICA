"""The decoder corpus: helper traces come out right by hand-check, hygiene
violations refuse, the origin rule is what the module says, and one real small
session flows end to end."""
import pytest

from decoder_fixtures import fused_record, toy_placements
from mica.capture.scripted_goals import build_from_plan, plan_variants
from mica.capture.synthetic import ScriptedPlacement, generate_session
from mica.contracts.b0 import BlockPos
from mica.contracts.b3 import fuse
from mica.contracts.goals import GOALS
from mica.data import decoder_corpus
from mica.decoder import context as context_builder
from mica.decoder.grammar import Break, Place
from mica.intent.tracker import uniform_belief
from mica.perception.evidence2d import evidence_stream
from mica.perception.evidence3d import build_evidence3d
from mica.perception.voxel_replay import ReplayWorld, region_around_events

_ORIGIN = (0, 64, 0)


def test_mistake_pairing_finds_the_placement_each_break_corrects():
    skips = decoder_corpus.skipped_place_ids(toy_placements())
    assert skips == {2}      # the (2,64,1) placement, not its break


def test_redundant_placement_is_skipped_not_violated():
    # A wrong-cell draw that collides with an already-standing cell (the generator
    # does this rarely): the second click never landed in reality, so the helper
    # skips it — and the hygiene rule no longer fires on the plan's own artifact.
    plan = (ScriptedPlacement(tick=10, pos=BlockPos(0, 64, 0),
                              block_type="minecraft:oak_planks"),
            ScriptedPlacement(tick=20, pos=BlockPos(0, 64, 0),
                              block_type="minecraft:oak_planks"),
            ScriptedPlacement(tick=30, pos=BlockPos(1, 64, 0),
                              block_type="minecraft:oak_planks"))
    skips = decoder_corpus.skipped_place_ids(plan)
    assert skips == {1}
    actions = decoder_corpus.helper_actions(plan, skips, 10, _ORIGIN)
    assert actions == [Place(0, 0, 0, "minecraft:oak_planks"),
                       Place(1, 0, 0, "minecraft:oak_planks")]


def test_helper_trace_skips_future_mistakes():
    # From tick 10 (nothing built yet): the trace is the five REAL cells in order —
    # the mistake and its break both vanish.
    actions = decoder_corpus.helper_actions(toy_placements(), {2}, 10, _ORIGIN)
    assert actions == [Place(0, 0, 0, "minecraft:oak_planks"),
                       Place(1, 0, 0, "minecraft:oak_planks"),
                       Place(2, 0, 0, "minecraft:oak_planks"),
                       Place(3, 0, 0, "minecraft:oak_planks"),
                       Place(4, 0, 0, "minecraft:oak_planks")]


def test_helper_trace_breaks_a_standing_mistake():
    # From tick 40 the mistake at (2,64,1) is already standing: the helper places
    # (2,64,0), breaks the mistake at its plan position, then continues.
    actions = decoder_corpus.helper_actions(toy_placements(), {2}, 40, _ORIGIN)
    assert actions[0] == Place(2, 0, 0, "minecraft:oak_planks")
    assert actions[1] == Break(2, 0, 1)
    assert actions[2] == Place(3, 0, 0, "minecraft:oak_planks")


def test_hygiene_refuses_a_target_that_is_already_standing():
    # A duplicate placement of an already-standing cell: the "future" op's effects
    # are already in the pre-action world, which is exactly the leak the rule stops.
    doubled = (ScriptedPlacement(tick=10, pos=BlockPos(0, 64, 0),
                                 block_type="minecraft:oak_planks"),
               ScriptedPlacement(tick=20, pos=BlockPos(0, 64, 0),
                                 block_type="minecraft:oak_planks"))
    with pytest.raises(decoder_corpus.CorpusHygieneViolation):
        decoder_corpus.helper_actions(doubled, set(), 20, _ORIGIN)


def test_hygiene_refuses_out_of_range_targets():
    far = (ScriptedPlacement(tick=10, pos=BlockPos(100, 64, 0),
                             block_type="minecraft:oak_planks"),)
    with pytest.raises(decoder_corpus.CorpusHygieneViolation):
        decoder_corpus.helper_actions(far, set(), 5, _ORIGIN)


def test_build_origin_is_the_earliest_built_records_centroid():
    records = [fused_record(tick=50, built_count=0, centroid=None),
               fused_record(tick=100, built_count=1, centroid=(3.4, 64.0, 2.6)),
               fused_record(tick=150, built_count=9, centroid=(9.9, 64.0, 9.9))]
    assert context_builder.build_origin(records) == (3, 64, 3)
    assert context_builder.build_origin([records[0]]) is None


def test_intervened_belief_is_a_point_mass_that_keeps_the_mode():
    belief = uniform_belief()
    forced = context_builder.intervened_belief(belief, "defense")
    assert abs(sum(forced.values()) - 1.0) < 1e-12
    for (goal, _), mass in forced.items():
        if goal != "defense":
            assert mass == 0.0
    slot = context_builder.arm3_slot(forced)
    assert slot["top_goal"] == "defense" and slot["p_top"] == pytest.approx(1.0)


def test_arm_slots_have_the_pinned_shapes():
    zero = context_builder.arm0_slot()
    assert set(zero["goal_marginal"]) == {0.0} and zero["top_goal"] is None
    dist = {g: 1.0 / len(GOALS) for g in GOALS}
    slot = context_builder.arm1_slot(dist)
    assert sum(slot["goal_marginal"]) == pytest.approx(1.0)
    assert slot["p_z1"] == 0.5     # neutral: no mode estimate exists


def test_load_rows_never_trains_on_eval_groups():
    import os

    if not os.path.exists(os.path.join(decoder_corpus.CORPUS_DIR, "decoder_labels.json")):
        pytest.skip("no banked decoder corpus on this machine")
    train, evaluation, labels = decoder_corpus.load_rows()
    train_sessions = {row["session"] for row in train}
    for session_id, label in labels.items():
        if label["group"] != "train":
            assert session_id not in train_sessions, (
                f"{session_id} ({label['group']}) leaked into the train split")
    assert any(labels[row["session"]]["group"] == "banked_holdout"
               for row in evaluation), "the transfer group must sit on the eval side"


def test_session_rows_end_to_end_on_a_small_real_plan():
    plan = plan_variants("defense", 1, 99)[0]
    build, label = build_from_plan(plan)
    session = generate_session(build)
    corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    world = ReplayWorld(region_around_events(session), {})
    b2 = build_evidence3d(world, corrections, events)
    fused = [fuse(a, b) for a, b in zip(corrections, b2)]

    rows = decoder_corpus.session_rows(build.session_id, label, fused,
                                       build.placements)
    assert len(rows) == len(fused)
    for row in rows:
        assert len(row["evidence"]) == context_builder.EVIDENCE_DIM
        assert sum(row["slots"]["arm3"]["goal_marginal"]) == pytest.approx(1.0, abs=1e-4)
    # early steps owe work, the last step owes (almost) nothing
    assert rows[0]["target"], "an unstarted build must yield a non-empty trace"
    assert len(rows[-1]["target"]) <= 2
