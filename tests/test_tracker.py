"""The belief tracker's proven guarantees, as executable properties: mass preservation,
exact lazy prediction, a normalizer that cannot vanish, the humility bound, and the
goal-freeze under heuristic mode. Random streams, not fixed fixtures."""
import math
import random

from mica.contracts.b1 import GOALS, Focus, MacroAction, StateFeats
from mica.contracts.b2 import GlobalStructure, PerGoalStructure, Pose
from mica.contracts.b3 import FusedEvidence, fuse
from mica.intent.heads_v0 import ACTIONS, heuristic, likelihood, strip_behavior
from mica.intent.tracker import (
    MODES, TrackerParams, category_marginal, correct, predict, uniform_belief,
)

_SUBTYPE_OF = {"habitation": "cabin", "infrastructure": "flat span",
               "production": "fence pen", "defense": "square tower",
               "decorative": "fountain"}


def _fused(held="minecraft:oak_fence", progressing="production"):
    per_goal = {
        g: PerGoalStructure(comp=0.5, edit_distance=5, fit=0.5, pose=Pose(0, 0, 0),
                            subtype=_SUBTYPE_OF[g],
                            delta_comp=0.1 if g == progressing else 0.0)
        for g in GOALS
    }
    return FusedEvidence(
        tick=100, event_ids=(1,), a_hat=MacroAction.PLACE, idle=False,
        state_feats=StateFeats(held_item=held, hotbar=(held,), pos_delta=(0.0, 0.0, 0.0),
                               yaw_delta=0.0, pitch_delta=0.0,
                               recent_actions=("place", "place", "idle")),
        focus=Focus(block=None, dwell_ticks=0),
        global_feats=GlobalStructure(built_count=8, bbox=(0, 64, 0, 4, 64, 4),
                                     centroid=(2.0, 64.0, 2.0), planar_runs=2,
                                     has_enclosure=False, symmetry=0.5,
                                     symmetry_support=8, edit_locality=1.0),
        s_goal=None, per_goal=per_goal,
    )

_P = TrackerParams()


def _random_belief(rng):
    raw = {(g, z): rng.random() + 1e-6 for g in GOALS for z in MODES}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def _random_likelihood(rng):
    return {(g, z): rng.random() for g in GOALS for z in MODES}


def test_predict_preserves_mass_and_positivity():
    rng = random.Random(3)
    for _ in range(50):
        b = predict(_random_belief(rng), rng.uniform(0.05, 30.0), _P)
        assert abs(sum(b.values()) - 1.0) < 1e-12
        assert all(v > 0 for v in b.values())


def test_lazy_prediction_is_exact():
    # Chapman-Kolmogorov: one long gap equals its pieces (Theorem 5)
    rng = random.Random(4)
    for _ in range(20):
        b = _random_belief(rng)
        s, t = rng.uniform(0.1, 5.0), rng.uniform(0.1, 5.0)
        one_step = predict(b, s + t, _P)
        two_step = predict(predict(b, s, _P), t, _P)
        assert all(abs(one_step[k] - two_step[k]) < 1e-12 for k in b)


def test_normalizer_never_below_the_floor():
    rng = random.Random(5)
    for _ in range(50):
        b = predict(_random_belief(rng), 1.0, _P)
        _, z_k = correct(b, _random_likelihood(rng), _P)
        assert z_k >= _P.epsilon / _P.action_count - 1e-15


def test_humility_bound_caps_every_odds_shift():
    # Theorem 6: posterior odds / predicted odds within [1/M, M] for every pair
    rng = random.Random(6)
    m = _P.humility_bound()
    for _ in range(30):
        b_bar = predict(_random_belief(rng), 1.0, _P)
        b, _ = correct(b_bar, _random_likelihood(rng), _P)
        keys = list(b)
        for _ in range(10):
            i, j = rng.sample(keys, 2)
            shift = (b[i] / b[j]) / (b_bar[i] / b_bar[j])
            assert 1.0 / m - 1e-9 <= shift <= m + 1e-9


def test_goal_belief_freezes_under_heuristic_mode():
    # Proposition 7: a likelihood constant across goals at z=1 leaves P(g | z=1) alone
    rng = random.Random(7)
    b_bar = predict(_random_belief(rng), 1.0, _P)
    flat = {(g, z): (0.7 if z == 1 else rng.random()) for g in GOALS for z in MODES}
    b, _ = correct(b_bar, flat, _P)
    before = {g: b_bar[(g, 1)] / sum(b_bar[(x, 1)] for x in GOALS) for g in GOALS}
    after = {g: b[(g, 1)] / sum(b[(x, 1)] for x in GOALS) for g in GOALS}
    assert all(abs(before[g] - after[g]) < 1e-12 for g in GOALS)


def test_heads_honor_the_constraints():
    evidence = _fused()
    # heuristic head: distribution over exactly A, and blind to everything g-indexed —
    # permuting the per-goal features must not move it at all
    h = heuristic(evidence)
    assert set(h) == set(ACTIONS) and abs(sum(h.values()) - 1.0) < 1e-12
    reshuffled = _fused(progressing="decorative")
    assert heuristic(reshuffled) == h
    # full likelihood: z=1 constant in g; z=0 favors the progressing goal for PLACE
    like = likelihood(evidence, MacroAction.PLACE)
    z1 = {like[(g, 1)] for g in GOALS}
    assert len(z1) == 1
    assert like[("production", 0)] > like[("decorative", 0)]


def test_the_d1_wire_is_load_bearing():
    # the held fence is production evidence the structure stream cannot see: with it,
    # production's PLACE likelihood beats the same evidence holding a generic block
    with_fence = likelihood(_fused(held="minecraft:oak_fence"), MacroAction.PLACE)
    with_dirt = likelihood(_fused(held="minecraft:dirt"), MacroAction.PLACE)
    assert with_fence[("production", 0)] > with_dirt[("production", 0)]
    # stripping the behavior channels must remove exactly that advantage
    stripped = strip_behavior(_fused(held="minecraft:oak_fence"))
    assert stripped.state_feats.held_item == "minecraft:air"
    assert stripped.state_feats.recent_actions == ()
    assert stripped.focus.block is None and stripped.focus.dwell_ticks == 0
    assert stripped.s_goal is None
    assert stripped.per_goal == _fused().per_goal   # the 3D structure is untouched


def test_tracker_converges_on_a_scripted_pen():
    from mica.capture.sample_builds import pen_build
    from mica.capture.synthetic import generate_session
    from mica.perception.evidence2d import evidence_stream
    from mica.perception.evidence3d import build_evidence3d
    from mica.perception.voxel_replay import ReplayWorld, region_around_events

    session = generate_session(pen_build())
    corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    b2 = build_evidence3d(ReplayWorld(region_around_events(session), {}), corrections, events)

    belief = uniform_belief()
    previous_tick = 0
    for b1_record, b2_record in zip(corrections, b2):
        fused = fuse(b1_record, b2_record)   # the verified handoff, on typed records
        dt = max(fused.tick - previous_tick, 1) / 20.0
        previous_tick = fused.tick
        belief = predict(belief, dt, _P)
        belief, _ = correct(belief, likelihood(fused, fused.a_hat), _P)
        assert abs(sum(belief.values()) - 1.0) < 1e-9

    marginal = category_marginal(belief)
    assert max(marginal, key=marginal.get) == "production"
    assert math.isfinite(sum(marginal.values()))
