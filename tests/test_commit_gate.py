"""The commit gate: the staircase against hand-computed numbers, the two
hysteresis rules, the a-priori reachability machinery, the frozen masked-arm
mapping, and the reversibility table."""
import math

import pytest

from mica.contracts.b5 import ProposalChunk
from mica.contracts.goals import GOALS
from mica.decoder.grammar import Break, Move, Place, Say
from mica.gate import commit, reversibility
from mica.intent.tracker import MODES, TrackerParams

_PARAMS = TrackerParams(lambda_g=0.05, lambda_z=1.0, epsilon=0.01)
_DELTA = 1.0
_THRESHOLDS = commit.GateThresholds(c_min=0.5, theta_1=0.4, slope=0.02,
                                    irrev_penalty=0.2, m_consecutive=3)


def _belief(top_mass=0.6):
    """Goal marginal (top, rest even) with a uniform mode split — under this shape
    the propagated top mass has the closed form stay*m + (1-stay)/|G|."""
    rest = (1.0 - top_mass) / (len(GOALS) - 1)
    belief = {}
    for index, goal in enumerate(GOALS):
        mass = top_mass if index == 0 else rest
        for mode in MODES:
            belief[(goal, mode)] = mass / len(MODES)
    return belief


def _hand_p_star(top_mass, position):
    stay = math.exp(-_PARAMS.lambda_g * position * _DELTA)
    return stay * top_mass + (1.0 - stay) / len(GOALS)


def _chunk(confs, kinds=None):
    kinds = kinds or [Place(i, 0, 0, "minecraft:oak_planks") for i in range(len(confs))]
    return ProposalChunk(actions=tuple(kinds), token_conf=tuple(confs),
                         rationale_goal=None)


def test_staircase_matches_hand_computation():
    chunk = _chunk([1.0, 0.9, 0.4])
    k, rows = commit.k_commit(chunk, [False, False, False], "arm3", _belief(0.6),
                              0.6, _PARAMS, _DELTA, _THRESHOLDS)
    # positions 1 and 2 clear both bars; position 3's confidence 0.4 < c_min 0.5
    assert k == 2
    for row, position in zip(rows, (1, 2, 3)):
        assert row["p_star"] == pytest.approx(_hand_p_star(0.6, position), abs=1e-4)
    assert rows[0]["theta"] == pytest.approx(0.40)
    assert rows[1]["theta"] == pytest.approx(0.42)
    assert rows[2]["ok"] is False


def test_staircase_is_a_prefix_never_a_subset():
    # position 2 fails on confidence, position 3 would pass — K stays 1
    chunk = _chunk([1.0, 0.2, 0.9])
    k, rows = commit.k_commit(chunk, [False] * 3, "arm3", _belief(0.6), 0.6,
                              _PARAMS, _DELTA, _THRESHOLDS)
    assert k == 1
    assert rows[2]["ok"] is True    # recorded honestly, but not committed


def test_low_belief_commits_nothing():
    demanding = commit.GateThresholds(c_min=0.5, theta_1=0.7, slope=0.02,
                                      irrev_penalty=0.2, m_consecutive=3)
    k, _ = commit.k_commit(_chunk([1.0, 1.0]), [False, False], "arm3",
                           _belief(0.6), 0.6, _PARAMS, _DELTA, demanding)
    assert k == 0


def test_irreversible_positions_demand_more_mass():
    chunk = _chunk([1.0, 1.0])
    k_reversible, _ = commit.k_commit(chunk, [False, False], "arm3", _belief(0.6),
                                      0.6, _PARAMS, _DELTA, _THRESHOLDS)
    k_irrev, rows = commit.k_commit(chunk, [False, True], "arm3", _belief(0.6),
                                    0.6, _PARAMS, _DELTA, _THRESHOLDS)
    assert k_reversible == 2 and k_irrev == 1
    assert rows[1]["theta"] == pytest.approx(0.62)   # 0.4 + 0.02 + 0.2


def test_commit_hysteresis_grows_slowly_and_shrinks_instantly():
    hysteresis = commit.CommitHysteresis()
    assert hysteresis.read(5) == 1     # growth capped at +1 per read
    assert hysteresis.read(5) == 2
    assert hysteresis.read(0) == 0     # caution never waits
    assert hysteresis.read(3) == 1


def test_p_star_max_is_a_real_ceiling():
    ceiling = commit.p_star_max(_PARAMS, _DELTA)
    assert 1.0 / len(GOALS) < ceiling < 1.0
    # the map's fixed point: one more boost-then-decay step stays put
    boosted = commit._boost(ceiling, _PARAMS.humility_bound())
    assert commit._decay(boosted, _PARAMS, _DELTA) == pytest.approx(ceiling, abs=1e-9)
    # a MORE humble filter (bigger epsilon, smaller M) has a LOWER ceiling
    humble = TrackerParams(lambda_g=0.05, lambda_z=1.0, epsilon=0.2)
    assert commit.p_star_max(humble, _DELTA) < ceiling


def test_reachability_is_checkable_a_priori():
    report = commit.reachability_report(_THRESHOLDS, _PARAMS, _DELTA)
    assert report["all_reachable"] is True
    impossible = commit.GateThresholds(c_min=0.5, theta_1=0.99, slope=0.0,
                                       irrev_penalty=0.2, m_consecutive=3)
    report = commit.reachability_report(impossible, _PARAMS, _DELTA)
    assert report["all_reachable"] is False
    # the per-position bound is exactly the propagated ceiling
    ceiling = report["p_star_max"]
    assert report["positions"][0]["bound"] == pytest.approx(
        commit.propagated_bound(ceiling, 1, _PARAMS, _DELTA), abs=1e-4)


def test_masked_arm_mapping_is_frozen_and_static():
    # arm0: the uniform floor at every horizon
    assert commit.arm_p_star("arm0_zero", None, 0.9, 1, _PARAMS, _DELTA) == 0.2
    assert commit.arm_p_star("arm0_zero", None, 0.9, 8, _PARAMS, _DELTA) == 0.2
    # arm1/arm2: their stated mass, constant — nothing to propagate
    assert commit.arm_p_star("arm1_dense", None, 0.7, 5, _PARAMS, _DELTA) == 0.7
    assert commit.arm_p_star("arm2_llm", None, 0.7, 1, _PARAMS, _DELTA) == 0.7
    # arm3: kernel-propagated, decaying toward uniform as the horizon grows
    near = commit.arm_p_star("arm3", _belief(0.6), 0.6, 1, _PARAMS, _DELTA)
    far = commit.arm_p_star("arm3", _belief(0.6), 0.6, 8, _PARAMS, _DELTA)
    assert near > far > 1.0 / len(GOALS)
    with pytest.raises(ValueError):
        commit.arm_p_star("arm3", None, 0.6, 1, _PARAMS, _DELTA)


def test_expected_gap_freeze():
    gap = commit.expected_gap([[0, 20, 40], [0, 40]])
    assert gap["delta_hat_seconds"] == pytest.approx((1.0 + 2.0) / 2)
    with pytest.raises(ValueError):
        commit.expected_gap([[5]])


def test_reversibility_table():
    human = frozenset({(1, 0, 0)})
    assert reversibility.is_reversible(Place(0, 0, 0, "minecraft:oak_planks"), human)
    assert not reversibility.is_reversible(Place(0, 0, 0, "minecraft:water"), human)
    assert not reversibility.is_reversible(Place(0, 0, 0, "minecraft:tnt"), human)
    assert not reversibility.is_reversible(Break(1, 0, 0), human)   # human work
    assert reversibility.is_reversible(Break(2, 0, 0), human)
    assert reversibility.is_reversible(Move(1, 1, 1), human)
    assert reversibility.is_reversible(Say("ask_goal"), human)
    flags = reversibility.prefix_flags(
        [Place(0, 0, 0, "minecraft:oak_planks"), Break(1, 0, 0)], human)
    assert flags == [False, True]
