"""Heads v2 (D8) — the review checklist as tests.

Each pre-implementation checklist item from 2026-07-12 - D8 Design Review gets its
regression here: the channel partition (F5), the event gate (F1), the GM
normalization's odds preservation and mode-axis neutrality (F3), the floor-once
ordering and the C_gamma ceiling (F2). The math tests run against a stubbed model
cache so they need no trained weights on disk.
"""
import math

import pytest

from decoder_fixtures import fused_record
from mica.contracts.b1 import GOALS
from mica.intent import features
from mica.intent.heads_v2 import TrainedHeadsV2


# ---------------------------------------------------------------- F5: partition

def test_goal_block_partition_recomposes_v1_exactly():
    fused = fused_record()
    for goal in GOALS:
        whole = features.goal_dense(fused, goal)
        d3 = features.goal_dense_3d(fused, goal)
        d2 = features.goal_dense_2d(fused, goal)
        assert d3 + d2 == whole            # disjoint halves, nothing lost or shared
        assert len(d3) == features.GOAL_3D_DIM and len(d2) == features.GOAL_2D_DIM


def test_stream_heads_cannot_swallow_each_others_goal_block():
    # The structural half of F5: the two delib heads take different input widths,
    # so wiring the 3D block into the 2D head is a shape error, not a silent leak.
    th = pytest.importorskip("torch")
    from mica.intent.adapter_v2 import HeadsV2Model

    model = HeadsV2Model(vocab_size=4)
    e = th.zeros((1, 48))
    hot = th.zeros((1, len(GOALS)))
    block3 = th.zeros((1, features.GOAL_3D_DIM))
    with pytest.raises(RuntimeError):
        model.delib_logits_2d(e, block3, hot)


# ------------------------------------------------- a stubbed v2 for the math tests

def _stubbed(monkeypatch, gamma=1.0, posterior=None, offsets=None):
    heads = TrainedHeadsV2("heads_v2_test_stub")
    meta = {
        "goals": list(GOALS), "w2": 1.0, "w3": 1.0, "gamma": gamma,
        "c_gamma": round(len(GOALS) ** gamma, 4),
        "goal_offsets": offsets or {g: 0.0 for g in GOALS},
        "p_hat": {g: 1.0 / len(GOALS) for g in GOALS},
        "temperature_2d_delib": 1.0, "temperature_2d_heur": 1.0,
        "temperature_3d_delib": 1.0, "temperature_3d_heur": 1.0,
        "epsilon": 0.01, "lambda_g": 0.02, "lambda_z": 0.5,
    }
    heads._cache = ({}, meta)
    if posterior is not None:
        from mica.intent import heads_v2 as module
        monkeypatch.setattr(module.arm1, "distribution", lambda fused: dict(posterior))
    # uniform stream tables: any g-dependence in the output is the readout's alone
    uniform = [0.2] * 5
    monkeypatch.setattr(
        heads, "stream_tables",
        lambda fused: (({g: uniform for g in GOALS}, uniform),
                       ({g: uniform for g in GOALS}, uniform)))
    return heads


# ---------------------------------------------------------------- F1: event gate

def test_readout_never_moves_the_goal_axis_without_block_events(monkeypatch):
    posterior = {"habitation": 0.7, "production": 0.1, "defense": 0.1,
                 "infrastructure": 0.05, "decorative": 0.05}
    heads = _stubbed(monkeypatch, posterior=posterior)
    from mica.contracts.b1 import MacroAction

    idle = fused_record(event_ids=())
    table = heads.likelihood(idle, MacroAction.IDLE)
    values = {table[(g, 0)] for g in GOALS}
    assert len(values) == 1                # static scene: no goal got an edge

    scored = fused_record(event_ids=(7,))
    table = heads.likelihood(scored, MacroAction.PLACE)
    assert table[("habitation", 0)] > table[("production", 0)]   # events: readout live


# ------------------------------------------------ F3: GM normalization semantics

def test_gm_normalization_preserves_pairwise_goal_odds(monkeypatch):
    posterior = {"habitation": 0.5, "production": 0.25, "defense": 0.15,
                 "infrastructure": 0.06, "decorative": 0.04}
    heads = _stubbed(monkeypatch, gamma=0.5, posterior=posterior)
    factor = heads.readout_factor(fused_record())
    for a in GOALS:
        for b in GOALS:
            expected = (posterior[a] / posterior[b]) ** 0.5
            assert factor[a] / factor[b] == pytest.approx(expected, rel=1e-6)


def test_uniform_readout_is_exactly_neutral(monkeypatch):
    uniform = {g: 1.0 / len(GOALS) for g in GOALS}
    heads = _stubbed(monkeypatch, posterior=uniform)
    factor = heads.readout_factor(fused_record())
    assert all(v == pytest.approx(1.0) for v in factor.values())
    # and the z=1 branch never sees the factor at all, events or not:
    from mica.contracts.b1 import MacroAction

    table = heads.likelihood(fused_record(event_ids=(3,)), MacroAction.PLACE)
    assert len({table[(g, 1)] for g in GOALS}) == 1


# --------------------------------------------------- F2: ceiling and floor order

def test_readout_factor_is_clipped_at_the_pinned_ceiling(monkeypatch):
    # a (numerically) certain classifier: without the clip the GM-normalized
    # factor would exceed |G|^gamma; the pin caps it there.
    posterior = {"habitation": 1.0 - 4e-9, "production": 1e-9, "defense": 1e-9,
                 "infrastructure": 1e-9, "decorative": 1e-9}
    heads = _stubbed(monkeypatch, gamma=1.0, posterior=posterior)
    factor = heads.readout_factor(fused_record())
    ceiling = len(GOALS) ** 1.0
    assert max(factor.values()) == pytest.approx(ceiling)
    assert min(factor.values()) >= 1.0 / ceiling - 1e-9


def test_v2_tables_are_raw_and_the_tracker_floors_once(monkeypatch):
    from mica.contracts.b1 import MacroAction
    from mica.intent.tracker import TrackerParams, correct, uniform_belief

    posterior = {"habitation": 0.9, "production": 0.05, "defense": 0.03,
                 "infrastructure": 0.01, "decorative": 0.01}
    heads = _stubbed(monkeypatch, posterior=posterior)
    table = heads.likelihood(fused_record(event_ids=(1,)), MacroAction.PLACE)
    params = TrackerParams()
    floor = params.epsilon / params.action_count
    # raw: the readout pushes some entries BELOW where a pre-floored table could sit
    assert min(table.values()) < (1.0 - params.epsilon) * 0.2 + floor
    _, normalizer = correct(uniform_belief(), table, params)
    assert normalizer >= floor - 1e-15     # the once-only floor holds the Z_k bound


def test_gamma_zero_recovers_an_action_only_deliberative_branch(monkeypatch):
    heads = _stubbed(monkeypatch, gamma=0.0)
    factor = heads.readout_factor(fused_record())
    assert all(v == 1.0 for v in factor.values())
