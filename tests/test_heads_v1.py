"""The trained heads' structural guarantees, checked on an untrained model — they must
hold by construction, not by luck of the weights: the heuristic side cannot see goal
channels, missing dense channels are represented (not crashed on), and the v1
likelihood speaks the exact interface the tracker already consumes from v0."""
import dataclasses
import json

import pytest

th = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from mica.capture.sample_builds import pen_build
from mica.capture.synthetic import generate_session
from mica.contracts.b1 import GOALS, MacroAction
from mica.contracts.b3 import fuse
from mica.intent import features, heads_v1
from mica.intent.adapter import HeadsV1Model
from mica.intent.tracker import TrackerParams
from mica.perception.evidence2d import evidence_stream
from mica.perception.evidence3d import build_evidence3d
from mica.perception.voxel_replay import ReplayWorld, region_around_events


def _fused_record():
    session = generate_session(pen_build())
    corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    b2 = build_evidence3d(ReplayWorld(region_around_events(session), {}), corrections, events)
    return next(fuse(a, b) for a, b in zip(corrections, b2) if a.event_ids)


@pytest.fixture()
def shipped_model(tmp_path, monkeypatch):
    """An untrained model exported exactly the way train_heads.py ships one, loaded
    through heads_v1's own file path — the full inference route, no shortcuts."""
    th.manual_seed(0)
    vocab = ("<unk>", "minecraft:oak_fence", "minecraft:air")
    model = HeadsV1Model(len(vocab))
    weights = tmp_path / "heads_v1.npz"
    meta = tmp_path / "heads_v1.json"
    np.savez(str(weights), **model.export_arrays())
    meta.write_text(json.dumps({
        "goals": list(GOALS), "actions": [a.value for a in features.ACTION_ORDER],
        "vocab": list(vocab), "temperature_delib": 1.0, "temperature_heur": 1.0,
        "epsilon": 0.1, "lambda_g": 0.05, "lambda_z": 1.0,
    }), encoding="utf-8")
    monkeypatch.setattr(heads_v1, "_WEIGHTS", str(weights))
    monkeypatch.setattr(heads_v1, "_META", str(meta))
    monkeypatch.setattr(heads_v1, "_cache", None)
    return model


def test_heuristic_ignores_goal_channels(shipped_model):
    fused = _fused_record()
    # Same record with every goal-bearing channel altered: s_goal filled in, the
    # per-goal structure reads swapped. The deliberative side must move; the
    # heuristic side must not — its signature has no path to these inputs (09-F4).
    goals = list(fused.per_goal)
    swapped = {a: fused.per_goal[b] for a, b in zip(goals, goals[1:] + goals[:1])}
    altered = dataclasses.replace(fused, s_goal=(0.9, 0.1, 0.1, 0.1, 0.1), per_goal=swapped)
    original = heads_v1.likelihood(fused, fused.a_hat)
    changed = heads_v1.likelihood(altered, fused.a_hat)
    for goal in GOALS:
        assert original[(goal, 1)] == pytest.approx(changed[(goal, 1)], abs=1e-9)
    assert any(abs(original[(g, 0)] - changed[(g, 0)]) > 1e-9 for g in GOALS)


def test_heuristic_constant_in_goal(shipped_model):
    table = heads_v1.likelihood(_fused_record(), MacroAction.PLACE)
    heur = {table[(g, 1)] for g in GOALS}
    assert len(heur) == 1, "P(a | e, z=1) must not vary with g"


def test_none_channels_survive(shipped_model):
    fused = dataclasses.replace(_fused_record(), h2d=None, h3d=None, s_goal=None)
    table = heads_v1.likelihood(fused, MacroAction.IDLE)
    assert set(table) == {(g, z) for g in GOALS for z in (0, 1)}
    assert all(0.0 < v < 1.0 for v in table.values())


def test_v1_speaks_the_v0_interface(shipped_model):
    from mica.intent import heads_v0

    fused = _fused_record()
    v0 = heads_v0.likelihood(fused, fused.a_hat)
    v1 = heads_v1.likelihood(fused, fused.a_hat)
    assert set(v0) == set(v1)
    params = heads_v1.tracker_params()
    assert isinstance(params, TrackerParams)
    assert params.epsilon == 0.1 and params.lambda_g == 0.05


def test_scaffold_folds_into_place():
    assert features.action_index(MacroAction.SCAFFOLD) == features.action_index(MacroAction.PLACE)


def test_channel_absence_is_flagged():
    vec, flag = features.channel(None, 4)
    assert vec == [0.0, 0.0, 0.0, 0.0] and flag == 0.0
    vec, flag = features.channel((1.0, 2.0), 2)
    assert vec == [1.0, 2.0] and flag == 1.0
