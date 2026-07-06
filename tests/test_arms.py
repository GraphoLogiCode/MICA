"""The comparison arms' structural guarantees: Arm 2's answer mapping and cache,
Arm 1's export/inference agreement, and the pivot stitcher's truth seam."""
import json

import pytest

np = pytest.importorskip("numpy")

from mica.capture.pivot_builds import pivot_build
from mica.capture.sample_builds import pen_build
from mica.capture.scripted_goals import plan_variants
from mica.capture.synthetic import generate_session
from mica.contracts.b1 import GOALS
from mica.contracts.b3 import fuse
from mica.intent import arm1, arm2 as arm2_module
from mica.intent.arm2 import Arm2, _mapping
from mica.perception.evidence2d import evidence_stream
from mica.perception.evidence3d import build_evidence3d
from mica.perception.voxel_replay import ReplayWorld, region_around_events


def _fused_record():
    session = generate_session(pen_build())
    corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    b2 = build_evidence3d(ReplayWorld(region_around_events(session), {}), corrections, events)
    return next(fuse(a, b) for a, b in zip(corrections, b2) if a.event_ids)


# ------------------------------------------------------------------ arm 2 mapping

def test_mapping_is_a_distribution():
    dist = _mapping("habitation", 0.8)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert dist["habitation"] == pytest.approx(0.8 + 0.2 / 5)
    assert dist["defense"] == pytest.approx(0.2 / 5)


def test_unknown_answer_degrades_to_uniform():
    for bad in (None, "castle", ""):
        dist = _mapping(bad, 0.9)
        assert all(abs(p - 0.2) < 1e-9 for p in dist.values())


def test_confidence_is_clamped():
    assert abs(sum(_mapping("defense", 7.0).values()) - 1.0) < 1e-9
    assert _mapping("defense", -3.0)["defense"] == pytest.approx(0.2)


def test_parse_tolerates_non_json():
    assert Arm2._parse('{"category": "production", "confidence": 0.7}') == ("production", 0.7)
    assert Arm2._parse("I think this is a defense structure") == ("defense", 0.5)
    assert Arm2._parse("no idea whatsoever") == (None, 0.0)
    assert Arm2._parse(None) == (None, 0.0)


def test_cache_makes_reruns_free(tmp_path, monkeypatch):
    calls = []

    def fake_ask(images, prompt, json_format=False):
        calls.append(prompt)
        return '{"category": "production", "confidence": 0.6}'

    monkeypatch.setattr(arm2_module.vlm, "ask", fake_ask)
    fused = _fused_record()
    cache = str(tmp_path / "cache.jsonl")

    reader = Arm2(cache_path=cache)
    first = reader.read(fused)
    second = reader.read(fused)                 # same scene -> cache, no new call
    assert first == second and len(calls) == 1 and reader.cache_hits == 1

    rerun = Arm2(cache_path=cache)              # fresh instance, same file
    assert rerun.read(fused) == first and len(calls) == 1


# --------------------------------------------------------------- arm 1 agreement

def test_arm1_numpy_matches_torch(tmp_path, monkeypatch):
    th = pytest.importorskip("torch")

    fused = _fused_record()
    vocab = ("<unk>", fused.state_feats.held_item)
    held_idx, dense = arm1.input_vector(fused, vocab)

    th.manual_seed(3)
    embed = th.nn.Embedding(len(vocab), 8)
    net = th.nn.Sequential(th.nn.Linear(8 + len(dense), 16), th.nn.ReLU(),
                           th.nn.Linear(16, 8), th.nn.ReLU(), th.nn.Linear(8, len(GOALS)))
    with th.no_grad():
        x = th.cat([embed(th.tensor([held_idx]))[0], th.tensor(dense)])
        expected = th.softmax(net(x) / 2.0, dim=-1).tolist()

    def linear(layer):
        return layer.weight.detach().numpy().T, layer.bias.detach().numpy()

    arrays = {"held_embed": embed.weight.detach().numpy()}
    for name, layer in (("1", net[0]), ("2", net[2]), ("out", net[4])):
        arrays[f"w_{name}"], arrays[f"b_{name}"] = linear(layer)
    np.savez(str(tmp_path / "arm1.npz"), **arrays)
    (tmp_path / "arm1.json").write_text(json.dumps(
        {"goals": list(GOALS), "vocab": list(vocab), "temperature": 2.0}), encoding="utf-8")
    monkeypatch.setattr(arm1, "_WEIGHTS", str(tmp_path / "arm1.npz"))
    monkeypatch.setattr(arm1, "_META", str(tmp_path / "arm1.json"))
    monkeypatch.setattr(arm1, "_cache", None)

    dist = arm1.distribution(fused)
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    for goal, want in zip(GOALS, expected):
        assert dist[goal] == pytest.approx(want, abs=1e-5)


# ------------------------------------------------------------------ pivot builds

def test_pivot_stitches_two_goals_in_order():
    plan_a = plan_variants("habitation", 1, 33)[0]
    plan_b = plan_variants("defense", 1, 44)[0]
    build, truth = pivot_build(plan_a, plan_b)
    assert truth["goal_a"] == "habitation" and truth["goal_b"] == "defense"
    ticks = [p.tick for p in build.placements]
    assert ticks == sorted(ticks), "plan B must start strictly after plan A ends"
    before = [p for p in build.placements if p.tick < truth["seam_tick"]]
    after = [p for p in build.placements if p.tick >= truth["seam_tick"]]
    assert before and after
    # the two structures never share cells (plan B is shifted ground)
    cells_a = {p.pos for p in before}
    cells_b = {p.pos for p in after}
    assert not cells_a & cells_b
