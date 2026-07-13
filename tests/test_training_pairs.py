"""The training-data builder must refuse a pair whose evidence is not strictly
pre-action (09-F1 — the named most-dangerous silent failure), and its split must hold
out whole sessions, never records."""
import json

import pytest

from mica.capture.sample_builds import pen_build
from mica.capture.synthetic import generate_session
from mica.contracts.serialize import evidence2d_to_dict, evidence3d_to_dict
from mica.data import training_pairs
from mica.data.source_b import FinishedLabel, build_pairs
from mica.perception.evidence2d import evidence_stream
from mica.perception.evidence3d import build_evidence3d
from mica.perception.voxel_replay import ReplayWorld, region_around_events


def _pair_lines():
    """Serialized Source-B pairs from one synthetic session — the exact on-disk form."""
    session = generate_session(pen_build())
    records = tuple(evidence_stream(session.packets))
    corrections = tuple(r for r in records if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    b2 = build_evidence3d(ReplayWorld(region_around_events(session), {}), corrections, events)
    label = FinishedLabel(goal="production", subtype="animal pen", score=0.9, fit=0.9,
                          comp=1.0, margin=0.5, runner_up="habitation", kept=True, reason="")
    return build_pairs([evidence2d_to_dict(r) for r in records],
                       [evidence3d_to_dict(r) for r in b2], label, "pen-test")


def _write(tmp_path, pairs):
    path = tmp_path / "pen-test.source_b.jsonl"
    path.write_text("\n".join(json.dumps(p) for p in pairs) + "\n", encoding="utf-8")
    return str(path)


def test_clean_pairs_load_and_carry_labels(tmp_path):
    samples = training_pairs._load_session_pairs(_write(tmp_path, _pair_lines()), "A", 1)
    assert samples, "the synthetic session must produce pairs"
    assert all(s.goal == "production" and s.mode == 1 and s.source == "A" for s in samples)
    # progress is the fraction built BEFORE the action: starts at zero, never decreases
    assert samples[0].progress == 0.0
    assert all(a.progress <= b.progress for a, b in zip(samples, samples[1:]))


def test_fuse_streams_refuses_mismatched_counts():
    # The 210003 lesson (2026-07-12): bare zip silently truncated a broken session
    # inside every eval for a week. fuse_streams must refuse loudly, naming the
    # session — and round-trip matched streams unchanged.
    from mica.contracts.b3 import fuse_streams

    pairs = _pair_lines()
    b1 = [p["b1"] for p in pairs]
    b2 = [p["b2"] for p in pairs]
    fused = fuse_streams(b1, b2, "pen-test")
    assert len(fused) == len(b2)                      # matched streams fuse fully
    with pytest.raises(ValueError, match="pen-test.*B1.*B2"):
        fuse_streams(b1, b2[:-1], "pen-test")         # one missing record -> loud


def test_contested_suffix_is_invisible_to_the_production_loader(tmp_path, monkeypatch):
    # The 2026-07-12 comparison experiment writes contested pairs to their own
    # suffix. The production loader must never even OPEN such a file — proven by
    # planting one full of garbage that would raise if read — while the opt-in
    # loader must be the one that reads it (and therefore raises on the garbage).
    monkeypatch.setattr(training_pairs, "_RAW", str(tmp_path))
    (tmp_path / "fabric-x.source_b_contested.jsonl").write_text(
        "NOT JSON — reading me is a bug in the default path\n", encoding="utf-8")
    assert training_pairs.load_source_b() == []          # never touched the file
    with pytest.raises(json.JSONDecodeError):
        training_pairs.load_source_b_contested()          # the opt-in loader does


def test_join_mismatch_refused(tmp_path):
    pairs = _pair_lines()
    pairs[0]["b2"]["tick"] += 1                       # no longer the same correction
    with pytest.raises(Exception):                    # fuse_dicts refuses the join
        training_pairs._load_session_pairs(_write(tmp_path, pairs), "A", 0)


def test_leaked_action_effects_refused(tmp_path):
    # The 09-F1 construction: make the FIRST event-carrying pair's evidence already
    # contain built cells — as if the action's own placement leaked into its snapshot.
    pairs = _pair_lines()
    first = next(p for p in pairs if p["b1"]["event_ids"])
    first["b2"]["global"]["built_count"] += len(first["b1"]["event_ids"])
    with pytest.raises(training_pairs.SnapshotViolation):
        training_pairs._load_session_pairs(_write(tmp_path, pairs), "B", None)


def test_split_holds_out_whole_sessions(tmp_path):
    samples = training_pairs._load_session_pairs(_write(tmp_path, _pair_lines()), "A", 0)
    # clone the session under other names so the split has something to divide
    cloned = []
    for name in ("s-a", "s-b", "s-c"):
        for s in samples:
            cloned.append(training_pairs.Sample(session=name, source="A", goal=s.goal,
                                                mode=s.mode, progress=s.progress,
                                                fused=s.fused))
    train, validation = training_pairs.split(cloned)
    train_sessions = {s.session for s in train}
    val_sessions = {s.session for s in validation}
    assert not train_sessions & val_sessions, "a session may never sit on both sides"
    # the pinned rule: the LAST session id per goal (sorted) is the holdout
    assert val_sessions == {"s-c"}


def test_vocab_reserves_unknown_slot(tmp_path):
    samples = training_pairs._load_session_pairs(_write(tmp_path, _pair_lines()), "A", 0)
    vocab = training_pairs.held_item_vocab(samples)
    assert vocab[0] == "<unk>"
    assert len(vocab) == len(set(vocab))
