"""R-7 (2026-07-18 rig review): a failed run_d2 must not leave a fresh
evidence2d paired with stale siblings. On the 07-18 session, the offline regen
rewrote evidence2d and then crashed, leaving the quarantined LIVE evidence3d
sitting in the plain namespace as if it matched."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import after_game  # noqa: E402

_SID = "fabric-20260101-000000"


def _fake_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(after_game.session_store, "session_file",
                        lambda sid, suffix: os.path.join(str(tmp_path), sid + suffix))


def _touch(tmp_path, suffix):
    path = os.path.join(str(tmp_path), _SID + suffix)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{}\n")
    return path


def test_stale_siblings_are_shelved_and_evidence2d_is_kept(tmp_path, monkeypatch):
    _fake_layout(monkeypatch, tmp_path)
    kept = _touch(tmp_path, ".evidence2d.jsonl")          # fresh from run_d1
    for suffix in (".evidence3d.jsonl", ".fused.jsonl", ".belief.jsonl"):
        _touch(tmp_path, suffix)
    after_game._shelve_stale_evidence(_SID)
    assert os.path.exists(kept)
    for suffix in (".evidence3d.jsonl", ".fused.jsonl", ".belief.jsonl"):
        assert not os.path.exists(os.path.join(str(tmp_path), _SID + suffix))
        assert os.path.exists(os.path.join(
            str(tmp_path), _SID + suffix[:-len(".jsonl")] + ".stale.jsonl"))


def test_missing_siblings_and_repeat_crashes_are_fine(tmp_path, monkeypatch):
    # offline-only sessions have no fused/belief; a second crash re-shelves
    # over the previous .stale copy instead of failing
    _fake_layout(monkeypatch, tmp_path)
    _touch(tmp_path, ".evidence3d.jsonl")
    _touch(tmp_path, ".evidence3d.stale.jsonl")           # leftover from crash #1
    after_game._shelve_stale_evidence(_SID)
    assert not os.path.exists(os.path.join(str(tmp_path), _SID + ".evidence3d.jsonl"))
    assert os.path.exists(os.path.join(str(tmp_path), _SID + ".evidence3d.stale.jsonl"))
