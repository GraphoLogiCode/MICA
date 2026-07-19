"""R-5 (2026-07-18 rig review): newest_capture must recognize captures by what
they ARE (fabric-*.jsonl + manifest sidecar), not by a blacklist of derived
suffixes — the blacklist missed rig_log.jsonl and a no-session after_game
invocation targeted "rig_log" as if it were a game."""
import os

from mica.capture.discovery import newest_capture


def _touch(directory, name, mtime):
    path = os.path.join(str(directory), name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{}\n")
    os.utime(path, (mtime, mtime))
    return path


def test_non_capture_jsonls_are_never_the_newest_capture(tmp_path):
    _touch(tmp_path, "fabric-20260701-000000.jsonl", 100)
    _touch(tmp_path, "fabric-20260701-000000.manifest.json", 100)
    # every one of these is NEWER and every one bit a real caller once
    _touch(tmp_path, "rig_log.jsonl", 500)
    _touch(tmp_path, "agent-MICA_AI.voiced.jsonl", 500)
    _touch(tmp_path, "gate_trace.jsonl", 500)
    _touch(tmp_path, "arm2_cache.jsonl", 500)
    _touch(tmp_path, "fabric-20260701-000000.evidence2d.jsonl", 500)
    found = newest_capture(str(tmp_path))
    assert found is not None
    assert os.path.basename(found) == "fabric-20260701-000000.jsonl"


def test_a_capture_without_its_manifest_is_not_a_capture(tmp_path):
    _touch(tmp_path, "fabric-20260701-000000.jsonl", 100)
    _touch(tmp_path, "fabric-20260701-000000.manifest.json", 100)
    _touch(tmp_path, "fabric-20260702-000000.jsonl", 200)   # newer, no manifest
    found = newest_capture(str(tmp_path))
    assert os.path.basename(found) == "fabric-20260701-000000.jsonl"


def test_newest_wins_and_empty_dir_is_none(tmp_path):
    assert newest_capture(str(tmp_path)) is None
    _touch(tmp_path, "fabric-20260701-000000.jsonl", 100)
    _touch(tmp_path, "fabric-20260701-000000.manifest.json", 100)
    _touch(tmp_path, "fabric-20260703-000000.jsonl", 300)
    _touch(tmp_path, "fabric-20260703-000000.manifest.json", 300)
    assert os.path.basename(newest_capture(str(tmp_path))) == "fabric-20260703-000000.jsonl"
