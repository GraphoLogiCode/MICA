"""The after-session tool's pure logic — state derivation, inbox filtering, taxonomy
presets, and the skip store. These take plain data (or a tmp file) and need neither
the pipeline nor a running server."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sessions


# ---- derive_state: the one word each card shows -----------------------------

def test_state_ready_only_when_evidence_and_awaiting_and_no_verdict():
    report = {"evidence_ready": True, "awaiting_label": True, "verdict": None,
              "gate_file_checks": "PASS", "structure_quarantined": False}
    assert sessions.derive_state(report, has_evidence=True, job=None) == "ready"
    # missing evidence on disk keeps it out of "ready"
    assert sessions.derive_state(report, has_evidence=False, job=None) == "processing"


def test_state_labeled_when_verdict_present():
    report = {"evidence_ready": True, "awaiting_label": False,
              "verdict": {"agrees_with_builder": True}}
    assert sessions.derive_state(report, has_evidence=True, job=None) == "labeled"


def test_state_running_job_beats_everything():
    report = {"evidence_ready": True, "awaiting_label": True, "verdict": None}
    assert sessions.derive_state(report, has_evidence=True, job="running") == "labeling"


def test_state_quarantine_and_gate_fail():
    assert sessions.derive_state({"structure_quarantined": True}, True, None) == "quarantined"
    assert sessions.derive_state({"gate_file_checks": "FAIL"}, True, None) == "gate failed"


def test_state_no_report_no_evidence_is_unprocessed():
    # honest: a capture with no report and no evidence is NOT being worked on
    assert sessions.derive_state(None, has_evidence=False, job=None) == "unprocessed"


def test_state_no_report_with_evidence_is_processing():
    # evidence exists but no report yet = a chain genuinely mid-run
    assert sessions.derive_state(None, has_evidence=True, job=None) == "processing"


# ---- inbox filtering --------------------------------------------------------

def test_inbox_excludes_skipped():
    needs = ["fabric-a", "fabric-b", "fabric-c"]
    assert sessions.inbox_from(needs, ["fabric-b"]) == ["fabric-a", "fabric-c"]
    assert sessions.inbox_from(needs, []) == needs
    assert sessions.inbox_from([], ["fabric-b"]) == []


# ---- taxonomy presets -------------------------------------------------------

def test_subtype_presets_match_taxonomy():
    assert "crop field" in sessions.subtype_presets("production")
    assert sessions.subtype_presets("nonsense") == ()
    for goal in sessions.GOALS:
        assert isinstance(sessions.subtype_presets(goal), tuple)


# ---- pov_frame: highest-tick, not lexical -----------------------------------

def test_pov_frame_picks_highest_tick(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    frames.mkdir()
    for name in ("0.png", "999.png", "10266.png", "notaframe.txt"):
        (frames / name).write_bytes(b"x")
    monkeypatch.setattr(sessions, "frames_dir", lambda sid: str(frames))
    # lexically "999.png" > "10266.png"; the integer compare must win
    assert sessions.pov_frame("any").endswith("10266.png")


def test_pov_frame_none_when_no_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "frames_dir", lambda sid: str(tmp_path / "missing"))
    assert sessions.pov_frame("any") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(sessions, "frames_dir", lambda sid: str(empty))
    assert sessions.pov_frame("any") is None


# ---- skip store roundtrip ---------------------------------------------------

def test_skip_unskip_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions, "_SKIPPED", str(tmp_path / "skipped.json"))
    assert sessions.load_skipped() == []
    sessions.skip("fabric-x")
    sessions.skip("fabric-x")                    # idempotent
    sessions.skip("fabric-y")
    assert sorted(sessions.load_skipped()) == ["fabric-x", "fabric-y"]
    sessions.unskip("fabric-x")
    assert sessions.load_skipped() == ["fabric-y"]
    # persisted as a plain JSON list
    assert json.loads((tmp_path / "skipped.json").read_text(encoding="utf-8")) == ["fabric-y"]
