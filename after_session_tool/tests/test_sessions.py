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

def test_subtype_presets_are_the_template_backed_v3_styles():
    assert "crop_farm" in sessions.subtype_presets("production")
    assert sessions.subtype_presets("nonsense") == ()
    for goal in sessions.GOALS:
        assert isinstance(sessions.subtype_presets(goal), tuple)
        assert sessions.subtype_presets(goal), f"{goal} has no matcher-confirmable style"


def test_roadmap_presets_are_the_definitions_only_v3_styles():
    # the two chip groups split the v3 taxonomy exactly: template-backed + the rest
    for goal in sessions.GOALS:
        backed, rest = sessions.subtype_presets(goal), sessions.roadmap_presets(goal)
        assert not set(backed) & set(rest)
        assert set(backed) | set(rest) == set(sessions.TAXONOMY[goal])
    assert "mining_excavation" in sessions.roadmap_presets("production")
    assert sessions.roadmap_presets("nonsense") == ()


# ---- batch queue validation ---------------------------------------------------

def test_batch_problem_accepts_a_good_queue():
    queue = [{"session": "fabric-a", "goal": "defense", "subtype": "watchtower"},
             {"session": "fabric-b", "goal": "production", "subtype": "crop field"}]
    assert sessions.batch_problem(queue) is None


def test_batch_problem_names_whats_wrong():
    assert sessions.batch_problem([]) == "the queue is empty"
    assert "goal" in sessions.batch_problem(
        [{"session": "fabric-a", "goal": "castle", "subtype": "keep"}])
    assert "subtype" in sessions.batch_problem(
        [{"session": "fabric-a", "goal": "defense", "subtype": "  "}])
    assert "twice" in sessions.batch_problem(
        [{"session": "fabric-a", "goal": "defense", "subtype": "wall"},
         {"session": "fabric-a", "goal": "defense", "subtype": "tower"}])
    assert "session id" in sessions.batch_problem([{"goal": "defense", "subtype": "wall"}])


# ---- cascade checklist: who feeds a retrain, who is excluded and why ----------

def test_checklist_sorts_included_from_excluded():
    labels = {"fabric-in": {}, "fabric-contested": {}, "fabric-discarded": {},
              "scripted-x": {}}
    rows = {
        "fabric-in": {"label": {"kept": True, "goal": "defense", "subtype": "wall"},
                      "agrees_with_builder": True, "pairs": 42},
        "fabric-contested": {"label": {"kept": True, "goal": "habitation", "subtype": "cabin"},
                             "agrees_with_builder": False},
        "fabric-discarded": {"label": {"kept": False, "reason": "below threshold"},
                             "agrees_with_builder": False},
    }
    states = {"fabric-unlabeled": "ready", "fabric-quar": "quarantined"}
    got = sessions.checklist_from(labels, rows, states, ["fabric-skipped"])
    assert got["included"] == [{"sid": "fabric-in", "label": "defense/wall", "pairs": 42}]
    reasons = {e["sid"]: e["reason"] for e in got["excluded"]}
    assert "CONTESTS" in reasons["fabric-contested"]
    # contested pairs JOIN production training since the 2026-07-19 D3
    # amendment (the v2 comparison win) -- the checklist must say the
    # builder's word wins, not that the pairs are withheld
    assert "word wins" in reasons["fabric-contested"]
    assert "below threshold" in reasons["fabric-discarded"]
    assert "no label yet" in reasons["fabric-unlabeled"]
    assert "quarantined" in reasons["fabric-quar"]
    assert "skipped by you" in reasons["fabric-skipped"]
    assert "scripted-x" not in reasons                  # scripted labels aren't sessions


def test_checklist_flags_a_label_the_matcher_never_saw():
    got = sessions.checklist_from({"fabric-new": {}}, {}, {}, [])
    assert got["included"] == []
    assert "matcher not run" in got["excluded"][0]["reason"]


# ---- model status: present, dated, one line of facts --------------------------

def test_model_status_reads_present_and_missing(tmp_path):
    (tmp_path / "heads_v1.npz").write_bytes(b"x")
    (tmp_path / "heads_v1.json").write_text(json.dumps(
        {"trained": "2026-07-08 00:48", "temperature_delib": 1.4,
         "temperature_heur": 1.4, "epsilon": 0.01, "lambda_g": 0.02}), encoding="utf-8")
    (tmp_path / "gate_v1.json").write_text(json.dumps(
        {"thresholds": {"theta_1": 0.35},
         "fsm": {"theta_suggest": 0.3, "theta_place": 0.4}}), encoding="utf-8")
    rows = {r["name"]: r for r in sessions.model_status(str(tmp_path))}
    heads = rows["belief heads (heads_v1)"]
    assert heads["present"] and heads["trained"] == "2026-07-08 00:48"
    assert "1.4" in heads["facts"]
    gate = rows["commit gate (gate_v1)"]
    assert gate["present"] and "0.35" in gate["facts"]
    decoder = rows["action decoder (decoder_v1)"]
    assert not decoder["present"] and decoder["facts"] == "missing"
    bank = rows["'before' bank (pre_cascade_a)"]
    assert not bank["present"]


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
