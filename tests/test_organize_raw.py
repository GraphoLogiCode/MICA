"""organize_raw: strays find their session folder; mixed traces split by run."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import organize_raw  # noqa: E402

SID_A = "fabric-20260701-100000"
SID_B = "fabric-20260701-120000"


def _make_session(root, sid, start_ms):
    day = f"{sid[7:11]}-{sid[11:13]}-{sid[13:15]}"
    directory = root / day / sid
    directory.mkdir(parents=True)
    (directory / f"{sid}.jsonl").write_text("", encoding="utf-8")
    (directory / f"{sid}.manifest.json").write_text(
        json.dumps({"session_id": sid, "session_start_ms": start_ms}), encoding="utf-8")
    return directory


def _sweep(ts_ms, session=None):
    row = {"ts": ts_ms, "tick": 1, "pose": [0, 0, 0, 0, 0], "cells": [[0, 0, 0, "stone"]]}
    if session:
        row["session"] = session
    return json.dumps(row) + "\n"


def test_tagged_scan_moves_to_its_session(tmp_path):
    _make_session(tmp_path, SID_A, 1_000_000)
    scan = tmp_path / "agent-X.scan.9999.jsonl"
    scan.write_text(_sweep(2_000_000, session=SID_A), encoding="utf-8")
    plan = organize_raw.organize_scans(str(tmp_path), apply=True)
    assert any(line.startswith("move") for line in plan)
    moved = tmp_path / "2026-07-01" / SID_A / "agent-X.scan.9999.jsonl"
    assert moved.exists() and not scan.exists()


def test_untagged_scan_pairs_by_wallclock(tmp_path):
    # Session A starts at t=1000s, B at t=2000s; a scan first sweeping at t=2500s
    # belongs to B — the latest session that started before it.
    _make_session(tmp_path, SID_A, 1_000_000)
    _make_session(tmp_path, SID_B, 2_000_000)
    scan = tmp_path / "agent-X.scan.1234.jsonl"
    scan.write_text(_sweep(2_500_000), encoding="utf-8")
    organize_raw.organize_scans(str(tmp_path), apply=True)
    assert (tmp_path / "2026-07-01" / SID_B / "agent-X.scan.1234.jsonl").exists()


def test_scan_before_any_session_is_left_alone(tmp_path):
    _make_session(tmp_path, SID_A, 5_000_000)
    scan = tmp_path / "agent-X.scan.1.jsonl"
    scan.write_text(_sweep(1_000), encoding="utf-8")
    plan = organize_raw.organize_scans(str(tmp_path), apply=True)
    assert scan.exists()
    assert any(line.startswith("skip") for line in plan)


def test_session_scoped_call_ignores_other_sessions(tmp_path):
    _make_session(tmp_path, SID_A, 1_000_000)
    _make_session(tmp_path, SID_B, 2_000_000)
    for name, sid in (("agent-X.scan.10.jsonl", SID_A), ("agent-X.scan.20.jsonl", SID_B)):
        (tmp_path / name).write_text(_sweep(3_000_000, session=sid), encoding="utf-8")
    organize_raw.organize_scans(str(tmp_path), apply=True, only_session=SID_A)
    assert (tmp_path / "2026-07-01" / SID_A / "agent-X.scan.10.jsonl").exists()
    assert (tmp_path / "agent-X.scan.20.jsonl").exists()   # B's untouched


def _trace_row(k, tick):
    return json.dumps({"k": k, "tick": tick, "chosen_state": "observe",
                       "candidate_state": "observe", "reason": "r",
                       "K_commit": 0, "committed_actions": []}) + "\n"


def test_multirun_trace_splits_and_keeps_final_run(tmp_path):
    directory = _make_session(tmp_path, SID_A, 1_000_000)
    trace = directory / f"{SID_A}.gate_trace.jsonl"
    run1 = [_trace_row(1, 10), _trace_row(2, 20)]
    run2 = [_trace_row(1, 500), _trace_row(2, 510), _trace_row(3, 520)]
    trace.write_text("".join(run1 + run2), encoding="utf-8")
    plan = organize_raw.organize_traces(str(tmp_path), apply=True)
    assert any(line.startswith("split") for line in plan)
    kept = trace.read_text(encoding="utf-8").splitlines()
    assert len(kept) == 3 and json.loads(kept[0])["tick"] == 500
    banked = directory / "previous_runs" / "run-01" / f"{SID_A}.gate_trace.jsonl"
    assert len(banked.read_text(encoding="utf-8").splitlines()) == 2


def test_single_run_trace_is_untouched(tmp_path):
    directory = _make_session(tmp_path, SID_A, 1_000_000)
    trace = directory / f"{SID_A}.gate_trace.jsonl"
    content = _trace_row(1, 10) + _trace_row(2, 20)
    trace.write_text(content, encoding="utf-8")
    assert organize_raw.organize_traces(str(tmp_path), apply=True) == []
    assert trace.read_text(encoding="utf-8") == content
