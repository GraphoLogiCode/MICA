"""R-1 (2026-07-18 rig review): only run_d2's own FRESH replay verdict may stamp
a session STRUCTURE QUARANTINE. A run_d2 crash after a clean replay (or before
any replay) must read as a retryable chain failure — on 2026-07-18 a
GPU-contention crash was mislabeled quarantine and parked a clean session."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from after_game import _fresh_replay_quarantine  # noqa: E402


def _report(tmp_path, payload):
    path = os.path.join(str(tmp_path), "x.voxel_replay_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def test_fresh_quarantined_report_is_a_quarantine(tmp_path):
    path = _report(tmp_path, {"quarantined": True})
    assert _fresh_replay_quarantine(path, since=time.time() - 60) is True


def test_fresh_clean_report_is_not_a_quarantine(tmp_path):
    # the 2026-07-18 case: the replay PASSED, then run_d2 crashed in the h3d
    # stage — the exit code alone must not overrule the replay's own verdict
    path = _report(tmp_path, {"quarantined": False})
    assert _fresh_replay_quarantine(path, since=time.time() - 60) is False


def test_stale_report_is_unknown_not_a_quarantine(tmp_path):
    # the report predates this run_d2 invocation, so its verdict belongs to an
    # earlier run; this run's verdict is unknown — no quarantine stamp
    path = _report(tmp_path, {"quarantined": True})
    assert _fresh_replay_quarantine(path, since=time.time() + 60) is False


def test_missing_report_is_not_a_quarantine(tmp_path):
    missing = os.path.join(str(tmp_path), "never_written.json")
    assert _fresh_replay_quarantine(missing, since=0.0) is False
