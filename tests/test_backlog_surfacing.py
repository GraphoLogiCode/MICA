"""R-4 (2026-07-18 rig review): the labeling tool must SEE parked sessions.
_surfaces_in_backlog decides which with-evidence sessions still appear in the
backlog — quarantined and gate-failed sessions surface as read-only cards
(previously they were filtered out, making the GUI's "can't label this one"
state unreachable for exactly the sessions it was written for)."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from after_game import _surfaces_in_backlog  # noqa: E402


def test_no_report_surfaces():
    # a crashed chain leaves evidence but no report — must stay visible (2026-07-11)
    assert _surfaces_in_backlog(None) is True


def test_awaiting_label_surfaces():
    assert _surfaces_in_backlog(
        {"awaiting_label": True, "evidence_ready": True, "verdict": None}) is True


def test_quarantined_unlabeled_surfaces_as_read_only():
    # the 2026-07-18 case: quarantined, no verdict — was invisible, must surface
    assert _surfaces_in_backlog(
        {"awaiting_label": False, "structure_quarantined": True,
         "gate_file_checks": "PASS", "verdict": None}) is True


def test_gate_failed_unlabeled_surfaces_as_read_only():
    assert _surfaces_in_backlog(
        {"awaiting_label": False, "structure_quarantined": False,
         "gate_file_checks": "FAIL", "verdict": None}) is True


def test_a_recorded_verdict_retires_the_session():
    # History owns labeled sessions — even a quarantine flag never resurfaces one
    assert _surfaces_in_backlog(
        {"awaiting_label": False, "structure_quarantined": True,
         "gate_file_checks": "PASS", "verdict": {"label": {"goal": "habitation"}}}) is False
    assert _surfaces_in_backlog(
        {"awaiting_label": False, "structure_quarantined": False,
         "gate_file_checks": "PASS", "verdict": {"label": {"goal": "defense"}}}) is False
