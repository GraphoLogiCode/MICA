"""The B0 acceptance gate as data: the strict checks a capture must meet before D1 may
consume it.

The CLI in scripts/b0_gate.py only prints these; keeping the logic here makes it
importable and testable, so we can trust a PASS on a real capture means what it says.
A capture passes only if it is clean (the sync-report invariants) AND actually exercises
B0 — it must contain real, undistorted frames and real block events, not just a walk-around.
"""
from __future__ import annotations

import os

from ..contracts.manifest import CapturedSession
from .sync_report import build_sync_report


def _snapshot_check(session: CapturedSession, capture_dir: str) -> tuple[str, bool, str]:
    """Region snapshots are the ground truth D2 replays block events against. A recording
    that declares them (mod 0.0.3+) must actually contain at least the initial one; an
    older recording can't have any, so it passes with a not-D2-ready label instead of
    failing retroactively."""
    if session.manifest.snapshot_quiet_ticks <= 0:
        return ("region snapshots (D2)", True, "none declared - pre-0.0.3 capture, not D2-ready")
    snap_dir = os.path.join(capture_dir, session.manifest.session_id, "snapshots")
    count = len([n for n in os.listdir(snap_dir) if n.endswith(".json")]) if os.path.isdir(snap_dir) else 0
    return ("region snapshots (D2)", count >= 1, f"{count} snapshot files")


def gate_checks(session: CapturedSession, capture_dir: str | None = None) -> list[tuple[str, bool, str]]:
    """Return (name, passed, detail) for every gate criterion, in display order.

    capture_dir is where the session's sidecar folders (frames, snapshots) live; without
    it the on-disk checks are skipped — unit tests pass sessions that never touched disk.
    """
    report = build_sync_report(session)
    frames = [packet.client.pov_frame for packet in session.packets if packet.client.pov_frame is not None]
    event_count = sum(len(packet.server.block_events) for packet in session.packets)

    checks = [
        # A provisional count means the recording never closed cleanly (crash) — its
        # end may be missing with no way to tell, so the completeness check below
        # would compare the recording against itself. Hard reject.
        ("manifest finalized (clean stop)", not session.declared_is_provisional,
         "final event count present" if not session.declared_is_provisional
         else "count is provisional - recording may be truncated"),
        ("tick contiguity", report.ticks.contiguous, f"{report.ticks.count} ticks"),
        ("alignment bounded", report.alignment.within_bound,
         f"max {report.alignment.max_abs_ms:.0f} ms (singleplayer = one clock)"),
        ("no dropped events", report.block_events.complete,
         f"declared {report.block_events.declared} == present {report.block_events.count}"),
        ("snapshot precondition", report.actions_have_prior_moment, "every action has a prior tick"),
    ]
    for cov in report.coverage:
        checks.append((f"coverage: {cov.reader}", cov.satisfied,
                       "ok" if cov.satisfied else "MISSING " + ", ".join(cov.missing)))
    checks.append(("frames present", bool(frames), f"{len(frames)} frames"))
    checks.append(("frames undistorted (non-square)", any(f.width != f.height for f in frames),
                   f"{frames[0].width}x{frames[0].height}" if frames else "no frames"))
    checks.append(("block events exercised", event_count > 0, f"{event_count} events"))
    if capture_dir is not None:
        checks.append(_snapshot_check(session, capture_dir))
    return checks


def gate_passes(session: CapturedSession, capture_dir: str | None = None) -> bool:
    """True only if every gate criterion passes."""
    return all(passed for _, passed, _ in gate_checks(session, capture_dir))
