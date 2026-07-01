"""The B0 acceptance gate as data: the strict checks a capture must meet before D1 may
consume it.

The CLI in scripts/b0_gate.py only prints these; keeping the logic here makes it
importable and testable, so we can trust a PASS on a real capture means what it says.
A capture passes only if it is clean (the sync-report invariants) AND actually exercises
B0 — it must contain real, undistorted frames and real block events, not just a walk-around.
"""
from __future__ import annotations

from ..contracts.manifest import CapturedSession
from .sync_report import build_sync_report


def gate_checks(session: CapturedSession) -> list[tuple[str, bool, str]]:
    """Return (name, passed, detail) for every gate criterion, in display order."""
    report = build_sync_report(session)
    frames = [packet.client.pov_frame for packet in session.packets if packet.client.pov_frame is not None]
    event_count = sum(len(packet.server.block_events) for packet in session.packets)

    checks = [
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
    return checks


def gate_passes(session: CapturedSession) -> bool:
    """True only if every gate criterion passes."""
    return all(passed for _, passed, _ in gate_checks(session))
