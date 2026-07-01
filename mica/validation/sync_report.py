"""Checks whether one recording is clean enough to trust.

Clean means: no missing moments, the clocks line up, no block changes were lost,
and it has the fields the later steps need. Fully proving the structure can be
rebuilt is done in a later step; this is the basic check that makes that possible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..contracts.b0 import BlockEvent, ObservationPacket
from ..contracts.manifest import CapturedSession
from ..contracts.timebase import DEFAULT_ALIGNMENT_BOUND_MS, alignment_residual_ms
from .coverage import CoverageResult, evaluate_coverage, gating_gaps


@dataclass(frozen=True)
class TickSummary:
    """How completely the recording covers every moment."""

    count: int
    first: int
    last: int
    contiguous: bool
    missing: int


@dataclass(frozen=True)
class AlignmentSummary:
    """How far the player's clock drifts from the server's clock."""

    max_abs_ms: float
    mean_abs_ms: float
    bound_ms: float
    within_bound: bool


@dataclass(frozen=True)
class BlockEventSummary:
    """Whether any block change was lost."""

    count: int  # block changes actually in the recording
    declared: int  # how many the recorder said there were
    ids_unique: bool
    complete: bool  # the changes are numbered 0,1,...,declared-1 with none missing
    dropped: int  # how many are missing


@dataclass(frozen=True)
class SyncReport:
    """The full result for one recording: each check, plus an overall pass or fail."""

    session_id: str
    ticks: TickSummary
    alignment: AlignmentSummary
    block_events: BlockEventSummary
    actions_have_prior_moment: bool
    coverage: tuple[CoverageResult, ...]
    gating_gaps: tuple[str, ...]
    passed: bool


def _summarize_ticks(packets: tuple[ObservationPacket, ...]) -> TickSummary:
    """Check the moments are complete. A gap means a missing moment, so some action has nothing 'before' it."""
    ticks = sorted(packet.tick for packet in packets)
    first, last = ticks[0], ticks[-1]
    span = last - first + 1
    missing = span - len(set(ticks))
    return TickSummary(
        count=len(ticks),
        first=first,
        last=last,
        contiguous=(missing == 0 and len(set(ticks)) == len(ticks)),
        missing=missing,
    )


def _summarize_alignment(
    packets: tuple[ObservationPacket, ...], bound_ms: float
) -> AlignmentSummary:
    """Measure how far the player's clock drifts from the server's clock for the same moment."""
    residuals = [
        abs(alignment_residual_ms(packet.wallclock_ms, packet.client.capture_wallclock_ms))
        for packet in packets
    ]
    max_abs = max(residuals)
    mean_abs = sum(residuals) / len(residuals)
    return AlignmentSummary(
        max_abs_ms=max_abs,
        mean_abs_ms=mean_abs,
        bound_ms=bound_ms,
        within_bound=(max_abs <= bound_ms),
    )


def _collect_block_events(packets: tuple[ObservationPacket, ...]) -> list[BlockEvent]:
    """Every block change, in time order."""
    ordered = sorted(packets, key=lambda packet: packet.tick)
    return [event for packet in ordered for event in packet.server.block_events]


def _summarize_block_events(
    packets: tuple[ObservationPacket, ...], declared: int
) -> BlockEventSummary:
    """Look for a lost block change.

    The changes should be numbered 0,1,...,declared-1. Checking against the
    recorder's declared total catches a loss anywhere — start, middle, or end.
    """
    ids = [event.event_id for event in _collect_block_events(packets)]
    present = set(ids)
    expected = set(range(declared))
    unique = len(ids) == len(present)
    complete = unique and present == expected
    dropped = len(expected - present)   # expected ids that never showed up (ignores any out-of-range id)
    return BlockEventSummary(
        count=len(ids),
        declared=declared,
        ids_unique=unique,
        complete=complete,
        dropped=dropped,
    )


def _actions_have_prior_moment(packets: tuple[ObservationPacket, ...]) -> bool:
    """Check every action has an earlier moment to read its 'before' state from."""
    present = {packet.tick for packet in packets}
    action_ticks = {packet.tick for packet in packets if packet.server.block_events}
    return all((tick - 1) in present for tick in action_ticks)


def _empty_report(session: CapturedSession, bound_ms: float) -> SyncReport:
    # A recording with no moments can't be checked, so it fails plainly instead of crashing.
    declared = session.declared_event_count
    return SyncReport(
        session_id=session.manifest.session_id,
        ticks=TickSummary(count=0, first=0, last=0, contiguous=False, missing=0),
        alignment=AlignmentSummary(max_abs_ms=0.0, mean_abs_ms=0.0, bound_ms=bound_ms, within_bound=False),
        block_events=BlockEventSummary(
            count=0, declared=declared, ids_unique=True, complete=(declared == 0), dropped=declared
        ),
        actions_have_prior_moment=False,
        coverage=evaluate_coverage(session.packets),
        gating_gaps=gating_gaps(session.packets),
        passed=False,
    )


def build_sync_report(
    session: CapturedSession, bound_ms: float = DEFAULT_ALIGNMENT_BOUND_MS
) -> SyncReport:
    """Run every check and decide pass or fail.

    The screen image is reported but not required — it comes from the real game
    capture, not from here.
    """
    if not session.packets:
        return _empty_report(session, bound_ms)
    ticks = _summarize_ticks(session.packets)
    alignment = _summarize_alignment(session.packets, bound_ms)
    block_events = _summarize_block_events(session.packets, session.declared_event_count)
    prior_moment_ok = _actions_have_prior_moment(session.packets)
    coverage = evaluate_coverage(session.packets)
    gaps = gating_gaps(session.packets)
    passed = (
        ticks.contiguous
        and alignment.within_bound
        and block_events.complete
        and prior_moment_ok
        and not gaps
    )
    return SyncReport(
        session_id=session.manifest.session_id,
        ticks=ticks,
        alignment=alignment,
        block_events=block_events,
        actions_have_prior_moment=prior_moment_ok,
        coverage=coverage,
        gating_gaps=gaps,
        passed=passed,
    )


def report_to_dict(report: SyncReport) -> dict:
    """Turn the result into plain data so it can be saved as a file."""
    return {
        "session_id": report.session_id,
        "passed": report.passed,
        "ticks": vars(report.ticks),
        "alignment": vars(report.alignment),
        "block_events": vars(report.block_events),
        "actions_have_prior_moment": report.actions_have_prior_moment,
        "coverage": [vars(result) | {"missing": list(result.missing)} for result in report.coverage],
        "gating_gaps": list(report.gating_gaps),
    }


def write_sync_report(report: SyncReport, path: str) -> None:
    """Save the result to a file; one command rebuilds it."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report_to_dict(report), handle, indent=2)
