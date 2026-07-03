"""Validate a B2 (Evidence3D) stream against its contract before D3 consumes it.

Checks every field's type and range, plus the cross-stream invariant that matters most:
each scored B1 correction has exactly one B2 record with the same tick and event ids —
the one-to-one join D3 fuses on. Returns plain problem strings; empty means safe.
"""
from __future__ import annotations

from ..contracts.b1 import GOALS, Evidence2D
from ..contracts.b2 import ROTATIONS, Evidence3D, GlobalStructure, PerGoalStructure
from ..contracts.goals import TAXONOMY


def check_record(record: Evidence3D) -> list[str]:
    """Every contract problem with a single record (empty list = valid)."""
    issues: list[str] = []
    if not isinstance(record.tick, int):
        issues.append(f"tick not an int: {record.tick!r}")
    if not record.scored:
        issues.append("v1 emits only scored records; found scored=False")
    if set(record.per_goal) != set(GOALS):
        issues.append(f"per_goal keys {sorted(record.per_goal)} != goal set")
    for goal, feats in record.per_goal.items():
        if not isinstance(feats, PerGoalStructure):
            issues.append(f"per_goal[{goal}] not a PerGoalStructure")
            continue
        if not 0.0 <= feats.comp <= 1.0:
            issues.append(f"comp({goal}) outside [0,1]: {feats.comp}")
        if not 0.0 <= feats.fit <= 1.0:
            issues.append(f"fit({goal}) outside [0,1]: {feats.fit}")
        if feats.edit_distance < 0:
            issues.append(f"edit_distance({goal}) negative: {feats.edit_distance}")
        if not -1.0 <= feats.delta_comp <= 1.0:
            issues.append(f"delta_comp({goal}) outside [-1,1]: {feats.delta_comp}")
        if feats.pose.rot not in ROTATIONS:
            issues.append(f"pose({goal}) has rotation {feats.pose.rot}, not a quarter turn")
        if feats.subtype not in TAXONOMY[goal]:
            issues.append(f"subtype({goal}) '{feats.subtype}' is not one of its taxonomy styles")
    g = record.global_feats
    if not isinstance(g, GlobalStructure):
        issues.append("global_feats not a GlobalStructure")
    else:
        if g.built_count < 0 or g.planar_runs < 0 or g.symmetry_support < 0:
            issues.append("global counts must be nonnegative")
        if not 0.0 <= g.symmetry <= 1.0:
            issues.append(f"symmetry outside [0,1]: {g.symmetry}")
        if (g.built_count == 0) is not (g.bbox is None):
            issues.append("bbox must exist exactly when something is built")
    return issues


def check_stream(records: list[Evidence3D]) -> list[str]:
    """Every contract problem across a whole B2 stream (empty list = ready for D3)."""
    issues: list[str] = []
    for index, record in enumerate(records):
        for problem in check_record(record):
            issues.append(f"record {index}: {problem}")
    consumed = [eid for r in records for eid in r.event_ids]
    if len(consumed) != len(set(consumed)):
        issues.append("a block event appears in more than one B2 record (single-consumption broken)")
    return issues


def check_join(b1_records: list[Evidence2D], b2_records: list[Evidence3D]) -> list[str]:
    """The D3 join must be one-to-one: every scored B1 correction pairs with exactly one
    B2 record carrying the same action tick and the same consumed event ids."""
    issues: list[str] = []
    scored = [r for r in b1_records if r.scored]
    if len(scored) != len(b2_records):
        issues.append(f"{len(scored)} scored B1 corrections but {len(b2_records)} B2 records")
    for b1, b2 in zip(scored, b2_records):
        if b1.tick_range[1] + 1 != b2.tick:
            issues.append(f"tick mismatch: B1 action at {b1.tick_range[1] + 1}, B2 says {b2.tick}")
        if tuple(b1.event_ids) != tuple(b2.event_ids):
            issues.append(f"event_ids mismatch at tick {b2.tick}: {b1.event_ids} vs {b2.event_ids}")
    return issues
