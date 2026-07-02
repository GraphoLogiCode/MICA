"""Validate a B1 (Evidence2D) stream against its contract before D3 consumes it.

Checks every field's type and range, plus the cross-record invariants (single consumption,
a constant h2d width, pixel channels filled-or-None together). Returns a list of plain
problem strings; an empty list means the stream is well-formed and safe to feed D3.

Scope: this validates the records' own consistency. The snapshot rule — each evidence window
ending strictly before the *actual* action tick — needs the source session, so it is checked
in tests/test_d1.py, not here.
"""
from __future__ import annotations

from ..contracts.b1 import GOALS, Evidence2D, Focus, MacroAction, StateFeats

_BUILD = (MacroAction.PLACE, MacroAction.BREAK)


def check_record(record: Evidence2D, h2d_dim: int | None = None) -> list[str]:
    """Every contract problem with a single record (empty list = valid)."""
    issues: list[str] = []
    t0, t1 = record.tick_range
    if not (isinstance(t0, int) and isinstance(t1, int) and t0 <= t1):
        issues.append(f"tick_range not ordered ints: {record.tick_range}")
    if not isinstance(record.a_hat, MacroAction):
        issues.append(f"a_hat not a MacroAction: {record.a_hat!r}")
    if not 0.0 <= record.a_hat_conf <= 1.0:
        issues.append(f"a_hat_conf outside [0,1]: {record.a_hat_conf}")
    if record.idle is not (record.a_hat is MacroAction.IDLE):
        issues.append(f"idle flag disagrees with a_hat ({record.a_hat.value})")
    if record.scored:
        # A scored PLACE/BREAK consumed real events; anything else consumed none.
        if (record.a_hat in _BUILD) is not bool(record.event_ids):
            issues.append(f"event_ids/a_hat mismatch: {record.a_hat.value} ids={record.event_ids}")
    elif record.event_ids:
        # Context records are read, never scored — consuming an event here would let
        # the same event count twice (once here, once at its correction).
        issues.append(f"unscored context record consumes events: ids={record.event_ids}")

    sf = record.state_feats
    if not isinstance(sf, StateFeats):
        issues.append("state_feats not a StateFeats")
    else:
        if not isinstance(sf.held_item, str):
            issues.append("state_feats.held_item not a str")
        if not (isinstance(sf.hotbar, tuple) and all(isinstance(x, str) for x in sf.hotbar)):
            issues.append("state_feats.hotbar not a tuple[str]")
        if len(sf.pos_delta) != 3:
            issues.append(f"state_feats.pos_delta not length 3: {sf.pos_delta}")
        if not all(isinstance(a, str) for a in sf.recent_actions):
            issues.append("state_feats.recent_actions not a tuple[str]")

    if not isinstance(record.focus, Focus) or record.focus.dwell_ticks < 0:
        issues.append(f"focus invalid: {record.focus}")

    if (record.h2d is None) is not (record.s_goal is None):
        issues.append("h2d and s_goal must be filled or None together")
    if record.s_goal is not None:
        if len(record.s_goal) != len(GOALS):
            issues.append(f"s_goal length {len(record.s_goal)} != |G|={len(GOALS)}")
        if any(not -1.001 <= s <= 1.001 for s in record.s_goal):
            issues.append("s_goal has a value outside [-1, 1]")
    if record.h2d is not None:
        if not record.h2d:
            issues.append("h2d is empty")
        elif h2d_dim is not None and len(record.h2d) != h2d_dim:
            issues.append(f"h2d length {len(record.h2d)} != {h2d_dim}")
    return issues


def check_stream(records: list[Evidence2D]) -> list[str]:
    """Every contract problem across a whole B1 stream (empty list = ready for D3)."""
    issues: list[str] = []
    h2d_dims = {len(r.h2d) for r in records if r.h2d is not None}
    if len(h2d_dims) > 1:
        issues.append(f"h2d width is not constant across the stream: {sorted(h2d_dims)}")
    h2d_dim = next(iter(h2d_dims)) if len(h2d_dims) == 1 else None

    for index, record in enumerate(records):
        for problem in check_record(record, h2d_dim):
            issues.append(f"record {index}: {problem}")

    consumed = [eid for r in records for eid in r.event_ids]
    if len(consumed) != len(set(consumed)):
        issues.append("a block event was consumed by more than one record (single-consumption broken)")
    return issues
