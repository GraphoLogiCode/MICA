"""The suggestion-acceptance signal (D9 §3): did the human do the suggested thing?

An advisory session surfaces suggestions ("Shall I help? I could add ..."); this
module measures, per voiced suggestion, whether the human then placed the suggested
block themselves — the label-free appropriateness signal the acting layer trains on,
and a behavioral ground truth for intention recognition (the human confirming the
inferred goal with their hands).

Three inputs, all already banked per session:
  - the gate trace (one row per read; rows carry `proposal_first` since 2026-07-17)
  - the body's voicing log (`agent-<NAME>.voiced.jsonl` — the 30 s throttle lives
    body-side, so only the body knows which suggestions actually reached the human)
  - the capture's human PLACE events (A7-filtered by actor, as everywhere)

The CONTROL group is the confound killer: reads where a proposal existed but was
never voiced give the base rate at which the human does the proposed thing anyway.
The reportable number is the LIFT: P(followed | voiced) - P(followed | unvoiced).

Every constant below is pre-registered in D9 §3. Changing one changes what every
banked acceptance number means: bump the version, never edit silently.
"""
from __future__ import annotations

from dataclasses import dataclass

# v2 (2026-07-18): two implementation-bug fixes after the first real session, no
# semantic change to the pre-registered constants. (1) Block names are compared
# namespace-blind — the capture writes "minecraft:spruce_planks" while proposals
# carry "spruce_planks", so v1 could never score a follow on real data. (2) A
# voicing pairs with the latest matching read AT OR BEFORE it (a voicing cannot
# come from a read that hadn't happened yet); v1's nearest-|gap| match let body
# latency attach it to the following read.
ACCEPTANCE_VERSION = "2"
FOLLOW_WINDOW_TICKS = 600   # 30 s at 20 ticks/s — how long a suggestion stays open
NEAR_RADIUS = 2             # Chebyshev: "followed near" = same block within this box
TYPE_RADIUS = 8             # "followed type" = same block anywhere within this box
VOICE_MATCH_TICKS = 100     # a voicing pairs with the nearest trace read within 5 s


@dataclass(frozen=True)
class Outcome:
    """One suggestion (or control read) and what the human did with it."""

    tick: int
    voiced: bool
    block: str
    cell: tuple[int, int, int]
    outcome: str   # followed_exact | followed_near | followed_type | contradicted | ignored


def _chebyshev(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def _bare(block: str) -> str:
    # "minecraft:spruce_planks" and "spruce_planks" are the same block: the
    # capture keeps the namespace, the gate's proposals do not.
    return block.rsplit(":", 1)[-1]


def classify(tick: int, block: str, cell, human_places) -> str:
    """What the human did in the window after one suggestion.

    `human_places` is [(tick, (x, y, z), block_name), ...] — human-actor PLACE
    events only (the A7 filter is the caller's job, same as everywhere else).
    Tiers are checked strongest-first; `contradicted` outranks the weak tiers
    because a different block at the exact suggested cell is an answer, not noise.
    """
    window = [(t, pos, _bare(name))
              for t, pos, name in human_places
              if tick < t <= tick + FOLLOW_WINDOW_TICKS]
    cell = tuple(cell)
    block = _bare(block)
    for _, pos, name in window:
        if tuple(pos) == cell and name == block:
            return "followed_exact"
    if any(tuple(pos) == cell and name != block for _, pos, name in window):
        return "contradicted"
    if any(name == block and _chebyshev(pos, cell) <= NEAR_RADIUS
           for _, pos, name in window):
        return "followed_near"
    if any(name == block and _chebyshev(pos, cell) <= TYPE_RADIUS
           for _, pos, name in window):
        return "followed_type"
    return "ignored"


def _followed(outcome: str) -> bool:
    # The lift counts the two strong tiers; followed_type is reported but too weak
    # to count as acceptance (a builder placing planks 8 blocks away proves little).
    return outcome in ("followed_exact", "followed_near")


def pair_voicings(voiced_rows, trace_rows, tick_of_ms) -> tuple[list[dict], int]:
    """Attach each body voicing to the nearest suggestion-bearing trace read.

    `voiced_rows` carry body wallclock ms (`ts`); `tick_of_ms` maps wallclock to
    session ticks (built from the capture — one JVM, one clock on this rig). A
    voicing with no proposal-bearing SUGGEST/PREVIEW read within VOICE_MATCH_TICKS
    is dropped and counted — a mismatch means clock or logging trouble, and the
    report must show it rather than absorb it.

    A voicing comes from a read that ALREADY happened, so among candidates in the
    window the latest read at-or-before the voiced tick wins; a read after it is
    only accepted when nothing precedes it (clock skew). When the voiced row logs
    the proposal cell, only reads proposing that cell are candidates — the body's
    ~2 s speaking latency spans a read boundary at 1 Hz, and the cell pins which
    read actually produced the words.
    """
    candidates = [r for r in trace_rows
                  if r.get("proposal_first")
                  and r.get("chosen_state") in ("suggest", "preview")]
    paired, unmatched = [], 0
    for voiced in voiced_rows:
        tick = tick_of_ms(voiced["ts"])
        pool = candidates
        if voiced.get("cell") is not None:
            same_cell = [r for r in candidates
                         if tuple(r["proposal_first"]["cell"]) == tuple(voiced["cell"])]
            if same_cell:
                pool = same_cell
        in_window = [r for r in pool if abs(r["tick"] - tick) <= VOICE_MATCH_TICKS]
        before = [r for r in in_window if r["tick"] <= tick]
        if before:
            best = max(before, key=lambda r: r["tick"])
        elif in_window:
            best = min(in_window, key=lambda r: r["tick"] - tick)
        else:
            unmatched += 1
            continue
        paired.append(best)
    return paired, unmatched


def report(trace_rows, voiced_trace_rows, human_places) -> dict:
    """The per-session acceptance report: voiced outcomes, control outcomes, lift.

    `voiced_trace_rows` are the trace rows pair_voicings matched; the control is
    every OTHER proposal-bearing read (unvoiced by definition — including throttled
    SUGGESTs, which the human never heard).
    """
    voiced_ids = {id(r) for r in voiced_trace_rows}
    control_rows = [r for r in trace_rows
                    if r.get("proposal_first") and id(r) not in voiced_ids]

    def outcomes(rows, voiced):
        out = []
        for row in rows:
            first = row["proposal_first"]
            out.append(Outcome(tick=row["tick"], voiced=voiced,
                               block=first["block"], cell=tuple(first["cell"]),
                               outcome=classify(row["tick"], first["block"],
                                                first["cell"], human_places)))
        return out

    voiced = outcomes(voiced_trace_rows, True)
    control = outcomes(control_rows, False)

    def rate(rows):
        return round(sum(_followed(o.outcome) for o in rows) / len(rows), 4) if rows else None

    def counts(rows):
        tally: dict[str, int] = {}
        for o in rows:
            tally[o.outcome] = tally.get(o.outcome, 0) + 1
        return tally

    voiced_rate, control_rate = rate(voiced), rate(control)
    return {
        "acceptance_version": ACCEPTANCE_VERSION,
        "constants": {"follow_window_ticks": FOLLOW_WINDOW_TICKS,
                      "near_radius": NEAR_RADIUS, "type_radius": TYPE_RADIUS},
        "voiced": {"n": len(voiced), "followed_rate": voiced_rate,
                   "outcomes": counts(voiced)},
        "control_unvoiced": {"n": len(control), "followed_rate": control_rate,
                             "outcomes": counts(control)},
        "lift": (round(voiced_rate - control_rate, 4)
                 if voiced_rate is not None and control_rate is not None else None),
        "detail": [vars(o) | {"cell": list(o.cell)} for o in voiced],
    }
