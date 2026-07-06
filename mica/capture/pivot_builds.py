"""Pivot sessions: one pretend recording in which the builder FINISHES one goal and
starts another. This is the scenario the time-to-recover-after-a-pivot metric needs
(vault note 06): inside a single session the true goal changes, so consistency stops
being the right target and tracking takes over.

A pivot build is two scripted plans stitched end to end: plan A builds normally, then
after a thinking pause plan B starts on fresh ground far enough away that the two
structures cannot share cells. Ground truth per tick is goal A before the seam, goal
B from the seam on. Everything else (region, evidence streams, replay) treats the
result as an ordinary session — which is the point: only the metrics know a pivot
happened.
"""
from __future__ import annotations

import dataclasses

from .scripted_goals import BuildPlan, build_from_plan
from .synthetic import ScriptedBuild

_PIVOT_GAP_TICKS = 120     # the pause between finishing A and starting B (~6 s)
_PIVOT_OFFSET = 40         # plan B's ground is shifted this far so the builds never touch


def pivot_build(plan_a: BuildPlan, plan_b: BuildPlan) -> tuple[ScriptedBuild, dict]:
    """Stitch two plans into one session. Returns the build plus a truth record:
    goal_a, goal_b, and seam_tick (the first tick that belongs to plan B)."""
    build_a, truth_a = build_from_plan(plan_a)
    moved_b = dataclasses.replace(
        plan_b, origin=(plan_b.origin[0] + _PIVOT_OFFSET, plan_b.origin[1] + _PIVOT_OFFSET))
    build_b, truth_b = build_from_plan(moved_b)

    offset = build_a.placements[-1].tick + _PIVOT_GAP_TICKS
    shifted = tuple(dataclasses.replace(p, tick=p.tick + offset)
                    for p in build_b.placements)
    build = dataclasses.replace(
        build_a,
        session_id=f"pivot-{truth_a['goal']}-to-{truth_b['goal']}-{plan_a.seed}-{plan_b.seed}",
        placements=build_a.placements + shifted)
    # The seam is plan B's FIRST placement — the moment new-goal evidence begins.
    # Corrections during the pause keep goal A: scoring them against B would make
    # recovery unrecoverable by construction (harness review 2026-07-05, F2), and
    # recovery therefore reads as "steps from first new-goal evidence".
    seam_tick = min(p.tick for p in shifted)
    return build, {"goal_a": truth_a["goal"], "goal_b": truth_b["goal"],
                   "seam_tick": seam_tick}
