"""The Uni3D earn-its-place probe: does pretrained 3D shape understanding separate the
goal categories on the labeled corpus — and how early?

    python scripts/probe_shape3d.py [--per-goal N] [--seed S]

Regenerates the seeded scripted corpus in-process, replays each session, and at each
fraction of build progress scores the player-built cells' point cloud with the frozen
Uni3D encoder against the taxonomy's pooled subtype prompts. Reports accuracy and
true-vs-best-distractor margin per progress bin — directly comparable to the symbolic
structure channel's separation curve (d2_progress_report.json), which is the floor this
pretrained encoder must beat, and to MineCLIP's s_goal once real pixels exist.

Fusion is gated on this probe: only if the curve clears the floor does s_shape become a
B2 field and a deliberative-head input (the CLIP4MC earn-its-place discipline).
Requires scripts/setup_uni3d.py to have completed; says exactly what is missing if not.

Verdict 2026-07-03: FAILED — near chance while building, 0.5 at completion against the
floor's 0.967, negative mean margin at every bin. Checked for artifacts: any cloud
rotation makes it worse (the y-up convention is right), and plain one-noun prompts tie
the subtype ensemble. So s_shape stays out of B2. The encoder itself is NOT dead: the
raw embedding passes a learned-readout probe (scripts/probe_h3d_linear.py) and becomes
the h3d channel when Phase E trains the adapter.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture.scripted_goals import build_from_plan, plan_variants   # noqa: E402
from mica.capture.synthetic import generate_session                      # noqa: E402
from mica.contracts.b1 import GOALS                                      # noqa: E402
from mica.perception.shape3d import Uni3DShapeHead, assets_ready         # noqa: E402
from mica.perception.voxel_replay import ReplayWorld, region_around_events  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BINS = [round(0.1 * b, 1) for b in range(1, 11)]


def _flag_value(name: str, default: int) -> int:
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


def main() -> int:
    ready, missing = assets_ready()
    if not ready:
        print("Uni3D assets missing: " + missing)
        print("run scripts/setup_uni3d.py first (clones the repo, fetches the checkpoint,"
              " caches the goal-prompt text embeddings with the paired CLIP tower)")
        return 1

    head = Uni3DShapeHead()
    per_goal = _flag_value("--per-goal", 6)
    seed = _flag_value("--seed", 7)

    curve = {str(b): {"margins": [], "correct": 0, "total": 0} for b in _BINS}
    for goal in GOALS:
        for plan in plan_variants(goal, per_goal, seed):
            build, label = build_from_plan(plan)
            session = generate_session(build)
            events = sorted((e for p in session.packets for e in p.server.block_events),
                            key=lambda e: e.event_id)
            world = ReplayWorld(region_around_events(session), {})
            checkpoints = {max(0, round(b * len(events)) - 1) for b in _BINS}
            scores_at = {}
            for index, event in enumerate(events):
                world.apply(event)
                if index in checkpoints:
                    scores_at[index] = head.score(world.built(), seed=plan.seed)
            for b in _BINS:
                index = max(0, round(b * len(events)) - 1)
                sims = scores_at.get(index)
                if sims is None:
                    continue
                by_goal = dict(zip(GOALS, sims))
                true_score = by_goal.pop(label["goal"])
                best_other = max(by_goal.values())
                bucket = curve[str(b)]
                bucket["margins"].append(true_score - best_other)
                bucket["correct"] += true_score > best_other
                bucket["total"] += 1

    report = {
        "encoder": "uni3d-b (frozen, zero-shot)",
        "prompts": "taxonomy subtype prompts, pooled per category",
        "curve": {
            b: {"accuracy": round(v["correct"] / v["total"], 3) if v["total"] else None,
                "mean_margin": round(statistics.mean(v["margins"]), 4) if v["margins"] else None}
            for b, v in curve.items()
        },
        "chance_level": round(1 / len(GOALS), 3),
    }
    out = os.path.join(_ROOT, "capture", "scripted", "shape3d_probe.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("Uni3D zero-shot separation by progress (chance 0.2):")
    print("  " + "  ".join(f"{b}:{v['accuracy']}" for b, v in report["curve"].items()))
    print("  margins: " + "  ".join(f"{b}:{v['mean_margin']}" for b, v in report["curve"].items()))
    print(f"  -> {os.path.relpath(out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
