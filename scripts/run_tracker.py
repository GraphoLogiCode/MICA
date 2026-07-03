"""Run the belief tracker (hand-coded heads v0) over the scripted corpus — twice.

    python scripts/run_tracker.py

Reads capture/scripted (labels + evidence jsonl; run make_scripted_corpus.py first).
Every correction's evidence is built through the VERIFIED B3 handoff (contracts/b3.py:
the B1 and B2 records must be the same step or fusion refuses), then the tracker runs
two arms over identical sessions:

  fused     — the full B3 bundle: D2 structure + D1 behavior (held item, action streak,
              crosshair dwell, s_goal when pixels exist)
  stripped  — the same records with the 2D behavior channels blanked (strip_behavior),
              i.e. structure + action label only

The gap between the arms is the D1 stream's measured contribution — the with/without
probe the D1 design prescribes, at the belief level. Also reports the structure-only
floor and checks the structural invariants at every step. Writes belief_summary.json.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.contracts.b3 import fuse_dicts                     # noqa: E402
from mica.intent.heads_v0 import likelihood, strip_behavior, strip_structure  # noqa: E402
from mica.intent.tracker import (                            # noqa: E402
    TrackerParams, category_marginal, correct, mode_marginal, predict, uniform_belief,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORPUS = os.path.join(_ROOT, "capture", "scripted")


def _sustained_from(calls: list[str], truth: str) -> float:
    """The earliest progress fraction from which every call is the true category."""
    start = len(calls)
    for index in range(len(calls) - 1, -1, -1):
        if calls[index] != truth:
            break
        start = index
    return start / len(calls) if calls else 1.0


def _track(fused_records, truth: str, params: TrackerParams) -> dict:
    belief = uniform_belief()
    previous_tick = 0
    calls, z1 = [], []
    for fused in fused_records:
        dt = max(fused.tick - previous_tick, 1) / 20.0
        previous_tick = fused.tick
        belief = predict(belief, dt, params)
        belief, normalizer = correct(belief, likelihood(fused, fused.a_hat), params)
        assert abs(sum(belief.values()) - 1.0) < 1e-9, "belief mass drifted"
        assert normalizer >= params.epsilon / params.action_count - 1e-15, "floor violated"
        marginal = category_marginal(belief)
        calls.append(max(marginal, key=marginal.get))
        z1.append(mode_marginal(belief)[1])
    return {"final_correct": calls[-1] == truth,
            "sustained_from": round(_sustained_from(calls, truth), 3),
            "mean_p_heuristic": round(statistics.mean(z1), 3)}


def main() -> int:
    labels_path = os.path.join(_CORPUS, "labels.json")
    if not os.path.exists(labels_path):
        print("no corpus found - run scripts/make_scripted_corpus.py first")
        return 1
    with open(labels_path, encoding="utf-8") as handle:
        labels = json.load(handle)
    params = TrackerParams()

    results = []
    for session_id, label in labels.items():
        b1 = [json.loads(line) for line in
              open(os.path.join(_CORPUS, f"{session_id}.evidence2d.jsonl"), encoding="utf-8")]
        b2 = [json.loads(line) for line in
              open(os.path.join(_CORPUS, f"{session_id}.evidence3d.jsonl"), encoding="utf-8")]
        scored = [r for r in b1 if r["scored"]]
        fused = [fuse_dicts(a, b) for a, b in zip(scored, b2)]   # the handoff, verified per record

        floor_calls = [max(f.per_goal, key=lambda g: (f.per_goal[g].fit, f.per_goal[g].comp))
                       for f in fused]
        entry = {"session": session_id, "goal": label["goal"], "mode": label["mode"],
                 "fused": _track(fused, label["goal"], params),
                 "no_behavior": _track([strip_behavior(f) for f in fused], label["goal"], params),
                 "no_structure": _track([strip_structure(f) for f in fused], label["goal"], params),
                 "floor_sustained_from": round(_sustained_from(floor_calls, label["goal"]), 3),
                 "floor_final_correct": floor_calls[-1] == label["goal"]}
        results.append(entry)

    def arm(name):
        return (statistics.mean(r[name]["final_correct"] for r in results),
                statistics.mean(r[name]["sustained_from"] for r in results))

    fused_acc, fused_early = arm("fused")
    no_b_acc, no_b_early = arm("no_behavior")
    no_s_acc, no_s_early = arm("no_structure")
    floor_acc = statistics.mean(r["floor_final_correct"] for r in results)
    floor_early = statistics.mean(r["floor_sustained_from"] for r in results)

    summary = {
        "params": {"lambda_g": params.lambda_g, "lambda_z": params.lambda_z,
                   "epsilon": params.epsilon, "heads": "hand-coded v0"},
        "sessions": len(results),
        "fused": {"final_accuracy": round(fused_acc, 3), "mean_sustained_from": round(fused_early, 3)},
        "no_behavior_3d_only": {"final_accuracy": round(no_b_acc, 3),
                                "mean_sustained_from": round(no_b_early, 3)},
        "no_structure_2d_only": {"final_accuracy": round(no_s_acc, 3),
                                 "mean_sustained_from": round(no_s_early, 3)},
        "structure_only_floor": {"final_accuracy": round(floor_acc, 3),
                                 "mean_sustained_from": round(floor_early, 3)},
        "per_session": results,
    }
    out = os.path.join(_CORPUS, "belief_summary.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"belief tracker (hand-coded heads v0) over {len(results)} sessions - three arms")
    print(f"  final accuracy:   fused {fused_acc:.3f}   3D-only {no_b_acc:.3f}"
          f"   2D-only {no_s_acc:.3f}   floor {floor_acc:.3f}")
    print(f"  sustained from:   fused {fused_early:.3f}   3D-only {no_b_early:.3f}"
          f"   2D-only {no_s_early:.3f}   floor {floor_early:.3f}   (lower = earlier)")
    print(f"  3D structure contribution (vs 2D-only): {no_s_early - fused_early:+.3f} earliness,"
          f" {fused_acc - no_s_acc:+.3f} accuracy")
    print(f"  2D behavior contribution (vs 3D-only):  {no_b_early - fused_early:+.3f} earliness,"
          f" {fused_acc - no_b_acc:+.3f} accuracy")
    print(f"  handoff verified per record; invariants held every step  ->  {os.path.basename(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
