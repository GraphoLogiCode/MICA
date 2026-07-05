"""Calibration report — the D3 pass-criterion artifact, harness built ahead of Phase E.

    python scripts/calibration_report.py

A belief is only useful to the gate if its probabilities MEAN something: when the
tracker says "70% habitation", it should be right about 70% of the time. This script
measures exactly that. It re-runs the tracker over the labeled corpus through the
same verified path run_tracker uses, collects every correction's (top-goal
confidence, was-it-right) point, and bins them into reliability data:

  per confidence bin -> how many corrections landed there, their mean confidence,
                        and the fraction that were actually correct
  ECE                -> the confidence-weighted gap between the two (0 = calibrated)

Today it runs on the scripted corpus with the hand-coded v0 heads — that proves the
harness and gives the baseline number, nothing more. The Phase-E criterion ("heads
calibrated on held-out sessions before their outputs are trusted") reuses this exact
artifact once trained heads and real captures exist. Honesty note printed with every
run: the corpus has ONE builder (the generator), so the builder-level holdout the
D3 note prescribes is degenerate here — real multi-session data is what makes the
number load-bearing.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.contracts.b1 import GOALS                          # noqa: E402
from mica.contracts.b3 import fuse_dicts                     # noqa: E402
from mica.intent.heads_v0 import likelihood                  # noqa: E402
from mica.intent.tracker import (                            # noqa: E402
    TrackerParams, category_marginal, correct, predict, uniform_belief,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORPUS = os.path.join(_ROOT, "capture", "scripted")
_BINS = 10


def _correction_points(session_id: str, truth: str, params: TrackerParams) -> list[tuple[float, bool]]:
    """(top-goal confidence, correct?) at every correction of one session."""
    b1 = [json.loads(line) for line in
          open(os.path.join(_CORPUS, f"{session_id}.evidence2d.jsonl"), encoding="utf-8")]
    b2 = [json.loads(line) for line in
          open(os.path.join(_CORPUS, f"{session_id}.evidence3d.jsonl"), encoding="utf-8")]
    scored = [record for record in b1 if record["scored"]]
    belief = uniform_belief()
    previous_tick = 0
    points = []
    for b1_record, b2_record in zip(scored, b2):
        fused = fuse_dicts(b1_record, b2_record)             # the verified join
        dt = max(fused.tick - previous_tick, 1) / 20.0
        previous_tick = fused.tick
        belief = predict(belief, dt, params)
        belief, _ = correct(belief, likelihood(fused, fused.a_hat), params)
        marginal = category_marginal(belief)
        top = max(marginal, key=marginal.get)
        points.append((marginal[top], top == truth))
    return points


def _reliability(points: list[tuple[float, bool]]) -> dict:
    """Bin the (confidence, correct) points; ECE is the weighted confidence-accuracy gap."""
    bins = [{"lo": b / _BINS, "hi": (b + 1) / _BINS, "count": 0,
             "confidence_sum": 0.0, "correct": 0} for b in range(_BINS)]
    for confidence, was_correct in points:
        index = min(int(confidence * _BINS), _BINS - 1)
        bins[index]["count"] += 1
        bins[index]["confidence_sum"] += confidence
        bins[index]["correct"] += was_correct
    total = len(points) or 1
    rows, ece = [], 0.0
    for entry in bins:
        if entry["count"] == 0:
            continue
        mean_confidence = entry["confidence_sum"] / entry["count"]
        accuracy = entry["correct"] / entry["count"]
        ece += (entry["count"] / total) * abs(accuracy - mean_confidence)
        rows.append({"bin": f"{entry['lo']:.1f}-{entry['hi']:.1f}", "count": entry["count"],
                     "mean_confidence": round(mean_confidence, 3),
                     "empirical_accuracy": round(accuracy, 3)})
    return {"points": len(points), "ece": round(ece, 4), "bins": rows}


def main() -> int:
    labels_path = os.path.join(_CORPUS, "labels.json")
    if not os.path.exists(labels_path):
        print("no corpus found - run scripts/make_scripted_corpus.py first")
        return 1
    with open(labels_path, encoding="utf-8") as handle:
        labels = json.load(handle)
    params = TrackerParams()

    pooled: list[tuple[float, bool]] = []
    finals: list[tuple[float, bool]] = []
    by_goal: dict[str, list[tuple[float, bool]]] = {goal: [] for goal in GOALS}
    for session_id, label in labels.items():
        points = _correction_points(session_id, label["goal"], params)
        pooled.extend(points)
        finals.append(points[-1])
        by_goal[label["goal"]].extend(points)

    report = {
        "heads": "hand-coded v0 (harness proof; Phase E swaps in the trained heads)",
        "split_caveat": "single builder (the scripted generator) — the builder-level "
                        "holdout is degenerate until real multi-session data exists",
        "all_corrections": _reliability(pooled),
        "final_corrections_only": _reliability(finals),
        "per_goal": {goal: _reliability(points) for goal, points in by_goal.items() if points},
    }
    out = os.path.join(_CORPUS, "calibration_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    overall = report["all_corrections"]
    print(f"calibration report (v0 heads, scripted corpus) -> {os.path.relpath(out, _ROOT)}")
    print(f"  CAVEAT: {report['split_caveat']}")
    print(f"  all corrections: {overall['points']} points, ECE {overall['ece']}")
    for row in overall["bins"]:
        bar = "#" * round(row["empirical_accuracy"] * 20)
        print(f"    {row['bin']}: n={row['count']:<5} conf {row['mean_confidence']:.3f}"
              f" -> acc {row['empirical_accuracy']:.3f}  {bar}")
    final = report["final_corrections_only"]
    print(f"  final corrections only: {final['points']} sessions, ECE {final['ece']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
