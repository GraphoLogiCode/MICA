"""The intent-recognition metric suite of the evaluation plan (vault note 06).

Every metric here reads the same minimal record of one session under one arm: an
ordered list of steps, each carrying the arm's goal distribution at that step, the
true category, and the build-progress fraction (how much of the build stood BEFORE
the step's action — the same definition Source B pins). Arms differ wildly in how
they produce the distribution (argmax floor, feedforward classifier, prompted LLM,
Bayes filter); by the time numbers are computed here they all look identical, which
is what makes the four-arm table comparable.

A step is a plain dict: {"dist": {goal: p}, "truth": goal, "progress": float}.
An arm with no confidence (the reactive floor emits a bare argmax) uses a one-hot
dist and is excluded from the calibration metric by the caller — a one-hot "belief"
would be trivially miscalibrated and the exclusion is the honest treatment.

The headline is EARLY-SEPARATION: P(truth) minus the nearest competitor, as a curve
over progress. The pre-registered falsification criterion (written before any run):
if an arm's mean separation is not above zero before 40% of the build is placed, on
held-out sessions, that arm's earliness claim fails.
"""
from __future__ import annotations

import math
import statistics

from ..contracts.b1 import GOALS

FALSIFICATION_PROGRESS = 0.40     # pinned in vault note 06, before any run
PROGRESS_BINS = tuple(round(0.1 * b, 1) for b in range(1, 11))


def step(dist: dict[str, float], truth: str, progress: float) -> dict:
    """One evaluation step. Normalizes defensively so no arm can leak an unnormalized
    distribution into the metrics."""
    total = sum(dist.values())
    if total <= 0:
        dist = {g: 1.0 / len(GOALS) for g in GOALS}
    else:
        dist = {g: dist.get(g, 0.0) / total for g in GOALS}
    return {"dist": dist, "truth": truth, "progress": progress}


def _top(dist: dict[str, float]) -> str:
    return max(dist, key=dist.get)


def _sustain_index(steps: list[dict]) -> int:
    calls = [_top(s["dist"]) for s in steps]
    start = len(calls)
    for index in range(len(calls) - 1, -1, -1):
        if calls[index] != steps[index]["truth"]:
            break
        start = index
    return start


def sustained_from(steps: list[dict]) -> float:
    """Time-to-correct-intent, as the fraction of STEPS after which the top-1 call is
    the truth and stays the truth (run_tracker's banked definition; lower = earlier)."""
    return _sustain_index(steps) / len(steps) if steps else 1.0


def sustained_from_progress(steps: list[dict]) -> float:
    """The same sustain point in the units vault note 06 defines "early" in: the
    FRACTION OF THE BUILD PLACED when the call locks on. Real sessions are idle-heavy,
    so step-fraction and build-fraction diverge — this is the quotable number
    (harness review 2026-07-05, F1)."""
    if not steps:
        return 1.0
    index = _sustain_index(steps)
    return steps[index]["progress"] if index < len(steps) else 1.0


def accuracy_by_progress(steps: list[dict]) -> dict[str, float | None]:
    """Early goal accuracy: top-1 accuracy inside each progress decile."""
    bins: dict[float, list[bool]] = {b: [] for b in PROGRESS_BINS}
    for s in steps:
        for b in PROGRESS_BINS:
            if s["progress"] <= b:
                bins[b].append(_top(s["dist"]) == s["truth"])
                break
    return {str(b): round(statistics.mean(v), 3) if v else None for b, v in bins.items()}


def top_k_accuracy(steps: list[dict], k: int = 2) -> float:
    hits = [s["truth"] in sorted(s["dist"], key=s["dist"].get, reverse=True)[:k]
            for s in steps]
    return round(statistics.mean(hits), 3) if hits else 0.0


def separation_curve(steps: list[dict]) -> dict[str, float | None]:
    """The headline: mean (P(truth) − nearest competitor) per progress decile.
    Positive means the truth leads; the curve's early bins are the thesis."""
    bins: dict[float, list[float]] = {b: [] for b in PROGRESS_BINS}
    for s in steps:
        margin = s["dist"][s["truth"]] - max(
            p for g, p in s["dist"].items() if g != s["truth"])
        for b in PROGRESS_BINS:
            if s["progress"] <= b:
                bins[b].append(margin)
                break
    return {str(b): round(statistics.mean(v), 4) if v else None for b, v in bins.items()}


def early_separation_verdict(steps: list[dict]) -> dict:
    """The pre-registered falsification check: mean separation over all steps with
    progress <= 40%. Stated either way, never relabeled."""
    early = [s["dist"][s["truth"]] - max(p for g, p in s["dist"].items() if g != s["truth"])
             for s in steps if s["progress"] <= FALSIFICATION_PROGRESS]
    mean = round(statistics.mean(early), 4) if early else None
    return {"early_steps": len(early), "mean_separation": mean,
            "criterion": f"mean separation > 0 before {FALSIFICATION_PROGRESS:.0%} of the build",
            "passes": (mean is not None and mean > 0.0)}


def entropy_curve(steps: list[dict]) -> dict[str, float | None]:
    """Entropy of the goal distribution per progress decile (nats). A falling curve
    means the mechanism is committing; a flat curve at ln(5) means it never does."""
    bins: dict[float, list[float]] = {b: [] for b in PROGRESS_BINS}
    for s in steps:
        h = -sum(p * math.log(p) for p in s["dist"].values() if p > 0.0)
        for b in PROGRESS_BINS:
            if s["progress"] <= b:
                bins[b].append(h)
                break
    return {str(b): round(statistics.mean(v), 3) if v else None for b, v in bins.items()}


def reliability(steps: list[dict], bin_count: int = 10) -> dict:
    """Belief calibration: (top confidence, was-it-right) points binned to reliability
    rows + ECE — the same shape calibration_report.py banks, computed on any arm."""
    rows = [{"count": 0, "confidence_sum": 0.0, "correct": 0} for _ in range(bin_count)]
    for s in steps:
        confidence = s["dist"][_top(s["dist"])]
        index = min(int(confidence * bin_count), bin_count - 1)
        rows[index]["count"] += 1
        rows[index]["confidence_sum"] += confidence
        rows[index]["correct"] += _top(s["dist"]) == s["truth"]
    total = len(steps) or 1
    out, ece = [], 0.0
    for i, row in enumerate(rows):
        if row["count"] == 0:
            continue
        mean_conf = row["confidence_sum"] / row["count"]
        acc = row["correct"] / row["count"]
        ece += (row["count"] / total) * abs(acc - mean_conf)
        out.append({"bin": f"{i / bin_count:.1f}-{(i + 1) / bin_count:.1f}",
                    "count": row["count"], "mean_confidence": round(mean_conf, 3),
                    "empirical_accuracy": round(acc, 3)})
    return {"points": len(steps), "ece": round(ece, 4), "bins": out}


def recovery_after_pivot(steps: list[dict], pivot_index: int) -> dict:
    """Time-to-recover-after-a-pivot: steps from the pivot until P(new truth)
    overtakes the OLD goal and stays ahead. The old goal is the truth of the step
    just before the pivot; recovery is measured against it specifically (06's
    definition), not against all competitors."""
    old_goal = steps[pivot_index - 1]["truth"]
    after = steps[pivot_index:]
    recovered_at = len(after)
    for index in range(len(after) - 1, -1, -1):
        s = after[index]
        if s["dist"][s["truth"]] <= s["dist"][old_goal]:
            break
        recovered_at = index
    return {"pivot_index": pivot_index, "old_goal": old_goal,
            "new_goal": after[0]["truth"] if after else None,
            "steps_to_recover": recovered_at if recovered_at < len(after) else None,
            "recovered": recovered_at < len(after)}


def session_metrics(steps: list[dict], calibrated: bool = True) -> dict:
    """The whole suite for one (session, arm) pair."""
    final = steps[-1]
    return {
        "steps": len(steps),
        "final_correct": _top(final["dist"]) == final["truth"],
        "sustained_from": round(sustained_from(steps), 3),
        "sustained_from_progress": round(sustained_from_progress(steps), 3),
        "top2_accuracy": top_k_accuracy(steps, 2),
        "accuracy_by_progress": accuracy_by_progress(steps),
        "separation_by_progress": separation_curve(steps),
        "early_separation": early_separation_verdict(steps),
        "entropy_by_progress": entropy_curve(steps),
        "reliability": reliability(steps) if calibrated else None,
    }
