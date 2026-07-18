"""Calibration report — the D3 pass-criterion artifact, harness built ahead of Phase E.

    python scripts/calibration_report.py [--real] [--heads v1]

A belief is only useful to the gate if its probabilities MEAN something: when the
tracker says "70% habitation", it should be right about 70% of the time. This script
measures exactly that. It re-runs the tracker over the labeled corpus through the
same verified path run_tracker uses, collects every correction's (top-goal
confidence, was-it-right) point, and bins them into reliability data:

  per confidence bin -> how many corrections landed there, their mean confidence,
                        and the fraction that were actually correct
  ECE                -> the confidence-weighted gap between the two (0 = calibrated)

The default run (scripted corpus, hand-coded v0 heads) proves the harness and gives
the baseline number, nothing more. Honesty note printed with every run: the corpus
has ONE builder (the generator), so the builder-level holdout the D3 note prescribes
is degenerate here — real multi-session data is what makes the number load-bearing.

--real scores the labeled real captures instead (truth = builder category, the
quarantined session excluded); --heads v1 swaps in the trained heads with their
jointly-fitted temperatures and filter knobs. --held-out further drops every real
session whose pairs entered v1 TRAINING, leaving only sessions the model has never
seen — that run is the Phase-E pass criterion (09-F9: "calibrated" only with a
held-out reliability diagram). Every run also reports the 12-F8 mode-informativeness
check: the belief-averaged deliberative-vs-heuristic probability gap of the observed
action at choice points (PLACE corrections) — trained heads must widen it over v0's
thin measured margin. Output names carry _real/_v1/_heldout suffixes so baselines
never overwrite.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.contracts.b1 import GOALS, MacroAction             # noqa: E402
from mica.capture import session_store                      # noqa: E402
from mica.contracts.b3 import fuse_streams                   # noqa: E402
from mica.intent.heads_v0 import likelihood                  # noqa: E402
from mica.intent.tracker import (                            # noqa: E402
    TrackerParams, category_marginal, correct, predict, uniform_belief,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORPUS = os.path.join(_ROOT, "capture", "scripted")
_RAW = os.path.join(_ROOT, "capture", "raw")
_BINS = 10


def _evidence(directory: str, session_id: str, suffix: str) -> str:
    """Resolve a session's evidence file. Scripted corpus sessions sit flat in
    `capture/scripted`; real captures live in the dated `capture/raw/<date>/<id>/`
    layout, so fall back to session_store when the flat path is absent."""
    flat = os.path.join(directory, f"{session_id}{suffix}")
    return flat if os.path.exists(flat) else session_store.session_file(session_id, suffix)


def _correction_points(directory: str, session_id: str, truth: str,
                       params: TrackerParams) -> tuple[list[tuple[float, bool]], list[float]]:
    """(top-goal confidence, correct?) at every correction of one session, plus the
    12-F8 margins: at each PLACE correction (a choice point), the belief-averaged
    deliberative probability of the observed action minus the heuristic one."""
    b1 = [json.loads(line) for line in
          open(_evidence(directory, session_id, ".evidence2d.jsonl"), encoding="utf-8")]
    b2 = [json.loads(line) for line in
          open(_evidence(directory, session_id, ".evidence3d.jsonl"), encoding="utf-8")]
    belief = uniform_belief()
    previous_tick = 0
    points, margins = [], []
    for fused in fuse_streams(b1, b2, session_id):           # counts asserted, joins verified
        dt = max(fused.tick - previous_tick, 1) / 20.0
        previous_tick = fused.tick
        belief = predict(belief, dt, params)
        table = likelihood(fused, fused.a_hat)
        if fused.a_hat == MacroAction.PLACE:
            delib_mass = sum(belief[(g, 0)] for g in GOALS)
            if delib_mass > 0:
                weighted_delib = sum(belief[(g, 0)] * table[(g, 0)] for g in GOALS) / delib_mass
                # the single-goal read of the heuristic term is valid only under the
                # 09-F4 goal-symmetry invariant — assert it instead of assuming it,
                # since --heads swaps in arbitrary head versions (review 13 F35)
                heur_values = [table[(g, 1)] for g in GOALS]
                assert max(heur_values) - min(heur_values) < 1e-9, (
                    "heuristic head is not goal-symmetric; the 12-F8 margin is undefined")
                margins.append(weighted_delib - table[(GOALS[0], 1)])
        belief, _ = correct(belief, table, params)
        marginal = category_marginal(belief)
        top = max(marginal, key=marginal.get)
        points.append((marginal[top], top == truth))
    return points, margins


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
    global likelihood
    real = "--real" in sys.argv
    version = (sys.argv[sys.argv.index("--heads") + 1]
               if "--heads" in sys.argv else None)
    v1 = version is not None          # "trained heads in play" (v1 or v2)
    if v1:
        from importlib import import_module
        heads_module = import_module(f"mica.intent.heads_{version}")
        likelihood = heads_module.likelihood
        params = heads_module.tracker_params()
    else:
        params = TrackerParams()
    held_out = "--held-out" in sys.argv
    if real:
        from run_tracker import real_labeled_sessions
        labels = real_labeled_sessions()
        if not labels:
            print("no labeled real captures with banked evidence found")
            return 1
        if held_out:
            # A session trained on v1 iff its pairs file exists AND it was not in the
            # training holdout (heads_v1.json pins that list). What remains is data
            # the model has never seen — the only rows a calibration CLAIM may cite.
            # The pairs file lives in the dated session dir (flat only for legacy
            # captures), so it is resolved through _evidence — the bare flat path
            # matched nothing after the layout migration, which silently made
            # "held-out" score every session, trained ones included.
            with open(os.path.join(_ROOT, "models", f"heads_{version or 'v1'}.json"), encoding="utf-8") as handle:
                model_data = json.load(handle)["data"]
            pinned_holdout = set(model_data["held_out_sessions"])
            # The pinned holdout is the training-time VALIDATION set: its real
            # sessions tuned the temperatures, the (eps, lambda) grid, and the early
            # stop. They are tuning-exposed, not never-seen, so a held-out
            # calibration CLAIM must drop them too (review 13 F6 — the fitted
            # session survived this filter and biased the pass artifact in the
            # flattering direction).
            tuning = {s for s in set(model_data.get("validation_sessions",
                                                    pinned_holdout))
                      if s.startswith("fabric-")}
            trained = {sid for sid in labels
                       if os.path.exists(_evidence(_RAW, sid, ".source_b.jsonl"))
                       and sid not in pinned_holdout}
            labels = {sid: meta for sid, meta in labels.items()
                      if sid not in trained and sid not in tuning}
            print(f"held-out only: dropped training sessions {sorted(trained)} "
                  f"and tuning-validation sessions {sorted(tuning)}")
        directory = _RAW
    else:
        labels_path = os.path.join(_CORPUS, "labels.json")
        if not os.path.exists(labels_path):
            print("no corpus found - run scripts/make_scripted_corpus.py first")
            return 1
        with open(labels_path, encoding="utf-8") as handle:
            labels = json.load(handle)
        directory = _CORPUS

    pooled: list[tuple[float, bool]] = []
    finals: list[tuple[float, bool]] = []
    all_margins: list[float] = []
    by_goal: dict[str, list[tuple[float, bool]]] = {goal: [] for goal in GOALS}
    for session_id, label in labels.items():
        points, margins = _correction_points(directory, session_id, label["goal"], params)
        pooled.extend(points)
        all_margins.extend(margins)
        finals.append(points[-1])
        by_goal[label["goal"]].extend(points)

    report = {
        "heads": f"trained {version}" if v1 else "hand-coded v0",
        "data": ("labeled real captures (truth = builder category)" if real
                 else "scripted corpus"),
        "params": {"lambda_g": params.lambda_g, "lambda_z": params.lambda_z,
                   "epsilon": params.epsilon},
        "split_caveat": ("held-out real sessions are the load-bearing rows; the "
                         "training holdout is pinned in heads_v1.json" if real else
                         "single builder (the scripted generator) — the builder-level "
                         "holdout is degenerate until real multi-session data exists"),
        "all_corrections": _reliability(pooled),
        "final_corrections_only": _reliability(finals),
        "per_goal": {goal: _reliability(points) for goal, points in by_goal.items() if points},
        "mode_margin_at_choice_points": {
            "mean": round(statistics.mean(all_margins), 4) if all_margins else None,
            "points": len(all_margins),
            "what": "belief-averaged P_delib(a|e,g) minus P_heur(a|e) at PLACE "
                    "corrections (12-F8: trained heads must widen this)",
        },
    }
    name = (f"calibration_report{'_real' if real else ''}{f'_{version}' if v1 else ''}"
            f"{'_heldout' if held_out else ''}.json")
    out = os.path.join(directory, name)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    overall = report["all_corrections"]
    print(f"calibration report ({report['heads']}, {'REAL captures' if real else 'scripted corpus'})"
          f" -> {os.path.relpath(out, _ROOT)}")
    print(f"  CAVEAT: {report['split_caveat']}")
    print(f"  all corrections: {overall['points']} points, ECE {overall['ece']}")
    for row in overall["bins"]:
        bar = "#" * round(row["empirical_accuracy"] * 20)
        print(f"    {row['bin']}: n={row['count']:<5} conf {row['mean_confidence']:.3f}"
              f" -> acc {row['empirical_accuracy']:.3f}  {bar}")
    final = report["final_corrections_only"]
    print(f"  final corrections only: {final['points']} sessions, ECE {final['ece']}")
    margin = report["mode_margin_at_choice_points"]
    print(f"  mode margin at choice points (12-F8): {margin['mean']}"
          f" over {margin['points']} PLACE corrections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
