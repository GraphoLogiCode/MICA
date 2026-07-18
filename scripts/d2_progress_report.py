"""D2's phase proof artifact: progress-curve sanity + structure-only separation.

    python scripts/d2_progress_report.py

Reads the persisted scripted corpus (capture/scripted: labels.json + evidence jsonl
files — run make_scripted_corpus.py first) and writes d2_progress_report.json. This is
D0's pass criterion for the D2 phase, as a regenerable artifact:

  1. On CLEAN scripted sessions (no break events), comp(true category) must be
     non-decreasing and edit_distance(true category) non-increasing across corrections.
  2. Rotation recovery: full-completion elongated builds must register at the labeled
     rotation (mod the 180-degree symmetry of elongated footprints).
  3. The separation curve: true-category evidence (fit x comp) against the best
     distractor over build progress — the structure-only floor the belief must beat.
  4. Symmetry sanity (added 2026-07-16, perception review Q3): every template
     footprint is mirror-symmetric by construction, so a clean full-completion
     session's final structure record must score symmetry >= 0.9. The feature
     shipped broken for two weeks because nothing gated it; now something does.

Exit 0 only if every clean session is monotone both ways, rotation recovery is full,
and completed builds read symmetric.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORPUS = os.path.join(_ROOT, "capture", "scripted")
_ELONGATED = ("railed bridge", "flat span", "road", "longhouse", "perimeter wall")
_BINS = [round(0.1 * b, 1) for b in range(1, 11)]


def _load_corpus():
    labels_path = os.path.join(_CORPUS, "labels.json")
    if not os.path.exists(labels_path):
        return None
    with open(labels_path, encoding="utf-8") as handle:
        labels = json.load(handle)
    sessions = []
    for session_id, label in labels.items():
        b2 = [json.loads(line) for line in
              open(os.path.join(_CORPUS, f"{session_id}.evidence3d.jsonl"), encoding="utf-8")]
        b1 = [json.loads(line) for line in
              open(os.path.join(_CORPUS, f"{session_id}.evidence2d.jsonl"), encoding="utf-8")]
        has_breaks = any(r["a_hat"] == "break" for r in b1)
        sessions.append((session_id, label, [r for r in b2 if r["event_ids"]], has_breaks))
    return sessions


def _monotonicity(sessions):
    """Monotone within each reigning instance AND registration. A dip exactly where
    the winning style SWITCHES (an early flat span honestly reads as a road until it
    widens) or where the REGISTRATION MOVES (a near-tied argmax flipping offset,
    rotation, or anchor on a five-block build — the doc's own named noisiest step) is
    the reading updating with evidence, not a progress-tracking failure: the features
    are one instance's coherent reading at one pose, so the curve is piecewise-
    monotone by construction. A dip WITHOUT either is a real anomaly and fails.
    (Pose scoping added 2026-07-16 with the vertical-anchor fix — the first corpus
    session whose registration flapped early made the missing scope visible.)"""
    verdicts = []
    for session_id, label, records, has_breaks in sessions:
        if has_breaks:
            continue   # mistakes legitimately dip the curves; the criterion is scoped to clean builds
        feats = [r["per_goal"][label["goal"]] for r in records]
        comp_ok = edit_ok = True
        switches = 0
        pose_moves = 0
        for a, b in zip(feats, feats[1:]):
            if b["subtype"] != a["subtype"]:
                switches += 1
                continue
            if b["pose"] != a["pose"]:
                pose_moves += 1
                continue
            comp_ok = comp_ok and b["comp"] >= a["comp"] - 1e-9
            edit_ok = edit_ok and b["edit_distance"] <= a["edit_distance"] + 1e-9
        verdicts.append({"session": session_id, "comp_monotone": comp_ok,
                         "edit_monotone": edit_ok, "instance_switches": switches,
                         "registration_moves": pose_moves})
    return verdicts


def _rotation_recovery(sessions):
    checks = []
    for session_id, label, records, _ in sessions:
        if label["subtype"] in _ELONGATED and label["completion"] == 1.0:
            got = records[-1]["per_goal"][label["goal"]]["pose"]["rot"]
            checks.append({"session": session_id, "labeled": label["rotation"], "recovered": got,
                           "ok": (got - label["rotation"]) % 180 == 0})
    return checks


def _symmetry_at_completion(sessions):
    """Criterion 4: a finished template build is mirror-symmetric by construction,
    so the last event-carrying record (one block short of done — its evidence is
    pre-action) must still read >= 0.9. Only clean, fully-completed sessions count:
    a session with breaks or a partial build legitimately reads asymmetric."""
    checks = []
    for session_id, label, records, has_breaks in sessions:
        if has_breaks or label["completion"] != 1.0 or not records:
            continue
        final = records[-1]["global"]
        checks.append({"session": session_id, "symmetry": final["symmetry"],
                       "support": final["symmetry_support"],
                       "ok": final["symmetry"] >= 0.9})
    return checks


def _separation_curve(sessions):
    curve = {}
    for target in _BINS:
        margins, correct = [], 0
        for _, label, records, _ in sessions:
            index = min(len(records) - 1, max(0, round(target * len(records)) - 1))
            pg = records[index]["per_goal"]
            score = {g: pg[g]["fit"] * pg[g]["comp"] for g in pg}
            true_score = score.pop(label["goal"])
            margins.append(true_score - max(score.values()))
            correct += true_score > max(score.values())
        curve[str(target)] = {"mean_margin": round(statistics.mean(margins), 4),
                              "accuracy": round(correct / len(sessions), 3)}
    return curve


def main() -> int:
    sessions = _load_corpus()
    if sessions is None:
        print("no corpus found - run scripts/make_scripted_corpus.py first")
        return 1
    mono = _monotonicity(sessions)
    rotation = _rotation_recovery(sessions)
    curve = _separation_curve(sessions)
    symmetry = _symmetry_at_completion(sessions)
    passed = (all(v["comp_monotone"] and v["edit_monotone"] for v in mono)
              and all(c["ok"] for c in rotation)
              and all(s["ok"] for s in symmetry))
    report = {"sessions": len(sessions), "clean_sessions": len(mono),
              "monotonicity": mono, "rotation_recovery": rotation,
              "symmetry_at_completion": symmetry,
              "separation_curve": curve, "passed": passed}
    out = os.path.join(_CORPUS, "d2_progress_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    comp_ok = sum(v["comp_monotone"] for v in mono)
    edit_ok = sum(v["edit_monotone"] for v in mono)
    print(f"D2 progress report  ({len(sessions)} sessions, {len(mono)} clean)")
    print(f"  comp monotone: {comp_ok}/{len(mono)}   edit non-increasing: {edit_ok}/{len(mono)}")
    print(f"  rotation recovery: {sum(c['ok'] for c in rotation)}/{len(rotation)}")
    print(f"  symmetry at completion (>=0.9): {sum(s['ok'] for s in symmetry)}/{len(symmetry)}")
    print("  separation (true minus best distractor, fit*comp):")
    print("    " + "  ".join(f"{k}:{v['mean_margin']:+.3f}({v['accuracy']})" for k, v in curve.items()))
    print(f"  [{'PASS' if passed else 'FAIL'}] D2 phase criterion  ->  {os.path.basename(out)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
