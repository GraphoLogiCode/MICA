"""The h3d fusion gate: does the learned shape readout improve the BELIEF, not just a
standalone classifier? Three arms over identical fused records, sessions held out:

    fused + h3d   the deliberative head's centered Uni3D-readout term active
    fused         the same records with the readout forced off (term inert)
    floor         structure-only argmax (fit-first), no behavior, no belief

    python scripts/probe_h3d_fusion.py [--per-goal N] [--seed S] [--real]

Fold discipline matches probe_h3d_linear: one session per goal is held out per fold,
the readout is trained on the remaining sessions' embeddings, and the tracker runs on
the held-out sessions only — so the belief never benefits from weights that saw its
session. The linear probe passing (classifier level) does NOT imply this passes: the
belief already carries fit x comp, delta_comp, and the material cue, and the readout
must add earliness on top of them to earn its default-on place. Verdict and per-arm
numbers land in capture/scripted/h3d_fusion_probe.json.

--real is the adoption-gate rerun the D3 note mandates: fused records from the banked
real evidence files, truth = builder category, leave-one-session-out folds (small N).
The scripted verdict was FAIL (earliness bought at one session's accuracy); on the
free real builds the readout's unique value gets its fair shot. Only a PASS here
licenses shipping the readout weights (train_h3d_readout.py reads this verdict).
Artifact: capture/raw/h3d_fusion_probe_real.json.
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
from mica.contracts.b3 import fuse                                       # noqa: E402
from mica.intent import h3d_readout                                      # noqa: E402
from mica.intent.heads_v0 import _H3D_WEIGHT                             # noqa: E402
from mica.intent.tracker import TrackerParams                            # noqa: E402
from mica.perception.evidence2d import evidence_stream                   # noqa: E402
from mica.perception.evidence3d import build_evidence3d                  # noqa: E402
from mica.perception.shape3d import Uni3DShapeHead, assets_ready         # noqa: E402
from mica.perception.voxel_replay import ReplayWorld, region_around_events  # noqa: E402
from probe_h3d_linear import _embed_corpus, _fit_readout                 # noqa: E402
from run_tracker import _sustained_from, _track                          # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _flag_value(name: str, default: int) -> int:
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


def _fused_sessions(head, per_goal: int, seed: int):
    """Every corpus session as typed fused records WITH h3d, plus its truth label —
    the same goal-blocked session order _embed_corpus uses, so fold indices line up."""
    sessions = []
    for goal in GOALS:
        for plan in plan_variants(goal, per_goal, seed):
            build, label = build_from_plan(plan)
            session = generate_session(build)
            corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
            events = {e.event_id: e for p in session.packets for e in p.server.block_events}
            world = ReplayWorld(region_around_events(session), {})
            b2 = build_evidence3d(world, corrections, events, h3d_fn=head.h3d)
            fused = [fuse(a, b) for a, b in zip(corrections, b2)]
            sessions.append((label["goal"], fused))
    return sessions


def _floor_metrics(truth: str, fused) -> dict:
    calls = [max(f.per_goal, key=lambda g: (f.per_goal[g].fit, f.per_goal[g].comp))
             for f in fused]
    return {"final_correct": calls[-1] == truth,
            "sustained_from": round(_sustained_from(calls, truth), 3)}


def _main_real() -> int:
    """The gate on real data: banked embeddings + banked fused records, one session
    held out per fold. Same three arms, same earn-its-place criterion."""
    import json as json_module

    import torch as th

    from mica.contracts.b3 import fuse_dicts, fuse_streams
    from probe_h3d_linear import real_rows
    from run_tracker import real_labeled_sessions

    params = TrackerParams()
    rows, row_sessions = real_rows()
    if not rows:
        print("no labeled real captures with banked h3d found")
        return 1
    meta = real_labeled_sessions()
    sessions = []
    for session_id, truth in row_sessions:      # same order as the rows' session indices
        b1 = [json_module.loads(line) for line in
              open(meta[session_id]["b1"], encoding="utf-8") if line.strip()]
        b2 = [json_module.loads(line) for line in
              open(meta[session_id]["b2"], encoding="utf-8") if line.strip()]
        fused = fuse_streams(b1, b2, session_id)
        sessions.append((session_id, truth, fused))

    X = th.tensor([r[0] for r in rows])
    y = th.tensor([GOALS.index(r[1]) for r in rows])
    sess = th.tensor([r[2] for r in rows])

    results = {"fused_h3d": [], "fused": [], "floor": []}
    per_session = {}
    try:
        for fold, (session_id, truth, fused) in enumerate(sessions):
            train_mask = sess != fold
            weights, bias = _fit_readout(th, X, y, train_mask)
            fold_readout = h3d_readout.H3dReadout(GOALS, weights.T.tolist(), bias.tolist())
            h3d_readout.activate(fold_readout)
            with_h3d = _track(fused, truth, params)
            h3d_readout.activate(None)          # same records, term inert — clean ablation
            without = _track(fused, truth, params)
            floor = _floor_metrics(truth, fused)
            results["fused_h3d"].append(with_h3d)
            results["fused"].append(without)
            results["floor"].append(floor)
            per_session[session_id] = {"truth": truth, "fused_h3d": with_h3d,
                                       "fused": without, "floor": floor}
    finally:
        h3d_readout.use_default()

    def arm(name):
        entries = results[name]
        return (statistics.mean(e["final_correct"] for e in entries),
                statistics.mean(e["sustained_from"] for e in entries))

    h3d_acc, h3d_early = arm("fused_h3d")
    plain_acc, plain_early = arm("fused")
    floor_acc, floor_early = arm("floor")
    passed = h3d_early < plain_early and h3d_acc >= plain_acc

    report = {
        "question": "does the learned Uni3D readout improve the belief on REAL captures?",
        "data": "banked evidence files; truth = builder category; leave-one-session-out",
        "arms": {
            "fused_h3d": {"final_accuracy": round(h3d_acc, 3), "mean_sustained_from": round(h3d_early, 3)},
            "fused": {"final_accuracy": round(plain_acc, 3), "mean_sustained_from": round(plain_early, 3)},
            "structure_floor": {"final_accuracy": round(floor_acc, 3), "mean_sustained_from": round(floor_early, 3)},
        },
        "earliness_gain": round(plain_early - h3d_early, 3),
        "accuracy_change": round(h3d_acc - plain_acc, 3),
        "folds": len(sessions),
        "sessions": len(sessions),
        "h3d_weight": _H3D_WEIGHT,
        "term_gates": "ramp_in(built_count/8) x unanchored(1 - max fit*comp), both goal-free",
        "verdict": "PASS" if passed else "FAIL",
        "per_session": per_session,
    }
    out = os.path.join(_ROOT, "capture", "raw", "h3d_fusion_probe_real.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"h3d fusion gate over {len(sessions)} REAL sessions, leave-one-session-out"
          " (lower earliness = earlier):")
    print(f"  fused+h3d: accuracy {h3d_acc:.3f}  sustained-from {h3d_early:.3f}")
    print(f"  fused:     accuracy {plain_acc:.3f}  sustained-from {plain_early:.3f}")
    print(f"  floor:     accuracy {floor_acc:.3f}  sustained-from {floor_early:.3f}")
    print(f"  readout contribution: {plain_early - h3d_early:+.3f} earliness,"
          f" {h3d_acc - plain_acc:+.3f} accuracy  ->  [{report['verdict']}]")
    print(f"  -> {os.path.relpath(out, _ROOT)}")
    return 0 if passed else 1


def main() -> int:
    if "--real" in sys.argv:
        return _main_real()

    ready, missing = assets_ready()
    if not ready:
        print("Uni3D assets missing: " + missing)
        print("run scripts/setup_uni3d.py first")
        return 1

    import torch as th

    per_goal = _flag_value("--per-goal", 6)
    seed = _flag_value("--seed", 7)
    params = TrackerParams()
    head = Uni3DShapeHead()

    print("embedding the corpus (training rows + per-correction clouds) ...")
    rows, session_count = _embed_corpus(head, per_goal, seed)
    sessions = _fused_sessions(head, per_goal, seed)
    X = th.stack([r[0] for r in rows])
    y = th.tensor([r[1] for r in rows])
    sess = th.tensor([r[2] for r in rows])

    results = {"fused_h3d": [], "fused": [], "floor": []}
    try:
        for fold in range(per_goal):
            holdout = {block * per_goal + fold for block in range(len(GOALS))}
            train_mask = th.tensor([int(s) not in holdout for s in sess])
            weights, bias = _fit_readout(th, X, y, train_mask)
            fold_readout = h3d_readout.H3dReadout(GOALS, weights.T.tolist(), bias.tolist())
            for index in sorted(holdout):
                truth, fused = sessions[index]
                h3d_readout.activate(fold_readout)
                results["fused_h3d"].append(_track(fused, truth, params))
                h3d_readout.activate(None)      # same records, term inert — clean ablation
                results["fused"].append(_track(fused, truth, params))
                results["floor"].append(_floor_metrics(truth, fused))
    finally:
        h3d_readout.use_default()

    def arm(name):
        entries = results[name]
        return (statistics.mean(e["final_correct"] for e in entries),
                statistics.mean(e["sustained_from"] for e in entries))

    h3d_acc, h3d_early = arm("fused_h3d")
    plain_acc, plain_early = arm("fused")
    floor_acc, floor_early = arm("floor")
    # Earn-its-place: earlier sustained-correct at no accuracy cost, on held-out sessions.
    passed = h3d_early < plain_early and h3d_acc >= plain_acc

    report = {
        "question": "does the learned Uni3D readout improve the belief, sessions held out?",
        "arms": {
            "fused_h3d": {"final_accuracy": round(h3d_acc, 3), "mean_sustained_from": round(h3d_early, 3)},
            "fused": {"final_accuracy": round(plain_acc, 3), "mean_sustained_from": round(plain_early, 3)},
            "structure_floor": {"final_accuracy": round(floor_acc, 3), "mean_sustained_from": round(floor_early, 3)},
        },
        "earliness_gain": round(plain_early - h3d_early, 3),
        "accuracy_change": round(h3d_acc - plain_acc, 3),
        "folds": per_goal,
        "sessions": len(sessions),
        "h3d_weight": _H3D_WEIGHT,
        "term_gates": "ramp_in(built_count/8) x unanchored(1 - max fit*comp), both goal-free",
        "verdict": "PASS" if passed else "FAIL",
    }
    out = os.path.join(_ROOT, "capture", "scripted", "h3d_fusion_probe.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"h3d fusion gate over {len(sessions)} sessions, {per_goal} folds (lower earliness = earlier):")
    print(f"  fused+h3d: accuracy {h3d_acc:.3f}  sustained-from {h3d_early:.3f}")
    print(f"  fused:     accuracy {plain_acc:.3f}  sustained-from {plain_early:.3f}")
    print(f"  floor:     accuracy {floor_acc:.3f}  sustained-from {floor_early:.3f}")
    print(f"  readout contribution: {plain_early - h3d_early:+.3f} earliness,"
          f" {h3d_acc - plain_acc:+.3f} accuracy  ->  [{report['verdict']}]")
    print(f"  -> {os.path.relpath(out, _ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
