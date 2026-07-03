"""The h3d earn-its-place probe: does the frozen Uni3D EMBEDDING carry goal-category
signal a learned readout can use — even though the zero-shot text cosines cannot?

    python scripts/probe_h3d_linear.py [--per-goal N] [--seed S]

Same corpus and progress bins as scripts/probe_shape3d.py, but instead of reading the
build's cloud against text prompts, this trains a 5-way linear (logistic) readout on
the raw 1024-dim embeddings and evaluates it with SESSIONS held out — one session per
goal leaves the training set each fold, so no session's clouds appear on both sides.
The labels are Source-A generator ground truth, which is exactly the bootstrap role
the D3 data design assigns them.

This is the measurement that decides whether embed() becomes the h3d channel of the
trained adapter in Phase E. Verdict 2026-07-03: PASSED — the linear readout beats the
symbolic structure floor (d2_progress_report.json) at every bin through 70% progress
and trails only late, where the template matcher locks on. Shape signal is in the
embedding; the zero-shot text alignment just cannot reach it on blocky builds.
"""
from __future__ import annotations

import json
import os
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


def _embed_corpus(head, per_goal: int, seed: int):
    """One embedding per (session, progress bin): the corpus replayed exactly as the
    zero-shot probe replays it, with the same checkpoint indices."""
    rows = []
    session_index = 0
    for goal in GOALS:
        for plan in plan_variants(goal, per_goal, seed):
            build, label = build_from_plan(plan)
            session = generate_session(build)
            events = sorted((e for p in session.packets for e in p.server.block_events),
                            key=lambda e: e.event_id)
            world = ReplayWorld(region_around_events(session), {})
            checkpoints = {max(0, round(b * len(events)) - 1): b for b in _BINS}
            for index, event in enumerate(events):
                world.apply(event)
                if index in checkpoints:
                    feats = head.embed(world.built(), seed=plan.seed)
                    if feats is not None:
                        rows.append((feats.cpu(), GOALS.index(label["goal"]),
                                     session_index, checkpoints[index]))
            session_index += 1
    return rows, session_index


def _fit_readout(th, X, y, train_mask):
    weights = th.zeros(X.size(1), len(GOALS), requires_grad=True)
    bias = th.zeros(len(GOALS), requires_grad=True)
    optimizer = th.optim.Adam([weights, bias], lr=0.05, weight_decay=1e-3)
    for _ in range(400):
        optimizer.zero_grad()
        loss = th.nn.functional.cross_entropy(X[train_mask] @ weights + bias, y[train_mask])
        loss.backward()
        optimizer.step()
    return weights.detach(), bias.detach()


def main() -> int:
    ready, missing = assets_ready()
    if not ready:
        print("Uni3D assets missing: " + missing)
        print("run scripts/setup_uni3d.py first")
        return 1

    import torch as th

    per_goal = _flag_value("--per-goal", 6)
    seed = _flag_value("--seed", 7)
    head = Uni3DShapeHead()
    rows, session_count = _embed_corpus(head, per_goal, seed)
    print(f"encoded {len(rows)} clouds from {session_count} sessions")

    X = th.stack([r[0] for r in rows])
    y = th.tensor([r[1] for r in rows])
    sess = th.tensor([r[2] for r in rows])
    bins = [r[3] for r in rows]

    # one held-out session per goal per fold (sessions are goal-blocked, per_goal each)
    per_bin = {b: [0, 0] for b in _BINS}
    overall = [0, 0]
    for fold in range(per_goal):
        holdout = {block * per_goal + fold for block in range(len(GOALS))}
        test = th.tensor([int(s) in holdout for s in sess])
        weights, bias = _fit_readout(th, X, y, ~test)
        with th.no_grad():
            pred = (X[test] @ weights + bias).argmax(-1)
        for p, t, b in zip(pred.tolist(), y[test].tolist(),
                           [bins[i] for i in th.nonzero(test).squeeze(1).tolist()]):
            per_bin[b][0] += p == t
            per_bin[b][1] += 1
            overall[0] += p == t
            overall[1] += 1

    report = {
        "encoder": "uni3d-b (frozen); 5-way linear readout, sessions held out",
        "labels": "Source-A generator ground truth (bootstrap role per D3)",
        "curve": {str(b): round(c / n, 3) if n else None for b, (c, n) in per_bin.items()},
        "overall_accuracy": round(overall[0] / overall[1], 3),
        "chance_level": round(1 / len(GOALS), 3),
        "folds": per_goal,
    }
    out = os.path.join(_ROOT, "capture", "scripted", "h3d_linear_probe.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("linear readout on frozen uni3d embeddings, session-held-out (chance 0.2):")
    print("  " + "  ".join(f"{b}:{v}" for b, v in report["curve"].items()))
    print(f"  overall: {report['overall_accuracy']}")
    print(f"  -> {os.path.relpath(out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
