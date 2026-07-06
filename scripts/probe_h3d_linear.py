"""The h3d earn-its-place probe: does the frozen Uni3D EMBEDDING carry goal-category
signal a learned readout can use — even though the zero-shot text cosines cannot?

    python scripts/probe_h3d_linear.py [--per-goal N] [--seed S] [--real]

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

--real is the D3-mandated rerun on the labeled real captures: embeddings come from the
banked evidence3d files (no GPU needed), truth is the builder's category, and folds are
leave-one-session-out because N is small (~10 sessions). The scripted corpus is
template-exact and single-builder; this run is the number that counts. The small-N
caveat is printed into the artifact — capture/raw/h3d_linear_probe_real.json.
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


def real_rows():
    """One (embedding, truth, session_index, bin) row per progress decile of each
    labeled real capture, read from the banked evidence3d files — the real-data mirror
    of _embed_corpus. A bin's row is the first h3d-carrying record at or past that
    fraction of the session's human events (h3d is None until something is built).
    Returns the rows plus the session list [(session_id, truth_goal), ...]."""
    from run_tracker import real_labeled_sessions

    rows, sessions = [], []
    for session_id, meta in real_labeled_sessions().items():
        b1 = [json.loads(line) for line in open(meta["b1"], encoding="utf-8") if line.strip()]
        b2 = [json.loads(line) for line in open(meta["b2"], encoding="utf-8") if line.strip()]
        scored = [record for record in b1 if record.get("scored")]
        total = sum(len(record["event_ids"]) for record in scored)
        if total == 0 or not any(record.get("h3d") for record in b2):
            continue
        consumed_after, cumulative = [], 0
        for record in scored:
            cumulative += len(record["event_ids"])
            consumed_after.append(cumulative)
        checkpoints = {}                      # record index -> bin (collisions: later bin wins)
        for b in _BINS:
            target = max(1, round(b * total))
            index = next((i for i, done in enumerate(consumed_after)
                          if done >= target and b2[i].get("h3d")), None)
            if index is not None:
                checkpoints[index] = b
        session_index = len(sessions)
        for index, b in checkpoints.items():
            rows.append((b2[index]["h3d"], meta["goal"], session_index, b))
        sessions.append((session_id, meta["goal"]))
    return rows, sessions


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


def _main_real() -> int:
    """Leave-one-session-out linear readout on the banked real embeddings."""
    import torch as th

    rows, sessions = real_rows()
    if not rows:
        print("no labeled real captures with banked h3d found")
        return 1
    print(f"loaded {len(rows)} banked embeddings from {len(sessions)} real sessions")

    X = th.tensor([r[0] for r in rows])
    y = th.tensor([GOALS.index(r[1]) for r in rows])
    sess = th.tensor([r[2] for r in rows])
    bins = [r[3] for r in rows]

    per_bin = {b: [0, 0] for b in _BINS}
    per_session: dict[str, dict] = {}
    overall = [0, 0]
    for fold in range(len(sessions)):
        test = sess == fold
        if not bool(test.any()):
            continue
        weights, bias = _fit_readout(th, X, y, ~test)
        with th.no_grad():
            pred = (X[test] @ weights + bias).argmax(-1)
        correct = 0
        for p, t, b in zip(pred.tolist(), y[test].tolist(),
                           [bins[i] for i in th.nonzero(test).squeeze(1).tolist()]):
            per_bin[b][0] += p == t
            per_bin[b][1] += 1
            overall[0] += p == t
            overall[1] += 1
            correct += p == t
        session_id, truth = sessions[fold]
        per_session[session_id] = {"truth": truth,
                                   "accuracy": round(correct / int(test.sum()), 3)}

    category_counts: dict[str, int] = {}
    for _, truth in sessions:
        category_counts[truth] = category_counts.get(truth, 0) + 1
    report = {
        "encoder": "uni3d-b (frozen, banked evidence3d); 5-way linear readout",
        "labels": "builder categories from capture/raw/labels.json",
        "caveat": f"small N — {len(sessions)} sessions, category counts {category_counts};"
                  " leave-one-session-out, so each fold trains on every category the"
                  " holdout is not the sole representative of",
        "curve": {str(b): round(c / n, 3) if n else None for b, (c, n) in per_bin.items()},
        "overall_accuracy": round(overall[0] / overall[1], 3),
        "chance_level": round(1 / len(GOALS), 3),
        "folds": len(sessions),
        "per_session": per_session,
    }
    out = os.path.join(_ROOT, "capture", "raw", "h3d_linear_probe_real.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("linear readout on banked REAL uni3d embeddings, leave-one-session-out (chance 0.2):")
    print("  " + "  ".join(f"{b}:{v}" for b, v in report["curve"].items()))
    print(f"  overall: {report['overall_accuracy']}   CAVEAT: {report['caveat']}")
    print(f"  -> {os.path.relpath(out, _ROOT)}")
    return 0


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
