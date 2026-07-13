"""Train the Phase-E adapter + likelihood heads and fit their calibration knobs.

    python scripts/train_heads.py [--epochs N] [--seed S]

The pipeline, in the order the D3 note pins it:

  1. Load the verified training pairs (mica/data/training_pairs.py re-asserts the
     snapshot rule per sample — a violating pair refuses to load at all).
  2. Train HeadsV1Model (adapter -> e_k, Q_delib, Q_heur) by NLL of the observed
     macro-action. Q_delib trains on Source A (both modes, exact labels) + Source B
     (deliberative supervision); Q_heur trains on Source A's shortcut sessions ONLY,
     reading only goal-free channels. Sessions weigh equally inside each loss so one
     long AFK-heavy capture cannot dominate. NTP only — the MTP-regularized variant
     is a pre-registered gated ablation that needs this baseline to exist first.
  3. Fit the temperatures T_delib / T_heur on held-out head-level NLL, then sweep the
     (epsilon, lambda_g, lambda_z) grid on the held-out SESSIONS through the actual
     filter, scoring the model's one-step predictive of the realized action (the
     normalizer Z_k — a proper score the filter itself computes). Grid search per the
     2026-07-04 user decision; every grid value is pinned in the report.
  4. Ship models/heads_v1.npz + models/heads_v1.json (weights, vocab, temperatures,
     fitted filter knobs, data manifest) + the training report. heads_v1 is opt-in
     (--heads v1 on the offline tools); v0 stays the baseline arm and the live
     default until the calibration report licenses more.

Holdout: the last scripted session per goal (sorted ids) + the pinned real session
(training_pairs.VALIDATION_REAL). The contested free builds have no pairs and stay
eval-only forever.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.contracts.b1 import GOALS                                # noqa: E402
from mica.data import training_pairs                               # noqa: E402
from mica.intent import features                                   # noqa: E402
from mica.intent.adapter import HeadsV1Model                       # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_ROOT, "models")
_META = os.path.join(_MODELS, "heads_v1.json")
_REPORT = os.path.join(_ROOT, "capture", "raw", "heads_v1_training_report.json")

# The pinned grids. Temperatures are fitted per head on held-out NLL; the filter knobs
# are swept jointly through the real filter on held-out sessions.
_T_GRID = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0)
_EPS_GRID = (0.01, 0.05, 0.1, 0.2, 0.3)
_LAMBDA_G_GRID = (0.02, 0.05, 0.1, 0.2)
_LAMBDA_Z_GRID = (0.5, 1.0, 2.0)

_BATCH = 512
_PATIENCE = 25


def _flag_value(name: str, default: int) -> int:
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


def _flag_str(name: str, default: str) -> str:
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def _featurize(samples, vocab, th, device):
    """Samples -> the tensor bundle the model consumes. The true-goal block is what
    Q_delib conditions on in training (NLL under the labeled goal)."""
    held, shared, h2d, f2, h3d, f3, glob, block, onehot, target = ([] for _ in range(10))
    weight_delib, heur_mask, session_ids = [], [], []
    counts: dict[str, int] = {}
    for s in samples:
        counts[s.session] = counts.get(s.session, 0) + 1
    for s in samples:
        fused = s.fused
        held.append(features.held_index(fused.state_feats.held_item, vocab))
        shared.append(features.shared_dense(fused))
        vec, flag = features.channel(fused.h2d, features.H2D_DIM)
        h2d.append(vec)
        f2.append(flag)
        vec, flag = features.channel(fused.h3d, features.H3D_DIM)
        h3d.append(vec)
        f3.append(flag)
        glob.append(features.global_dense(fused))
        block.append(features.goal_dense(fused, s.goal))
        hot = [0.0] * len(GOALS)
        hot[GOALS.index(s.goal)] = 1.0
        onehot.append(hot)
        target.append(features.action_index(fused.a_hat))
        weight_delib.append(1.0 / counts[s.session])   # sessions weigh equally
        heur_mask.append(s.source == "A" and s.mode == 1)
        session_ids.append(s.session)
    tensors = {
        "held": th.tensor(held, dtype=th.long, device=device),
        "shared": th.tensor(shared, dtype=th.float32, device=device),
        "h2d": th.tensor(h2d, dtype=th.float32, device=device),
        "f2": th.tensor(f2, dtype=th.float32, device=device),
        "h3d": th.tensor(h3d, dtype=th.float32, device=device),
        "f3": th.tensor(f3, dtype=th.float32, device=device),
        "glob": th.tensor(glob, dtype=th.float32, device=device),
        "block": th.tensor(block, dtype=th.float32, device=device),
        "onehot": th.tensor(onehot, dtype=th.float32, device=device),
        "target": th.tensor(target, dtype=th.long, device=device),
        "w_delib": th.tensor(weight_delib, dtype=th.float32, device=device),
        "heur_mask": th.tensor(heur_mask, dtype=th.bool, device=device),
    }
    tensors["w_delib"] = tensors["w_delib"] / tensors["w_delib"].mean()
    if tensors["heur_mask"].any():
        w_heur = tensors["w_delib"] * tensors["heur_mask"]
        tensors["w_heur"] = w_heur / w_heur[tensors["heur_mask"]].mean()
    return tensors, session_ids


def _losses(model, t, index, th):
    """Weighted delib + heur NLL over one batch of row indices."""
    e = model.fuse(t["held"][index], t["shared"][index], t["h2d"][index],
                   t["f2"][index], t["h3d"][index], t["f3"][index])
    logits_d = model.delib_logits(e, t["glob"][index], t["block"][index], t["onehot"][index])
    nll_d = th.nn.functional.cross_entropy(logits_d, t["target"][index], reduction="none")
    loss = (nll_d * t["w_delib"][index]).mean()
    heur_rows = t["heur_mask"][index]
    if heur_rows.any():
        logits_h = model.heur_logits(e[heur_rows], t["glob"][index][heur_rows])
        nll_h = th.nn.functional.cross_entropy(
            logits_h, t["target"][index][heur_rows], reduction="none")
        loss = loss + (nll_h * t["w_heur"][index][heur_rows]).mean()
    return loss, nll_d


def _head_nll(model, t, th, temperature=1.0, head="delib"):
    """Mean held-out NLL of one head at a temperature (the T-fitting objective)."""
    with th.no_grad():
        e = model.fuse(t["held"], t["shared"], t["h2d"], t["f2"], t["h3d"], t["f3"])
        if head == "delib":
            logits = model.delib_logits(e, t["glob"], t["block"], t["onehot"])
            mask = th.ones(len(logits), dtype=th.bool, device=logits.device)
        else:
            mask = t["heur_mask"]
            if not mask.any():
                return None
            logits = model.heur_logits(e[mask], t["glob"][mask])
        nll = th.nn.functional.cross_entropy(
            logits / temperature, t["target"][mask], reduction="mean")
    return float(nll)


def _fit_temperature(model, t, th, head):
    best_t, best_nll = 1.0, None
    for temperature in _T_GRID:
        nll = _head_nll(model, t, th, temperature, head)
        if nll is None:
            return None, None
        if best_nll is None or nll < best_nll:
            best_t, best_nll = temperature, nll
    return best_t, round(best_nll, 4)


def _likelihood_tables(model, validation, vocab, t_delib, t_heur, th, device):
    """Per held-out session, in record order: (tick, target index, delib table
    [goal][action], heur table [action]) — everything the filter grid needs, computed
    once so the (eps, lambda) sweep is pure filter arithmetic."""
    from mica.intent.tracker import MODES  # noqa: F401  (documents the (g,z) layout)

    sessions: dict[str, list] = {}
    order: dict[str, list] = {}
    for s in validation:
        order.setdefault(s.session, []).append(s)
    for session_id, samples in order.items():
        rows = []
        with th.no_grad():
            for s in samples:
                fused = s.fused
                held = th.tensor([features.held_index(fused.state_feats.held_item, vocab)],
                                 dtype=th.long, device=device)
                shared = th.tensor([features.shared_dense(fused)], dtype=th.float32, device=device)
                v2, f2 = features.channel(fused.h2d, features.H2D_DIM)
                v3, f3 = features.channel(fused.h3d, features.H3D_DIM)
                e = model.fuse(held, shared,
                               th.tensor([v2], dtype=th.float32, device=device),
                               th.tensor([f2], dtype=th.float32, device=device),
                               th.tensor([v3], dtype=th.float32, device=device),
                               th.tensor([f3], dtype=th.float32, device=device))
                glob = th.tensor([features.global_dense(fused)], dtype=th.float32, device=device)
                delib = {}
                for index, goal in enumerate(GOALS):
                    hot = [0.0] * len(GOALS)
                    hot[index] = 1.0
                    logits = model.delib_logits(
                        e, glob,
                        th.tensor([features.goal_dense(fused, goal)], dtype=th.float32, device=device),
                        th.tensor([hot], dtype=th.float32, device=device))
                    delib[goal] = th.softmax(logits / t_delib, dim=-1)[0].tolist()
                heur = th.softmax(model.heur_logits(e, glob) / t_heur, dim=-1)[0].tolist()
                rows.append({"tick": fused.tick, "goal": s.goal,
                             "target": features.action_index(fused.a_hat),
                             "delib": delib, "heur": heur})
        sessions[session_id] = rows
    return sessions


def _filter_score(tables, epsilon, lambda_g, lambda_z):
    """Run the exact filter over each held-out session with these knobs. The score is
    the mean negative log of Z_k — the model's own predictive probability of the
    realized action — plus final-call accuracy for the report."""
    from mica.intent.tracker import (TrackerParams, category_marginal, correct,
                                     predict, uniform_belief)

    params = TrackerParams(lambda_g=lambda_g, lambda_z=lambda_z, epsilon=epsilon)
    log_scores, finals = [], []
    for session_id, rows in tables.items():
        belief = uniform_belief()
        previous_tick = 0
        for row in rows:
            dt = max(row["tick"] - previous_tick, 1) / 20.0
            previous_tick = row["tick"]
            belief = predict(belief, dt, params)
            like = {}
            for goal in GOALS:
                like[(goal, 0)] = row["delib"][goal][row["target"]]
                like[(goal, 1)] = row["heur"][row["target"]]
            belief, normalizer = correct(belief, like, params)
            log_scores.append(math.log(normalizer))
        marginal = category_marginal(belief)
        finals.append(max(marginal, key=marginal.get) == rows[-1]["goal"])
    return -sum(log_scores) / len(log_scores), sum(finals) / len(finals)


def main() -> int:
    import numpy as np
    import torch as th

    epochs = _flag_value("--epochs", 300)
    seed = _flag_value("--seed", 7)
    # The comparison-experiment knobs (pre-registered 2026-07-12). All default OFF:
    # a no-flag run is byte-for-byte the production training.
    out_name = _flag_str("--out-name", "heads_v1")
    reserved = [s for s in _flag_str("--extra-holdout", "").split(",") if s]
    with_contested = "--include-contested-pairs" in sys.argv
    th.manual_seed(seed)
    device = "cuda" if th.cuda.is_available() else "cpu"

    print("loading verified training pairs (09-F1 asserted per sample) ...")
    source_a = training_pairs.load_source_a()
    source_b = training_pairs.load_source_b()
    contested = training_pairs.load_source_b_contested() if with_contested else []
    if with_contested:
        print(f"  CONTESTED pairs included (comparison arm): {len(contested)} pairs "
              f"from {len({s.session for s in contested})} sessions, builder labels as truth")
    samples = source_a + source_b + contested
    if reserved:
        # Reserved sessions leave ENTIRELY — not even into validation, or the
        # temperature/knob fits would be tuned on the comparison's shared eval set.
        before = len(samples)
        samples = [s for s in samples if s.session not in set(reserved)]
        print(f"  reserved eval sessions excluded from train AND validation: "
              f"{', '.join(reserved)} (-{before - len(samples)} pairs)")
    train, validation = training_pairs.split(samples)
    vocab = training_pairs.held_item_vocab(train)
    held_sessions = sorted({s.session for s in validation})
    print(f"  source A: {len(source_a)} pairs   source B: {len(source_b)} pairs")
    print(f"  train {len(train)} / validation {len(validation)}   vocab {len(vocab)} items")
    print(f"  held-out sessions: {', '.join(held_sessions)}")

    t_train, _ = _featurize(train, vocab, th, device)
    t_val, _ = _featurize(validation, vocab, th, device)
    heur_train = int(t_train["heur_mask"].sum())
    heur_val = int(t_val["heur_mask"].sum())
    print(f"  heuristic-head samples (Source A shortcut only): {heur_train} train / {heur_val} val")

    model = HeadsV1Model(len(vocab)).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = th.Generator(device="cpu").manual_seed(seed)

    best_nll, best_state, since_best = None, None, 0
    started = time.time()
    n = len(train)
    for epoch in range(epochs):
        model.train()
        for start in range(0, n, _BATCH):
            index = th.randperm(n, generator=generator)[start:start + _BATCH].to(device)
            optimizer.zero_grad()
            loss, _ = _losses(model, t_train, index, th)
            loss.backward()
            optimizer.step()
        model.eval()
        val_nll = _head_nll(model, t_val, th)
        if best_nll is None or val_nll < best_nll - 1e-4:
            best_nll, since_best = val_nll, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
        if epoch % 10 == 0 or since_best == 0:
            print(f"  epoch {epoch:>3}  val delib NLL {val_nll:.4f}"
                  + ("  *" if since_best == 0 else ""))
        if since_best >= _PATIENCE:
            print(f"  early stop at epoch {epoch} (no improvement for {_PATIENCE})")
            break
    model.load_state_dict(best_state)
    model.eval()

    t_delib, delib_nll = _fit_temperature(model, t_val, th, "delib")
    t_heur, heur_nll = _fit_temperature(model, t_val, th, "heur")
    heur_fit_source = "validation"
    if t_heur is None:
        # No shortcut sessions landed in the holdout; fit T_heur on its own training
        # samples instead and say so — a v1 without any T_heur would be worse.
        t_heur, heur_nll = _fit_temperature(model, t_train, th, "heur")
        heur_fit_source = "train (no shortcut sessions in the holdout)"
    print(f"  temperatures: T_delib {t_delib} (val NLL {delib_nll}),"
          f" T_heur {t_heur} ({heur_fit_source}, NLL {heur_nll})")

    print("sweeping the (epsilon, lambda_g, lambda_z) grid through the filter on the holdout ...")
    tables = _likelihood_tables(model, validation, vocab, t_delib, t_heur, th, device)
    grid_rows = []
    for epsilon in _EPS_GRID:
        for lambda_g in _LAMBDA_G_GRID:
            for lambda_z in _LAMBDA_Z_GRID:
                nlpd, accuracy = _filter_score(tables, epsilon, lambda_g, lambda_z)
                grid_rows.append({"epsilon": epsilon, "lambda_g": lambda_g,
                                  "lambda_z": lambda_z, "nlpd": round(nlpd, 4),
                                  "final_accuracy": round(accuracy, 3)})
    grid_rows.sort(key=lambda row: row["nlpd"])
    chosen = grid_rows[0]
    print(f"  chosen: epsilon {chosen['epsilon']}  lambda_g {chosen['lambda_g']}"
          f"  lambda_z {chosen['lambda_z']}  (NLPD {chosen['nlpd']},"
          f" holdout final accuracy {chosen['final_accuracy']})")

    os.makedirs(_MODELS, exist_ok=True)
    meta_path = os.path.join(_MODELS, f"{out_name}.json")
    report_path = (_REPORT if out_name == "heads_v1" else
                   os.path.join(_ROOT, "capture", "raw", f"{out_name}_training_report.json"))
    np.savez(os.path.join(_MODELS, f"{out_name}.npz"), **model.export_arrays())
    meta = {
        "goals": list(GOALS),
        "actions": [a.value for a in features.ACTION_ORDER],
        "vocab": list(vocab),
        "temperature_delib": t_delib,
        "temperature_heur": t_heur,
        "epsilon": chosen["epsilon"],
        "lambda_g": chosen["lambda_g"],
        "lambda_z": chosen["lambda_z"],
        "seed": seed,
        "trained": time.strftime("%Y-%m-%d %H:%M"),
        "data": {
            "source_a_pairs": len(source_a), "source_b_pairs": len(source_b),
            "contested_pairs": len(contested),
            "train": len(train), "validation": len(validation),
            "held_out_sessions": held_sessions,
            "reserved_sessions": reserved,
            "heur_samples": {"train": heur_train, "validation": heur_val,
                             "t_heur_fit": heur_fit_source},
        },
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    report = dict(meta)
    report["best_val_delib_nll"] = round(best_nll, 4)
    report["grid"] = grid_rows
    report["training_seconds"] = round(time.time() - started, 1)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"shipped models/{out_name}.npz + {out_name}.json"
          f"   report -> {os.path.relpath(report_path, _ROOT)}")
    print("  heads_v1 is OPT-IN (--heads v1 on run_tracker/calibration_report);"
          " v0 stays the baseline arm and the live default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
