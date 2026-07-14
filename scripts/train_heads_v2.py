"""Train the heads-v2 per-stream likelihood heads and fit the D8 fusion knobs.

    python scripts/train_heads_v2.py [--epochs N] [--seed S]

The pipeline, in the order D8 pins it (all review fixes folded in):

  1. Load the verified pairs (09-F1 asserted per sample). Weights are
     session-equal x CLASS-BALANCED (D8 §4.3) so the corpus's category skew
     stops becoming an implicit prior.
  2. Train HeadsV2Model: two per-stream heads (2D behavior / 3D structure),
     channel partition structural (review F5), each with the delib/heur
     asymmetry (09-F4). Held-item dropout on ~30% of samples (D7 §3) forces the
     new pooled-inventory channel to carry weight.
  3. Fit the four temperatures on held-out per-head NLL.
  4. Fit the per-goal readout offsets (D8 §4.2, vector scaling) on held-out
     REAL records: o_g minimizes the offset-corrected Arm-1 classifier's goal
     NLL. The classifier itself is Arm 1's frozen backbone (F7c) — this script
     trains no goal head of its own.
  5. Sweep (w2, w3, gamma) x (epsilon, lambda_g, lambda_z) through the ACTUAL
     filter on the held-out sessions, scoring one-step predictive NLPD (review
     F6 — never final-call accuracy). The readout inside the sweep is exactly
     the runtime composition: event-gated (F1), GM-normalized (F3), clipped at
     C_gamma = |G|^gamma (the pinned ceiling, review F2).
  6. Ship models/heads_v2.npz + .json (weights, vocab, temperatures, w2/w3,
     gamma, C_gamma, offsets, Laplace-smoothed p_hat, filter knobs, manifest).

heads_v2 is OPT-IN (--heads v2 on run_tracker/calibration_report); it becomes
anything more only if the D8 §6 pre-registered criteria ALL pass.
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
from mica.intent import arm1, features                             # noqa: E402
from mica.intent.adapter_v2 import HeadsV2Model                    # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_ROOT, "models")
_REPORT = os.path.join(_ROOT, "capture", "raw", "heads_v2_training_report.json")

_T_GRID = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0)
_W2_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
_W3_GRID = (0.5, 0.75, 1.0, 1.25)
_GAMMA_GRID = (0.0, 0.25, 0.5, 1.0)
_EPS_GRID = (0.01, 0.05, 0.1, 0.2, 0.3)
_LAMBDA_G_GRID = (0.02, 0.05, 0.1, 0.2)
_LAMBDA_Z_GRID = (0.5, 1.0, 2.0)

_BATCH = 512
_PATIENCE = 25
_HELD_DROPOUT = 0.3
_MAX_STACKS = 36


def _flag_value(name: str, default: int) -> int:
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


def _sample_weights(samples):
    """Session-equal x class-balanced, normalized to mean 1 (D8 §4.3)."""
    session_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for s in samples:
        session_counts[s.session] = session_counts.get(s.session, 0) + 1
        category_counts[s.goal] = category_counts.get(s.goal, 0) + 1
    weights = [(1.0 / session_counts[s.session]) * (1.0 / category_counts[s.goal])
               for s in samples]
    mean = sum(weights) / len(weights)
    return [w / mean for w in weights]


def _vocab_with_inventory(train):
    """held_item vocab extended with every item name the inventories carry, so the
    pooled channel embeds real items instead of collapsing them all to unknown."""
    base = list(training_pairs.held_item_vocab(train))
    seen = set(base)
    extra = set()
    for s in train:
        inventory = s.fused.state_feats.inventory
        if inventory:
            for item, _ in inventory:
                if item not in seen:
                    extra.add(item)
    return tuple(base + sorted(extra))


def _featurize(samples, vocab, th, device):
    """Samples -> the tensor bundle for BOTH stream heads."""
    held, shared, h2d, f2, h3d, f3, glob = ([] for _ in range(7))
    block2, block3, onehot, target = [], [], [], []
    inv_idx, inv_w, inv_scalars, inv_flag = [], [], [], []
    heur_mask = []
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
        block2.append(features.goal_dense_2d(fused, s.goal))
        block3.append(features.goal_dense_3d(fused, s.goal))
        hot = [0.0] * len(GOALS)
        hot[GOALS.index(s.goal)] = 1.0
        onehot.append(hot)
        target.append(features.action_index(fused.a_hat))
        items = features.inventory_items(fused, vocab)[:_MAX_STACKS]
        idx_row = [i for i, _ in items] + [0] * (_MAX_STACKS - len(items))
        w_row = [w for _, w in items] + [0.0] * (_MAX_STACKS - len(items))
        inv_idx.append(idx_row)
        inv_w.append(w_row)
        scalars, flag = features.inventory_dense(fused)
        inv_scalars.append(scalars)
        inv_flag.append(flag)
        heur_mask.append(s.source == "A" and s.mode == 1)
    weights = _sample_weights(samples)
    tensors = {
        "held": th.tensor(held, dtype=th.long, device=device),
        "shared": th.tensor(shared, dtype=th.float32, device=device),
        "h2d": th.tensor(h2d, dtype=th.float32, device=device),
        "f2": th.tensor(f2, dtype=th.float32, device=device),
        "h3d": th.tensor(h3d, dtype=th.float32, device=device),
        "f3": th.tensor(f3, dtype=th.float32, device=device),
        "glob": th.tensor(glob, dtype=th.float32, device=device),
        "block2": th.tensor(block2, dtype=th.float32, device=device),
        "block3": th.tensor(block3, dtype=th.float32, device=device),
        "onehot": th.tensor(onehot, dtype=th.float32, device=device),
        "target": th.tensor(target, dtype=th.long, device=device),
        "inv_idx": th.tensor(inv_idx, dtype=th.long, device=device),
        "inv_w": th.tensor(inv_w, dtype=th.float32, device=device),
        "inv_scalars": th.tensor(inv_scalars, dtype=th.float32, device=device),
        "inv_flag": th.tensor(inv_flag, dtype=th.float32, device=device),
        "weight": th.tensor(weights, dtype=th.float32, device=device),
        "heur_mask": th.tensor(heur_mask, dtype=th.bool, device=device),
    }
    if tensors["heur_mask"].any():
        w_heur = tensors["weight"] * tensors["heur_mask"]
        tensors["w_heur"] = w_heur / w_heur[tensors["heur_mask"]].mean()
    return tensors


def _embeddings(model, t, index, th, held_drop=None):
    pool = model.pool_inventory(t["inv_idx"][index], t["inv_w"][index])
    e2d = model.embed_2d(t["held"][index], t["shared"][index], pool,
                         t["inv_scalars"][index], t["inv_flag"][index],
                         t["h2d"][index], t["f2"][index], held_drop=held_drop)
    e3d = model.embed_3d(t["h3d"][index], t["f3"][index], t["glob"][index])
    return e2d, e3d


def _losses(model, t, index, th, generator):
    """Both streams' delib + heur NLL over one batch, class-balanced weights.
    The dropout mask draws on CPU (the generator's device) then moves over."""
    drop = (th.rand(len(index), generator=generator) < _HELD_DROPOUT).to(t["held"].device)
    e2d, e3d = _embeddings(model, t, index, th, held_drop=drop)
    weight = t["weight"][index]
    target = t["target"][index]
    loss = th.zeros((), device=e2d.device)
    for logits in (model.delib_logits_2d(e2d, t["block2"][index], t["onehot"][index]),
                   model.delib_logits_3d(e3d, t["block3"][index], t["onehot"][index])):
        nll = th.nn.functional.cross_entropy(logits, target, reduction="none")
        loss = loss + (nll * weight).mean()
    heur_rows = t["heur_mask"][index]
    if heur_rows.any():
        for logits in (model.heur_logits_2d(e2d[heur_rows]),
                       model.heur_logits_3d(e3d[heur_rows])):
            nll = th.nn.functional.cross_entropy(logits, target[heur_rows],
                                                 reduction="none")
            loss = loss + (nll * t["w_heur"][index][heur_rows]).mean()
    return loss


def _head_nll(model, t, th, temperature=1.0, head="2d_delib"):
    """Mean NLL of ONE head over a whole tensor bundle at a temperature. Heuristic
    heads score their Source-A shortcut rows only; None when the bundle has none."""
    with th.no_grad():
        index = th.arange(len(t["target"]), device=t["target"].device)
        e2d, e3d = _embeddings(model, t, index, th)
        if head.endswith("delib"):
            logits = (model.delib_logits_2d(e2d, t["block2"], t["onehot"])
                      if head.startswith("2d")
                      else model.delib_logits_3d(e3d, t["block3"], t["onehot"]))
            targets = t["target"]
        else:
            mask = t["heur_mask"]
            if not mask.any():
                return None
            logits = (model.heur_logits_2d(e2d[mask]) if head.startswith("2d")
                      else model.heur_logits_3d(e3d[mask]))
            targets = t["target"][mask]
        nll = th.nn.functional.cross_entropy(logits / temperature, targets,
                                             reduction="mean")
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


def _fit_offsets(validation, np):
    """Per-goal readout offsets (D8 §4.2) on held-out REAL records: o minimizes the
    offset-corrected Arm-1 goal NLL. Plain gradient descent over 5 numbers, with
    the sum-zero constraint (an overall shift is not identifiable).

    DIVERSITY GUARD (the F7 small-set trap, hit on the first fit 2026-07-13): with
    one single-category real session held out, this fit memorizes that session's
    label as a giant offset (+6 decorative, NLL 0.002) — label leakage dressed as
    bias correction. Offsets activate only when the fit rows span >= 3 categories
    from >= 2 sessions; otherwise they ship as zeros with the reason recorded, and
    the mechanism waits for the eval set to grow."""
    rows = []
    categories, sessions = set(), set()
    for s in validation:
        if s.source != "B":
            continue                              # held-out REAL reliability only
        posterior = arm1.distribution(s.fused)
        rows.append(([posterior[g] for g in GOALS], GOALS.index(s.goal)))
        categories.add(s.goal)
        sessions.add(s.session)
    if not rows or len(categories) < 3 or len(sessions) < 2:
        print(f"  offsets DISABLED: fit set spans {len(categories)} categories / "
              f"{len(sessions)} sessions (needs >= 3 / >= 2) — a fit here would "
              "memorize the holdout's labels, not correct bias")
        return {g: 0.0 for g in GOALS}, 0, None
    probs = np.asarray([r[0] for r in rows], dtype=np.float64)
    truth = np.asarray([r[1] for r in rows], dtype=np.int64)
    offsets = np.zeros(len(GOALS))
    onehot = np.eye(len(GOALS))[truth]
    for _ in range(800):
        adjusted = probs * np.exp(offsets)
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        gradient = (adjusted - onehot).mean(axis=0)
        offsets -= 0.5 * gradient
        offsets -= offsets.mean()                 # sum-zero: only relative tilts exist
    adjusted = probs * np.exp(offsets)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    nll = float(-np.log(adjusted[np.arange(len(truth)), truth] + 1e-12).mean())
    return ({g: round(float(o), 4) for g, o in zip(GOALS, offsets)}, len(rows),
            round(nll, 4))


def _validation_rows(model, validation, vocab, temps, th, device, np):
    """Everything the (w, gamma, eps, lambda) sweep needs, precomputed per record."""
    sessions: dict[str, list] = {}
    with th.no_grad():
        for s in validation:
            t = _featurize([s], vocab, th, device)
            index = th.arange(1, device=device)
            e2d, e3d = _embeddings(model, t, index, th)
            delib2, delib3 = {}, {}
            for gi, goal in enumerate(GOALS):
                hot = th.zeros((1, len(GOALS)), device=device)
                hot[0, gi] = 1.0
                block2 = th.tensor([features.goal_dense_2d(s.fused, goal)],
                                   dtype=th.float32, device=device)
                block3 = th.tensor([features.goal_dense_3d(s.fused, goal)],
                                   dtype=th.float32, device=device)
                delib2[goal] = th.softmax(
                    model.delib_logits_2d(e2d, block2, hot) / temps["2d_delib"],
                    dim=-1)[0].tolist()
                delib3[goal] = th.softmax(
                    model.delib_logits_3d(e3d, block3, hot) / temps["3d_delib"],
                    dim=-1)[0].tolist()
            heur2 = th.softmax(model.heur_logits_2d(e2d) / temps["2d_heur"], dim=-1)[0].tolist()
            heur3 = th.softmax(model.heur_logits_3d(e3d) / temps["3d_heur"], dim=-1)[0].tolist()
            posterior = arm1.distribution(s.fused)
            sessions.setdefault(s.session, []).append({
                "tick": s.fused.tick, "goal": s.goal,
                "target": features.action_index(s.fused.a_hat),
                "delib2": delib2, "delib3": delib3, "heur2": heur2, "heur3": heur3,
                "events": bool(s.fused.event_ids),
                "posterior": [posterior[g] for g in GOALS]})
    return sessions


def _readout(posterior, offsets, p_hat, gamma, np):
    """The runtime composition, vectorized: offsets -> prior division -> gamma ->
    GM normalization -> the C_gamma clip."""
    if gamma == 0.0:
        return np.ones(len(GOALS))
    adjusted = np.asarray(posterior) * np.exp(offsets)
    adjusted /= adjusted.sum()
    raw = (adjusted / p_hat) ** gamma
    log_raw = np.log(raw)
    factor = np.exp(log_raw - log_raw.mean())
    ceiling = len(GOALS) ** gamma
    return np.clip(factor, 1.0 / ceiling, ceiling)


def _filter_score(tables, offsets_vec, p_hat_vec, w2, w3, gamma,
                  epsilon, lambda_g, lambda_z, np):
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
            heur = (row["heur2"][row["target"]] ** w2) * (row["heur3"][row["target"]] ** w3)
            factor = (_readout(row["posterior"], offsets_vec, p_hat_vec, gamma, np)
                      if row["events"] else None)
            like = {}
            for gi, goal in enumerate(GOALS):
                delib = ((row["delib2"][goal][row["target"]] ** w2)
                         * (row["delib3"][goal][row["target"]] ** w3))
                if factor is not None:
                    delib *= float(factor[gi])
                like[(goal, 0)] = delib
                like[(goal, 1)] = heur
            belief, normalizer = correct(belief, like, params)
            log_scores.append(math.log(normalizer))
        marginal = category_marginal(belief)
        finals.append(max(marginal, key=marginal.get) == rows[-1]["goal"])
    return -sum(log_scores) / len(log_scores), sum(finals) / len(finals)


def main() -> int:
    import numpy as np
    import torch as th

    if not arm1.available():
        print("heads v2 needs the Arm-1 classifier on disk (the F7c shared readout "
              "backbone) — run scripts/train_arm1.py first")
        return 1

    epochs = _flag_value("--epochs", 300)
    seed = _flag_value("--seed", 7)
    th.manual_seed(seed)
    device = "cuda" if th.cuda.is_available() else "cpu"

    print("loading verified training pairs (09-F1 asserted per sample) ...")
    source_a = training_pairs.load_source_a()
    source_b = training_pairs.load_source_b()
    samples = source_a + source_b
    train, validation = training_pairs.split(samples)
    vocab = _vocab_with_inventory(train)
    held_sessions = sorted({s.session for s in validation})
    with_inventory = sum(1 for s in train if s.fused.state_feats.inventory is not None)
    print(f"  source A: {len(source_a)}  source B: {len(source_b)}  "
          f"train {len(train)} / validation {len(validation)}")
    print(f"  vocab {len(vocab)} items   train samples with inventory: {with_inventory}")
    print(f"  held-out sessions: {', '.join(held_sessions)}")

    # Laplace-smoothed class prior of the readout backbone's training distribution
    # (review F4): arm1 trains on this same pair universe.
    counts = {g: 0 for g in GOALS}
    for s in train:
        counts[s.goal] += 1
    p_hat = {g: (counts[g] + 1) / (len(train) + len(GOALS)) for g in GOALS}
    print("  p_hat (Laplace):", {g: round(v, 4) for g, v in p_hat.items()})

    t_train = _featurize(train, vocab, th, device)
    t_val = _featurize(validation, vocab, th, device)

    model = HeadsV2Model(len(vocab)).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = th.Generator(device="cpu").manual_seed(seed)
    drop_generator = th.Generator(device="cpu").manual_seed(seed + 1)

    best_nll, best_state, since_best = None, None, 0
    started = time.time()
    n = len(train)
    for epoch in range(epochs):
        model.train()
        for start in range(0, n, _BATCH):
            index = th.randperm(n, generator=generator)[start:start + _BATCH].to(device)
            optimizer.zero_grad()
            loss = _losses(model, t_train, index, th, drop_generator)
            loss.backward()
            optimizer.step()
        model.eval()
        val_nll = (_head_nll(model, t_val, th, head="2d_delib")
                   + _head_nll(model, t_val, th, head="3d_delib"))
        if best_nll is None or val_nll < best_nll - 1e-4:
            best_nll, since_best = val_nll, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
        if epoch % 10 == 0 or since_best == 0:
            print(f"  epoch {epoch:>3}  val delib NLL (2d+3d) {val_nll:.4f}"
                  + ("  *" if since_best == 0 else ""))
        if since_best >= _PATIENCE:
            print(f"  early stop at epoch {epoch} (no improvement for {_PATIENCE})")
            break
    model.load_state_dict(best_state)
    model.eval()

    temps = {}
    for head in ("2d_delib", "3d_delib", "2d_heur", "3d_heur"):
        temperature, nll = _fit_temperature(model, t_val, th, head)
        if temperature is None:
            temperature, nll = _fit_temperature(model, t_train, th, head)
            print(f"  T_{head} fitted on train (no shortcut sessions in the holdout)")
        temps[head] = temperature
        print(f"  T_{head} = {temperature}  (NLL {nll})")

    print("fitting the per-goal readout offsets on held-out REAL records (D8 §4.2) ...")
    offsets, offset_rows, offset_nll = _fit_offsets(validation, np)
    print(f"  offsets over {offset_rows} records (NLL {offset_nll}): "
          + ", ".join(f"{g} {o:+.3f}" for g, o in offsets.items()))

    print("precomputing holdout tables, then sweeping (w2, w3, gamma, eps, lambda) ...")
    tables = _validation_rows(model, validation, vocab, temps, th, device, np)
    offsets_vec = np.asarray([offsets[g] for g in GOALS])
    p_hat_vec = np.asarray([p_hat[g] for g in GOALS])
    grid_rows = []
    for w2 in _W2_GRID:
        for w3 in _W3_GRID:
            for gamma in _GAMMA_GRID:
                for epsilon in _EPS_GRID:
                    for lambda_g in _LAMBDA_G_GRID:
                        for lambda_z in _LAMBDA_Z_GRID:
                            nlpd, accuracy = _filter_score(
                                tables, offsets_vec, p_hat_vec, w2, w3, gamma,
                                epsilon, lambda_g, lambda_z, np)
                            grid_rows.append({
                                "w2": w2, "w3": w3, "gamma": gamma,
                                "epsilon": epsilon, "lambda_g": lambda_g,
                                "lambda_z": lambda_z, "nlpd": round(nlpd, 4),
                                "final_accuracy": round(accuracy, 3)})
        print(f"  ... w2={w2} swept ({len(grid_rows)} rows)")
    grid_rows.sort(key=lambda row: row["nlpd"])
    chosen = grid_rows[0]
    print(f"  chosen: w2 {chosen['w2']}  w3 {chosen['w3']}  gamma {chosen['gamma']}  "
          f"eps {chosen['epsilon']}  lg {chosen['lambda_g']}  lz {chosen['lambda_z']}"
          f"  (NLPD {chosen['nlpd']}, holdout final accuracy {chosen['final_accuracy']})")

    os.makedirs(_MODELS, exist_ok=True)
    np.savez(os.path.join(_MODELS, "heads_v2.npz"), **model.export_arrays())
    meta = {
        "goals": list(GOALS),
        "actions": [a.value for a in features.ACTION_ORDER],
        "vocab": list(vocab),
        "temperature_2d_delib": temps["2d_delib"],
        "temperature_2d_heur": temps["2d_heur"],
        "temperature_3d_delib": temps["3d_delib"],
        "temperature_3d_heur": temps["3d_heur"],
        "w2": chosen["w2"], "w3": chosen["w3"], "gamma": chosen["gamma"],
        # The pinned readout ceiling (review F2): the GM-normalized factor is
        # clipped to [1/C, C]; the per-correction odds cap for a v2 run is
        # 1 + |A|(1-eps)*C_gamma/eps and any v2-era gate freeze must use it.
        "c_gamma": round(len(GOALS) ** chosen["gamma"], 4),
        "goal_offsets": offsets,
        "p_hat": {g: round(v, 6) for g, v in p_hat.items()},
        "readout_backbone": "arm1 (frozen, F7c)",
        "epsilon": chosen["epsilon"],
        "lambda_g": chosen["lambda_g"],
        "lambda_z": chosen["lambda_z"],
        "held_item_dropout": _HELD_DROPOUT,
        "seed": seed,
        "trained": time.strftime("%Y-%m-%d %H:%M"),
        "data": {
            "source_a_pairs": len(source_a), "source_b_pairs": len(source_b),
            "train": len(train), "validation": len(validation),
            "train_with_inventory": with_inventory,
            "held_out_sessions": held_sessions,
            "offset_fit_records": offset_rows,
            "class_balanced": True,
        },
    }
    with open(os.path.join(_MODELS, "heads_v2.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    report = dict(meta)
    report["best_val_delib_nll"] = round(best_nll, 4)
    report["grid_top20"] = grid_rows[:20]
    report["training_seconds"] = round(time.time() - started, 1)
    with open(_REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("shipped models/heads_v2.npz + heads_v2.json   report -> "
          + os.path.relpath(_REPORT, _ROOT))
    print("  heads_v2 is OPT-IN (--heads v2); it ships as anything more only if the "
          "D8 §6 criteria ALL pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
