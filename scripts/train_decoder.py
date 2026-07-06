"""Train the D4 action decoder: Stage A (next-token) then Stage B (MTP retrofit).

    python scripts/train_decoder.py [--stage-a-only]

Reads capture/decoder/ (run scripts/make_decoder_corpus.py first). Ships
models/decoder_v1.pt + decoder_v1.json, and keeps the Stage-A-only checkpoint
(models/decoder_v1_stage_a.pt) because the OQ1 gating check compares the two.

Stage A — next-token warm-start. Every epoch, every sample draws its intent-slot
condition fresh with EQUAL RATIOS over the available slot kinds (contract C5: fixed
equal arm-mask ratios in training augmentation): the tracker's belief (arm3), zeros
(arm0_zero), the implicit classifier (arm1_dense), the LLM reader where its cached
coverage exists (arm2_llm). A slice of the arm3 draws is sharpened to a point-mass
on the TRUE goal (counterfactual goal-slot augmentation, D4 S2's prerequisite: the
forced contexts the intervenability test uses must be in-distribution).

Stage B — the FastSceneScript retrofit, from the Stage-A checkpoint: horizons
1..n (n = 8) through the shared projection block, loss weights lambda_h^(i-1) with
lambda_h = 0.8, plus the agreement-trained confidence for horizons >= 2 with
lambda_c = 0.5. Agreement targets compare each horizon head's pick against the
one-step head's pick at the same position under teacher forcing, with the tau rule
(exact for symbolic tokens, +/-2 for numeric). conf_1 := 1 — never trained, never
read. The confidence is an agreement score, not a calibrated probability (C1/09-F9).

The rationale_goal cross-entropy is active ONLY on arm3-slot samples (contract C2);
its weight and every other knob ships in the provenance json.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.contracts.b4 import NEUTRAL_P_Z1, SLOT_KINDS               # noqa: E402
from mica.contracts.goals import GOALS                               # noqa: E402
from mica.data import decoder_corpus                                 # noqa: E402
from mica.decoder import model as decoder_model                      # noqa: E402
from mica.decoder import tokenizer                                   # noqa: E402
from mica.decoder.grammar import action_from_json                    # noqa: E402

_SEED = 13
_BATCH = 64
_STAGE_A_EPOCHS, _STAGE_A_PATIENCE, _STAGE_A_LR = 40, 5, 3e-4
_STAGE_B_EPOCHS, _STAGE_B_PATIENCE, _STAGE_B_LR = 12, 3, 1e-4
_WEIGHT_DECAY = 0.01
_LAMBDA_H = 0.8          # D4: geometric horizon weights
_LAMBDA_C = 0.5          # D4: confidence loss weight
_LAMBDA_RATIONALE = 0.5  # rationale CE weight (not D4-pinned; recorded here)
_CF_SHARE = 0.15         # fraction of arm3 draws sharpened to a true-goal point-mass


def _mode_entropy(p_z1: float) -> float:
    import math

    if p_z1 <= 0.0 or p_z1 >= 1.0:
        return 0.0
    return -(p_z1 * math.log(p_z1) + (1 - p_z1) * math.log(1 - p_z1))


def _prepare(rows: list[dict]):
    """Corpus rows -> the fixed tensors (token ids padded, evidence, slot bank)."""
    import numpy as np

    prepared = []
    for row in rows:
        actions = [action_from_json(body) for body in row["target"]]
        ids = tokenizer.encode_chunk(actions)
        prepared.append({
            "ids": ids,
            "evidence": np.asarray(row["evidence"], dtype=np.float32),
            "goal_index": GOALS.index(row["goal"]),
            "slots": row["slots"],
        })
    return prepared


def _slot_vector(slot: dict | None):
    import numpy as np

    if slot is None:   # arm0_zero: all-zero marginal, neutral mode
        return np.asarray([0.0] * len(GOALS) + [0.0, 0.0, NEUTRAL_P_Z1], dtype=np.float32)
    return np.asarray(list(slot["goal_marginal"])
                      + [slot["p_top"], slot["entropy_nats"], slot["p_z1"]],
                      dtype=np.float32)


def _draw_slot(sample: dict, rng: random.Random):
    """One training draw: (slot vector, slot-kind index, rationale active?).
    Equal ratios over whichever kinds this sample has (C5)."""
    import numpy as np

    kinds = ["arm3", "arm0_zero"]
    if sample["slots"]["arm1"] is not None:
        kinds.append("arm1_dense")
    if sample["slots"]["arm2"] is not None:
        kinds.append("arm2_llm")
    kind = rng.choice(kinds)
    if kind == "arm0_zero":
        vector = _slot_vector(None)
    elif kind == "arm3":
        slot = sample["slots"]["arm3"]
        if rng.random() < _CF_SHARE:
            # counterfactual sharpening: all goal mass on the true goal, the mode
            # read kept — the exact shape intervened_belief() produces at eval time
            marginal = [0.0] * len(GOALS)
            marginal[sample["goal_index"]] = 1.0
            vector = np.asarray(marginal + [1.0, _mode_entropy(slot["p_z1"]),
                                            slot["p_z1"]], dtype=np.float32)
        else:
            vector = _slot_vector(slot)
    else:
        vector = _slot_vector(sample["slots"]["arm1" if kind == "arm1_dense" else "arm2"])
    return vector, SLOT_KINDS.index(kind), kind == "arm3"


def _batches(prepared, rng: random.Random, batch_size: int, shuffle: bool):
    import numpy as np
    import torch

    order = list(range(len(prepared)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [prepared[i] for i in order[start:start + batch_size]]
        longest = max(len(s["ids"]) for s in chunk)
        ids = np.full((len(chunk), longest), tokenizer.PAD_ID, dtype=np.int64)
        for row, sample in enumerate(chunk):
            ids[row, :len(sample["ids"])] = sample["ids"]
        slots, kinds, rationale_active = [], [], []
        for sample in chunk:
            vector, kind_index, active = _draw_slot(sample, rng)
            slots.append(vector)
            kinds.append(kind_index)
            rationale_active.append(active)
        yield {
            "ids": torch.from_numpy(ids),
            "evidence": torch.from_numpy(np.stack([s["evidence"] for s in chunk])),
            "slot": torch.from_numpy(np.stack(slots)),
            "kinds": torch.tensor(kinds, dtype=torch.long),
            "rationale_active": torch.tensor(rationale_active, dtype=torch.bool),
            "goal_index": torch.tensor([s["goal_index"] for s in chunk], dtype=torch.long),
        }


def _losses(model, batch, device, horizons: int):
    """Token CE per horizon + confidence BCE + rationale CE for one batch."""
    import torch
    import torch.nn.functional as functional

    ids = batch["ids"].to(device)
    inputs = ids[:, :-1]
    hidden = model.trunk_hidden(batch["evidence"].to(device), batch["slot"].to(device),
                                batch["kinds"].to(device), inputs)
    prefix_len = model.config.prefix_len
    token_hidden = hidden[:, prefix_len:, :]           # aligned with inputs
    length = inputs.shape[1]

    total = 0.0
    token_nll_h1 = None
    one_step_picks = None
    for horizon in range(1, horizons + 1):
        # position t predicts ids[t + horizon]; ids index 0 is <bos>
        max_from = length - (horizon - 1)
        if max_from <= 0:
            break
        logits = model.token_logits(token_hidden[:, :max_from, :], horizon - 1)
        labels = ids[:, horizon:horizon + max_from]
        mask = labels != tokenizer.PAD_ID
        if not mask.any():
            continue
        ce = functional.cross_entropy(
            logits[mask], labels[mask], reduction="mean")
        total = total + (_LAMBDA_H ** (horizon - 1)) * ce
        if horizon == 1:
            token_nll_h1 = float(ce.detach())
            one_step_picks = logits.argmax(dim=-1).detach()
        elif one_step_picks is not None:
            # agreement target: this horizon's pick vs the one-step path's pick for
            # the SAME target position, tau rule (targets built without gradients)
            picks = logits.argmax(dim=-1).detach()
            reference = one_step_picks[:, horizon - 1:horizon - 1 + max_from]
            agree = _agreement_matrix(picks, reference)
            conf = model.confidence(token_hidden[:, :max_from, :], horizon - 1).squeeze(-1)
            bce = functional.binary_cross_entropy(
                conf[mask], agree.to(conf.dtype)[mask], reduction="mean")
            total = total + _LAMBDA_C * bce

    rationale_loss = torch.tensor(0.0, device=device)
    active = batch["rationale_active"].to(device)
    if active.any():
        logits = model.rationale_logits(hidden)
        rationale_loss = functional.cross_entropy(
            logits[active], batch["goal_index"].to(device)[active], reduction="mean")
        total = total + _LAMBDA_RATIONALE * rationale_loss
    return total, token_nll_h1, float(rationale_loss.detach())


def _agreement_matrix(picks, reference):
    """Elementwise tau agreement between two id tensors (exact for symbolic tokens,
    +/-NUMERIC_TOLERANCE for numeric), vectorized."""
    import torch

    numeric_low, numeric_high = tokenizer.NUMERIC_ID_RANGE
    both_numeric = ((picks >= numeric_low) & (picks <= numeric_high)
                    & (reference >= numeric_low) & (reference <= numeric_high))
    close = (picks - reference).abs() <= tokenizer.NUMERIC_TOLERANCE
    return torch.where(both_numeric, close, picks == reference)


def _holdout_nll(model, prepared, device, rng_seed: int) -> float:
    """Mean one-step token NLL on the holdout rows (fixed draw for comparability)."""
    import torch

    rng = random.Random(rng_seed)
    model.eval()
    values = []
    with torch.no_grad():
        for batch in _batches(prepared, rng, _BATCH, shuffle=False):
            _, nll, _ = _losses(model, batch, device, horizons=1)
            if nll is not None:
                values.append(nll)
    model.train()
    return sum(values) / len(values)


def _run_stage(model, train_rows, holdout_rows, device, *, horizons, epochs,
               patience, learning_rate, label):
    import copy

    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=_WEIGHT_DECAY)
    rng = random.Random(_SEED)
    best_nll, best_state, best_epoch, history = float("inf"), None, -1, []
    for epoch in range(epochs):
        started = time.time()
        model.train()
        epoch_losses = []
        for batch in _batches(train_rows, rng, _BATCH, shuffle=True):
            loss, _, _ = _losses(model, batch, device, horizons)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        holdout = _holdout_nll(model, holdout_rows, device, _SEED)
        history.append({"epoch": epoch, "train_loss": round(
            sum(epoch_losses) / len(epoch_losses), 4),
            "holdout_token_nll": round(holdout, 4),
            "seconds": round(time.time() - started, 1)})
        print(f"  {label} epoch {epoch}: train {history[-1]['train_loss']}"
              f"  holdout NLL {history[-1]['holdout_token_nll']}"
              f"  ({history[-1]['seconds']}s)")
        if holdout < best_nll:
            best_nll, best_epoch = holdout, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            print(f"  {label}: no improvement for {patience} epochs — stopping")
            break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_holdout_token_nll": round(best_nll, 4),
            "history": history}


def main() -> int:
    import torch

    if not os.path.exists(os.path.join(decoder_corpus.CORPUS_DIR, "decoder_labels.json")):
        print("no decoder corpus found — run scripts/make_decoder_corpus.py first")
        return 1
    train_raw, holdout_raw, labels = decoder_corpus.load_rows()
    # the banked perception-corpus sessions are an eval-only transfer group — the
    # decoder must never train on them, and their rows are not the early-stop signal
    holdout_raw = [row for row in holdout_raw
                   if labels[row["session"]]["group"] == "holdout"]

    torch.manual_seed(_SEED)
    random.seed(_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = decoder_model.DecoderConfig()
    model = decoder_model.build_model(config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())

    train_rows = _prepare(train_raw)
    holdout_rows = _prepare(holdout_raw)
    print(f"decoder: {parameter_count/1e6:.1f}M parameters on {device.type}; "
          f"{len(train_rows)} train / {len(holdout_rows)} holdout samples")

    print("Stage A — next-token warm-start")
    stage_a = _run_stage(model, train_rows, holdout_rows, device, horizons=1,
                         epochs=_STAGE_A_EPOCHS, patience=_STAGE_A_PATIENCE,
                         learning_rate=_STAGE_A_LR, label="stage A")
    torch.save(model.state_dict(), decoder_model.STAGE_A_PATH)

    stage_b = None
    if "--stage-a-only" not in sys.argv:
        print(f"Stage B — MTP retrofit (n={config.mtp_horizon}, "
              f"lambda_h={_LAMBDA_H}, lambda_c={_LAMBDA_C})")
        stage_b = _run_stage(model, train_rows, holdout_rows, device,
                             horizons=config.mtp_horizon,
                             epochs=_STAGE_B_EPOCHS, patience=_STAGE_B_PATIENCE,
                             learning_rate=_STAGE_B_LR, label="stage B")

    torch.save(model.state_dict(), decoder_model.WEIGHTS_PATH)
    with open(os.path.join(decoder_corpus.CORPUS_DIR,
                           "decoder_corpus_report.json"), encoding="utf-8") as handle:
        corpus_report = json.load(handle)
    decoder_model.save_meta(config, {
        "trained": time.strftime("%Y-%m-%d %H:%M"),
        "device": device.type,
        "parameters": parameter_count,
        "seed": _SEED,
        "batch_size": _BATCH,
        "loss_knobs": {"lambda_h": _LAMBDA_H, "lambda_c": _LAMBDA_C,
                       "lambda_rationale": _LAMBDA_RATIONALE,
                       "counterfactual_share": _CF_SHARE,
                       "conf_convention": "conf_1 := 1, never trained (D4)"},
        "slot_ratios": "equal over available kinds per draw (C5); arm2 restricted "
                       "to its cached coverage sessions (recorded in corpus report)",
        "stage_a": stage_a,
        "stage_b": stage_b,
        "corpus": corpus_report,
    })
    print(f"shipped models/decoder_v1.pt + .json (stage A kept at "
          f"{os.path.basename(decoder_model.STAGE_A_PATH)} for the OQ1 check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
