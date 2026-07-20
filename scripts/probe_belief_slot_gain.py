"""Post-hoc probe (2026-07-19, NOT pre-registered — exploratory, labeled as such):
does the inferred belief improve next-action prediction on real sessions?

    python scripts/probe_belief_slot_gain.py

Forces the decoder's intent slot to (a) the tracker's real belief (arm3), (b) the
all-zero no-intent slot, (c) the arm1 classifier, and measures one-step NLL plus
tau accuracy at horizons 1/2/4/8 on the pre-registered real_holdout rows, for both
the production and the pretrained comparison decoder. Writes
capture/raw/belief_slot_probe.json.

Finding on first run: NO gain at any horizon (|delta| <= 0.001) — the belief is
computed from the same evidence stream the decoder already reads, so it is
conditionally redundant for action prediction; its demonstrated value lives in
situation classification (0.042 vs 0.375) and gate licensing, not prediction fuel.
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "capture", "raw", "belief_slot_probe.json")
HORIZONS = (1, 2, 4, 8)


def main() -> int:
    import numpy as np
    import torch

    from mica.contracts.b4 import SLOT_KINDS
    from mica.data import decoder_corpus
    from mica.decoder import model as decoder_model
    from mica.decoder import tokenizer
    from train_decoder import _BATCH, _agreement_matrix, _losses, _prepare, _slot_vector

    rows = decoder_corpus.load_group("real_holdout")
    if not rows:
        print("no real_holdout rows — run make_decoder_corpus.py --real first")
        return 1
    prepared = _prepare(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def batches(kind):
        for start in range(0, len(prepared), _BATCH):
            chunk = prepared[start:start + _BATCH]
            longest = max(len(s["ids"]) for s in chunk)
            ids = np.full((len(chunk), longest), tokenizer.PAD_ID, dtype=np.int64)
            for r, s in enumerate(chunk):
                ids[r, :len(s["ids"])] = s["ids"]
            slots = []
            for s in chunk:
                if kind == "arm3":
                    slots.append(_slot_vector(s["slots"]["arm3"]))
                elif kind == "arm1_dense":
                    slots.append(_slot_vector(s["slots"]["arm1"]))
                else:
                    slots.append(_slot_vector(None))
            yield chunk, ids, slots

    def one_step_nll(model, kind):
        values = []
        with torch.no_grad():
            for chunk, ids, slots in batches(kind):
                batch = {"ids": torch.from_numpy(ids),
                         "evidence": torch.from_numpy(
                             np.stack([s["evidence"] for s in chunk])),
                         "slot": torch.from_numpy(np.stack(slots)),
                         "kinds": torch.tensor([SLOT_KINDS.index(kind)] * len(chunk),
                                               dtype=torch.long),
                         "rationale_active": torch.zeros(len(chunk), dtype=torch.bool),
                         "goal_index": torch.tensor([0] * len(chunk), dtype=torch.long)}
                _, nll, _ = _losses(model, batch, device, horizons=1)
                if nll is not None:
                    values.append(nll)
        return round(sum(values) / len(values), 4)

    def horizon_acc(model, kind, horizon):
        hits = total = 0
        with torch.no_grad():
            for chunk, ids, slots in batches(kind):
                ids_t = torch.from_numpy(ids).to(device)
                hidden = model.trunk_hidden(
                    torch.from_numpy(np.stack([s["evidence"] for s in chunk])).to(device),
                    torch.from_numpy(np.stack(slots)).to(device),
                    torch.tensor([SLOT_KINDS.index(kind)] * len(chunk),
                                 dtype=torch.long).to(device),
                    ids_t[:, :-1])
                token_hidden = hidden[:, model.config.prefix_len:, :]
                max_from = (ids_t.shape[1] - 1) - (horizon - 1)
                if max_from <= 0:
                    continue
                logits = model.token_logits(token_hidden[:, :max_from, :], horizon - 1)
                labels = ids_t[:, horizon:horizon + max_from]
                mask = labels != tokenizer.PAD_ID
                agree = _agreement_matrix(logits.argmax(dim=-1), labels)
                hits += int(agree[mask].sum())
                total += int(mask.sum())
        return round(hits / total, 4)

    report = {"rows": len(prepared), "note": "POST-HOC exploratory probe, not "
              "pre-registered; first run 2026-07-19 found no belief gain at any "
              "horizon", "models": {}}
    arms = [("decoder_v1", decoder_model.WEIGHTS_PATH, decoder_model.META_PATH)]
    cmp_w = os.path.join(_ROOT, "models", "decoder_cmp_real.pt")
    if os.path.exists(cmp_w):
        arms.append(("decoder_cmp_real", cmp_w,
                     os.path.join(_ROOT, "models", "decoder_cmp_real.json")))
    for name, weights, meta in arms:
        model, _, _ = decoder_model.load(weights, meta)
        model.to(device)
        model.eval()
        entry = {"one_step_nll": {k: one_step_nll(model, k)
                                  for k in ("arm3", "arm0_zero", "arm1_dense")},
                 "tau_accuracy_by_horizon": {}}
        for h in HORIZONS:
            entry["tau_accuracy_by_horizon"][str(h)] = {
                "belief": horizon_acc(model, "arm3", h),
                "zeros": horizon_acc(model, "arm0_zero", h)}
        report["models"][name] = entry
        print(f"{name}: one-step NLL belief {entry['one_step_nll']['arm3']} "
              f"vs zeros {entry['one_step_nll']['arm0_zero']}")
        for h in HORIZONS:
            row = entry["tau_accuracy_by_horizon"][str(h)]
            print(f"  h{h}: belief {row['belief']} zeros {row['zeros']} "
                  f"gain {row['belief'] - row['zeros']:+.4f}")
    with open(_OUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print(f"report -> {os.path.relpath(_OUT, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
