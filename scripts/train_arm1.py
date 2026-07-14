"""Train Arm 1 — the implicit per-step goal classifier of the four-arm comparison.

    python scripts/train_arm1.py [--epochs N] [--seed S]

Same data discipline as the Arm 3 heads (scripts/train_heads.py): the verified
training pairs, the SAME pinned train/holdout split (mica/data/training_pairs.py),
sessions weighted equally AND classes balanced inside the loss (D8 review F7c —
one recipe for this arm and the heads-v2 goal readout that shares this backbone),
early stopping on held-out NLL, then a temperature fitted on the holdout so the
stated confidence is post-hoc calibrated the same way Arm 3's heads were. Goal
labels only — this arm predicts the goal directly; it never sees actions as
targets and carries nothing between steps.

Ships models/arm1.npz + models/arm1.json (weights, vocab, temperature, provenance).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.contracts.b1 import GOALS                                # noqa: E402
from mica.data import training_pairs                               # noqa: E402
from mica.intent.arm1 import input_vector                          # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_ROOT, "models")
_T_GRID = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0)
_BATCH = 512
_PATIENCE = 25
_EMBED = 8
# Small and heavily regularized on purpose: the first training run (256/64 hidden,
# 1e-4 decay, lr 1e-3) memorized the train sessions inside one epoch and scored WORSE
# than uniform on held-out sessions. A per-step classifier over ~6k samples needs to
# be starved, or Arm 1 becomes a strawman instead of a mechanism.
_HIDDEN_1 = 64
_HIDDEN_2 = 32
_LR = 3e-4
_WEIGHT_DECAY = 1e-2
_DROPOUT = 0.3


def _flag_value(name: str, default: int) -> int:
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


def _tensors(samples, vocab, th, device):
    import numpy as np

    counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for s in samples:
        counts[s.session] = counts.get(s.session, 0) + 1
        category_counts[s.goal] = category_counts.get(s.goal, 0) + 1
    held, dense, target, weight = [], [], [], []
    for s in samples:
        idx, vec = input_vector(s.fused, vocab)
        held.append(idx)
        dense.append(vec)
        target.append(GOALS.index(s.goal))
        # Session-equal x CLASS-BALANCED (D8 review F7c: ONE recipe for Arm 1 and
        # the v2 goal readout that shares this backbone) — the corpus's category
        # skew must not become an implicit prior.
        weight.append((1.0 / counts[s.session]) * (1.0 / category_counts[s.goal]))
    w = th.tensor(weight, dtype=th.float32, device=device)
    return {"held": th.tensor(held, dtype=th.long, device=device),
            "dense": th.tensor(np.stack(dense), dtype=th.float32, device=device),
            "target": th.tensor(target, dtype=th.long, device=device),
            "weight": w / w.mean()}


class _Arm1Model:
    def __init__(self, th, vocab_size, dense_dim):
        nn = th.nn
        self.held_embed = nn.Embedding(vocab_size, _EMBED)
        self.net = nn.Sequential(
            nn.Linear(_EMBED + dense_dim, _HIDDEN_1), nn.ReLU(), nn.Dropout(_DROPOUT),
            nn.Linear(_HIDDEN_1, _HIDDEN_2), nn.ReLU(),
            nn.Linear(_HIDDEN_2, len(GOALS)))
        self.modules = nn.ModuleList([self.held_embed, self.net])

    def logits(self, th, t, index):
        x = th.cat([self.held_embed(t["held"][index]), t["dense"][index]], dim=-1)
        return self.net(x)

    def infer_net(self, th):
        """The dropout-free stack in export order (inference never sees dropout)."""
        return [self.net[0], self.net[3], self.net[5]]


def main() -> int:
    import numpy as np
    import torch as th

    epochs = _flag_value("--epochs", 300)
    seed = _flag_value("--seed", 7)
    th.manual_seed(seed)
    device = "cuda" if th.cuda.is_available() else "cpu"

    print("loading verified training pairs (same split as the Arm 3 heads) ...")
    samples = training_pairs.load_source_a() + training_pairs.load_source_b()
    train, validation = training_pairs.split(samples)
    vocab = training_pairs.held_item_vocab(train)
    print(f"  train {len(train)} / validation {len(validation)}   vocab {len(vocab)}")

    t_train = _tensors(train, vocab, th, device)
    t_val = _tensors(validation, vocab, th, device)
    dense_dim = t_train["dense"].shape[1]

    model = _Arm1Model(th, len(vocab), dense_dim)
    model.modules.to(device)
    optimizer = th.optim.Adam(model.modules.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)
    generator = th.Generator(device="cpu").manual_seed(seed)

    def val_nll(temperature=1.0):
        with th.no_grad():
            all_index = th.arange(len(validation), device=device)
            logits = model.logits(th, t_val, all_index) / temperature
            return float(th.nn.functional.cross_entropy(logits, t_val["target"]))

    best_nll, best_state, since_best = None, None, 0
    n = len(train)
    started = time.time()
    for epoch in range(epochs):
        model.modules.train()
        for start in range(0, n, _BATCH):
            index = th.randperm(n, generator=generator)[start:start + _BATCH].to(device)
            optimizer.zero_grad()
            nll = th.nn.functional.cross_entropy(
                model.logits(th, t_train, index), t_train["target"][index], reduction="none")
            loss = (nll * t_train["weight"][index]).mean()
            loss.backward()
            optimizer.step()
        model.modules.eval()
        current = val_nll()
        if best_nll is None or current < best_nll - 1e-4:
            best_nll, since_best = current, 0
            best_state = {k: v.detach().clone() for k, v in model.modules.state_dict().items()}
        else:
            since_best += 1
        if epoch % 10 == 0 or since_best == 0:
            print(f"  epoch {epoch:>3}  val NLL {current:.4f}" + ("  *" if since_best == 0 else ""))
        if since_best >= _PATIENCE:
            print(f"  early stop at epoch {epoch} (no improvement for {_PATIENCE})")
            break
    model.modules.load_state_dict(best_state)
    model.modules.eval()

    temperature, best_t_nll = 1.0, None
    for t in _T_GRID:
        nll = val_nll(t)
        if best_t_nll is None or nll < best_t_nll:
            temperature, best_t_nll = t, nll
    print(f"  temperature {temperature} (val NLL {best_t_nll:.4f})")

    def linear(layer):
        return layer.weight.detach().cpu().numpy().T, layer.bias.detach().cpu().numpy()

    arrays = {"held_embed": model.held_embed.weight.detach().cpu().numpy()}
    for name, layer in zip(("1", "2", "out"), model.infer_net(th)):
        arrays[f"w_{name}"], arrays[f"b_{name}"] = linear(layer)
    os.makedirs(_MODELS, exist_ok=True)
    np.savez(os.path.join(_MODELS, "arm1.npz"), **arrays)
    meta = {
        "goals": list(GOALS), "vocab": list(vocab), "temperature": temperature,
        "seed": seed, "trained": time.strftime("%Y-%m-%d %H:%M"),
        "best_val_nll": round(best_nll, 4),
        "data": {"train": len(train), "validation": len(validation),
                 "held_out_sessions": sorted({s.session for s in validation})},
        "mechanism": "feedforward per-step goal classifier, no recursion (Arm 1)",
        "training_seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(_MODELS, "arm1.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print("shipped models/arm1.npz + arm1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
