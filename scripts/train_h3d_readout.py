"""Train the shipped h3d readout: frozen Uni3D embeddings -> per-goal linear weights.

    python scripts/train_h3d_readout.py [--per-goal N] [--seed S]

Same corpus, same embedding checkpoints, and same fitting recipe as the probe that
earned this channel its place (scripts/probe_h3d_linear.py, session-held-out PASS,
2026-07-03). The probe proved the recipe generalizes across sessions; the SHIPPED
readout then trains on every Source-A session, which is the standard split discipline:
validate held-out, ship trained-on-all. Weights land in models/h3d_readout.json, the
file mica/intent/h3d_readout.py loads — writing it is what turns the h3d term on.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.contracts.b1 import GOALS                                # noqa: E402
from mica.perception.shape3d import Uni3DShapeHead, assets_ready  # noqa: E402
from probe_h3d_linear import _embed_corpus, _fit_readout          # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _flag_value(name: str, default: int) -> int:
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ready, missing = assets_ready()
    if not ready:
        print("Uni3D assets missing: " + missing)
        print("run scripts/setup_uni3d.py first")
        return 1

    # Shipping the weights file is what turns the belief-side h3d term ON, so the
    # fusion gate's verdict is enforced here: a FAIL means the term costs accuracy
    # for its earliness on the current corpus, and defaulting it on would ignore the
    # project's own adoption-gate rule. --force overrides, eyes open.
    gate_path = os.path.join(_ROOT, "capture", "scripted", "h3d_fusion_probe.json")
    if os.path.exists(gate_path) and "--force" not in sys.argv:
        with open(gate_path, encoding="utf-8") as handle:
            gate = json.load(handle)
        if gate.get("verdict") != "PASS":
            print(f"REFUSED: the fusion gate's last verdict is {gate.get('verdict')}"
                  f" ({gate.get('earliness_gain'):+.3f} earliness,"
                  f" {gate.get('accuracy_change'):+.3f} accuracy — see {os.path.relpath(gate_path, _ROOT)}).")
            print("re-run scripts/probe_h3d_fusion.py after the heads or data change,"
                  " or pass --force to ship the weights anyway.")
            return 1

    import torch as th

    per_goal = _flag_value("--per-goal", 6)   # the banked corpus: 6 sessions per goal
    seed = _flag_value("--seed", 7)
    head = Uni3DShapeHead()
    rows, session_count = _embed_corpus(head, per_goal, seed)
    print(f"encoded {len(rows)} clouds from {session_count} sessions")

    X = th.stack([r[0] for r in rows])
    y = th.tensor([r[1] for r in rows])
    weights, bias = _fit_readout(th, X, y, th.ones(len(rows), dtype=th.bool))
    with th.no_grad():
        train_accuracy = ((X @ weights + bias).argmax(-1) == y).float().mean().item()

    payload = {
        "goals": list(GOALS),
        # one row of 1024 weights per goal — the shape mica/intent/h3d_readout.py loads
        "weights": weights.T.tolist(),
        "bias": bias.tolist(),
        "recipe": "probe_h3d_linear fit (Adam lr 0.05, wd 1e-3, 400 steps) on all"
                  f" {session_count} Source-A sessions; validated session-held-out by the probe",
        "encoder": "uni3d-b.pt",
        "encoder_sha256": _sha256(os.path.join(_ROOT, "models", "uni3d-b.pt")),
        "corpus_labels_sha256": _sha256(os.path.join(_ROOT, "capture", "scripted", "labels.json")),
        "trained": "2026-07-03",
    }
    out = os.path.join(_ROOT, "models", "h3d_readout.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"train accuracy {train_accuracy:.3f} (held-out generalization is the probe's number)")
    print(f"-> {os.path.relpath(out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
