"""Trained likelihood heads v1 — the Phase-E replacement for the hand-coded v0.

Same interface as heads_v0: likelihood(fused, action) -> the full L_k(g, z) table the
tracker corrects with. The difference is where the numbers come from: v0's transparent
hand-set forms are replaced by the adapter + heads trained by scripts/train_heads.py,
loaded from models/heads_v1.npz (weights) + models/heads_v1.json (vocabulary,
temperatures, fitted filter knobs, provenance). Inference is numpy matrix products —
no torch in the runtime path, mirroring how h3d_readout ships as plain arithmetic.

The temperatures are part of the model: each head's logits divide by its fitted T
before the softmax (calibration by temperature scaling, tuned jointly with ε on
validation — 09-F9's requirement that "calibrated" mean something measured).
tracker_params() returns the TrackerParams whose ε and λ values were fitted alongside,
so a v1 run uses the whole jointly-tuned configuration or none of it.

Loading is lazy and cached; available() says whether a trained model is on disk.

TrainedHeads(name) loads any model set shipped by train_heads --out-name (the
2026-07-12 comparison experiment scores two variants side by side through it); the
module-level functions delegate to the default heads_v1 instance, so every existing
consumer — run_tracker, calibration_report, the live pipeline's rebind — is untouched.
"""
from __future__ import annotations

import json
import os

from ..contracts.b1 import GOALS, MacroAction
from ..contracts.b3 import FusedEvidence
from . import features
from .tracker import Belief, TrackerParams

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _relu(x):
    return x * (x > 0)


def _softmax(logits):
    import numpy as np

    exps = np.exp(logits - logits.max())
    return exps / exps.sum()


class TrainedHeads:
    """One trained model set (models/<name>.npz + .json), lazily loaded and cached."""

    def __init__(self, name: str = "heads_v1"):
        self.name = name
        self._weights = os.path.join(_ROOT, "models", f"{name}.npz")
        self._meta = os.path.join(_ROOT, "models", f"{name}.json")
        self._cache = None

    def available(self) -> bool:
        return os.path.exists(self._weights) and os.path.exists(self._meta)

    def _load(self):
        if self._cache is None:
            import numpy as np

            if not self.available():
                raise FileNotFoundError(
                    f"no trained heads '{self.name}' on disk — run scripts/train_heads.py first")
            arrays = dict(np.load(self._weights))
            with open(self._meta, encoding="utf-8") as handle:
                meta = json.load(handle)
            if tuple(meta["goals"]) != GOALS:
                raise ValueError(f"{self.name} was trained for a different goal taxonomy")
            self._cache = (arrays, meta)
        return self._cache

    def embed(self, fused: FusedEvidence):
        """The frozen adapter's fused embedding e_k for one record — the dense evidence
        summary Phase E trained. D4's context builder reads it through here so the
        decoder sees the SAME e_k the likelihood heads see, from the same weights."""
        import numpy as np

        arrays, meta = self._load()
        vocab = tuple(meta["vocab"])
        held = arrays["held_embed"][features.held_index(fused.state_feats.held_item, vocab)]
        shared = np.asarray(features.shared_dense(fused), dtype=np.float32)
        state = _relu(np.concatenate([held, shared]) @ arrays["w_state"] + arrays["b_state"])
        h2d, f2 = features.channel(fused.h2d, features.H2D_DIM)
        h3d, f3 = features.channel(fused.h3d, features.H3D_DIM)
        trunk_in = np.concatenate([np.asarray(h2d, dtype=np.float32), state,
                                   np.asarray(h3d, dtype=np.float32),
                                   np.asarray([f2, f3], dtype=np.float32)])
        return _relu(_relu(trunk_in @ arrays["w_a1"] + arrays["b_a1"])
                     @ arrays["w_a2"] + arrays["b_a2"])

    def _forward(self, fused: FusedEvidence):
        """One record -> (per-goal deliberative action distribution, heuristic one)."""
        import numpy as np

        arrays, meta = self._load()
        e = self.embed(fused)
        glob = np.asarray(features.global_dense(fused), dtype=np.float32)

        delib = {}
        for index, goal in enumerate(GOALS):
            onehot = np.zeros(len(GOALS), dtype=np.float32)
            onehot[index] = 1.0
            block = np.asarray(features.goal_dense(fused, goal), dtype=np.float32)
            logits = (_relu(np.concatenate([e, glob, block, onehot])
                            @ arrays["w_d1"] + arrays["b_d1"])
                      @ arrays["w_d2"] + arrays["b_d2"])
            delib[goal] = _softmax(logits / meta["temperature_delib"])
        heur_logits = (_relu(np.concatenate([e, glob]) @ arrays["w_h1"] + arrays["b_h1"])
                       @ arrays["w_h2"] + arrays["b_h2"])
        return delib, _softmax(heur_logits / meta["temperature_heur"])

    def likelihood(self, evidence: FusedEvidence, action: MacroAction) -> Belief:
        """The full L_k(g, z) = P(a_k | e_k, g, z) under this model set."""
        index = features.action_index(action)
        delib, heur = self._forward(evidence)
        heur_p = float(heur[index])
        result = {}
        for goal in GOALS:
            result[(goal, 0)] = float(delib[goal][index])
            result[(goal, 1)] = heur_p          # constant in g, by construction
        return result

    def tracker_params(self) -> TrackerParams:
        """The filter knobs fitted jointly with the temperatures — a run uses these,
        not the v0 defaults, or the calibration claim would be about a mixed config."""
        _, meta = self._load()
        return TrackerParams(lambda_g=meta["lambda_g"], lambda_z=meta["lambda_z"],
                             epsilon=meta["epsilon"])


# The default heads_v1 instance and the module-level API every consumer already uses.
_default = TrainedHeads()


def available() -> bool:
    return _default.available()


def embed(fused: FusedEvidence):
    return _default.embed(fused)


def likelihood(evidence: FusedEvidence, action: MacroAction) -> Belief:
    return _default.likelihood(evidence, action)


def tracker_params() -> TrackerParams:
    return _default.tracker_params()
