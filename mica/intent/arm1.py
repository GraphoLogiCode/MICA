"""Arm 1 — the IMPLICIT intent mechanism of the four-arm comparison.

A feedforward per-step goal classifier (STEVE1-lineage precedent): it reads the same
fused evidence record the tracker reads — including every per-goal structure block
and the s_goal cosines, so it lacks nothing on the input side — and predicts the goal
category directly. What it deliberately does NOT have is recursion: no belief carried
between steps, no kernels, no likelihood decomposition. Each step is judged alone.
The gap between this arm and Arm 3 is therefore the measured value of the explicit
temporal belief itself, which is the four-arm experiment's question.

Training lives in scripts/train_arm1.py (torch); inference here is numpy over
models/arm1.npz + models/arm1.json, same convention as heads_v1. The stated
confidence is temperature-scaled on validation (fitted at training time) so this
arm's calibration row gets the same post-hoc courtesy Arm 3's heads got — arms are
compared on mechanism, not on who was denied a temperature.
"""
from __future__ import annotations

import json
import os

from ..contracts.b1 import GOALS
from ..contracts.b3 import FusedEvidence
from . import features

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_WEIGHTS = os.path.join(_ROOT, "models", "arm1.npz")
_META = os.path.join(_ROOT, "models", "arm1.json")

_cache = None


def available() -> bool:
    return os.path.exists(_WEIGHTS) and os.path.exists(_META)


def _load():
    global _cache
    if _cache is None:
        import numpy as np

        if not available():
            raise FileNotFoundError("no trained Arm 1 on disk — run scripts/train_arm1.py")
        arrays = dict(np.load(_WEIGHTS))
        with open(_META, encoding="utf-8") as handle:
            meta = json.load(handle)
        if tuple(meta["goals"]) != GOALS:
            raise ValueError("arm1 was trained for a different goal taxonomy")
        _cache = (arrays, meta)
    return _cache


def input_vector(fused: FusedEvidence, vocab: tuple[str, ...]):
    """The flat per-step input: every channel the tracker sees, no more, no less.
    The held item enters as an index the model embeds; everything else is dense."""
    import numpy as np

    h2d, f2 = features.channel(fused.h2d, features.H2D_DIM)
    h3d, f3 = features.channel(fused.h3d, features.H3D_DIM)
    goal_blocks = [v for goal in GOALS for v in features.goal_dense(fused, goal)]
    dense = np.asarray(
        features.shared_dense(fused) + features.global_dense(fused)
        + h2d + h3d + [f2, f3] + goal_blocks, dtype=np.float32)
    return features.held_index(fused.state_feats.held_item, vocab), dense


def distribution(fused: FusedEvidence) -> dict[str, float]:
    """One step -> the classifier's goal distribution (temperature applied)."""
    import numpy as np

    arrays, meta = _load()
    held_idx, dense = input_vector(fused, tuple(meta["vocab"]))
    x = np.concatenate([arrays["held_embed"][held_idx], dense])
    hidden = np.maximum(x @ arrays["w_1"] + arrays["b_1"], 0.0)
    hidden = np.maximum(hidden @ arrays["w_2"] + arrays["b_2"], 0.0)
    logits = (hidden @ arrays["w_out"] + arrays["b_out"]) / meta["temperature"]
    exps = np.exp(logits - logits.max())
    probs = exps / exps.sum()
    return {goal: float(p) for goal, p in zip(GOALS, probs)}
