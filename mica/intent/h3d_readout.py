"""The learned reader for the h3d channel: Uni3D shape embedding in, per-goal scores out.

The frozen Uni3D encoder cannot be read with zero-shot text prompts (that probe failed
the earn-its-place gate — shape3d.py's docstring tells the story), but a small linear
readout trained on labeled builds CAN read it (that probe passed, session-held-out).
This module is that readout at inference time: five dot products and a softmax, pure
stdlib, so the intent package stays torch-free. The weights are trained offline by
scripts/train_h3d_readout.py and shipped as models/h3d_readout.json.

The scores are g-indexed by construction (trained against goal labels), so they may
reach the DELIBERATIVE head only — same goal-symmetry rule as every other goal-bearing
signal. heads_v0 consumes them centered (score minus 1/|G|), which makes the term
zero-sum across goals: it discriminates between goals and cannot shift the base action
rates (the balance law).

Resolution order: an explicitly activated readout (probes use this to swap per-fold
weights, or to force the term OFF with activate(None)) beats the default weights file;
if neither exists, scores() returns None and the head term is simply absent.
"""
from __future__ import annotations

import json
import math
import os

from ..contracts.b1 import GOALS

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_WEIGHTS_PATH = os.path.join(_ROOT, "models", "h3d_readout.json")

_USE_DEFAULT = object()      # sentinel: "no explicit activation — use the shipped file"
_active = _USE_DEFAULT
_default_cache = _USE_DEFAULT  # lazily resolved: H3dReadout, or None when no file exists


class H3dReadout:
    """A linear per-goal readout: logits = W·h3d + b, softmaxed over the goal set."""

    def __init__(self, goals, weights, bias):
        self.goals = tuple(goals)
        self.weights = [tuple(float(v) for v in row) for row in weights]  # one row per goal
        self.bias = tuple(float(v) for v in bias)
        if self.goals != GOALS:
            raise ValueError("readout was trained for a different goal taxonomy")
        if len(self.weights) != len(self.goals) or len(self.bias) != len(self.goals):
            raise ValueError("readout shape does not match the goal set")

    def scores(self, h3d) -> dict[str, float]:
        """P(goal | shape embedding) under the readout — a softmax over five dot products."""
        logits = [sum(w * x for w, x in zip(row, h3d)) + b
                  for row, b in zip(self.weights, self.bias)]
        peak = max(logits)
        exps = [math.exp(value - peak) for value in logits]
        total = sum(exps)
        return {goal: value / total for goal, value in zip(self.goals, exps)}


def activate(readout: H3dReadout | None) -> None:
    """Override the shipped weights: a readout to use it, None to force the term OFF.
    Probes swap per-fold weights this way; tests force determinism this way."""
    global _active
    _active = readout


def use_default() -> None:
    """Back to the shipped weights file (or to no term at all if none is shipped)."""
    global _active
    _active = _USE_DEFAULT


def _default() -> H3dReadout | None:
    global _default_cache
    if _default_cache is _USE_DEFAULT:
        if os.path.exists(_WEIGHTS_PATH):
            with open(_WEIGHTS_PATH, encoding="utf-8") as handle:
                data = json.load(handle)
            _default_cache = H3dReadout(data["goals"], data["weights"], data["bias"])
        else:
            _default_cache = None
    return _default_cache


def scores(h3d) -> dict[str, float] | None:
    """Per-goal scores for one embedding, or None when no readout is available."""
    readout = _active if _active is not _USE_DEFAULT else _default()
    if readout is None or h3d is None:
        return None
    return readout.scores(h3d)
