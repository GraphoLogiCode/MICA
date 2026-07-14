"""Trained likelihood heads v2 — late fusion + the direct goal readout (D8).

Same interface as heads_v0/v1: likelihood(fused, action) -> the full L_k(g, z)
table, tracker_params() -> the jointly fitted filter knobs. What changes is the
composition (D8 §2–§3, all review fixes folded in):

  L(g,0) = L2D(g,0)^w2 · L3D(g,0)^w3 · f̃(g)^[event_ids ≠ ()]
  L(g,1) = L2D_heur^w2 · L3D_heur^w3                  (constant in g, 09-F4)

  f̃(g)  = GM-normalized, ceiling-clipped [ (P(g|e)·e^{o_g}) / P̂(g) ]^γ

- The per-stream tables are raw tempered softmaxes — UNFLOORED. The tracker's
  correct() applies the ε-floor exactly once to this fused table, so the F2
  ordering pin holds with no tracker change and Z_k ≥ ε/|A| stays asserted.
- The goal readout P(g|e) is Arm 1's frozen classifier (user decision F7c) with
  the fitted per-goal offsets o_g (D8 §4.2, vector scaling), divided by the
  Laplace-smoothed training prior P̂(g) (F4), tempered by γ, geometric-mean
  normalized over g (F3 — pairwise goal odds preserved, mode-axis effect bounded)
  and clipped to [1/C_γ, C_γ] so the humility story has a stated ceiling (F2).
- The EVENT GATE (F1, the review's blocker): the factor multiplies in only when
  the correction consumed block events — a static scene must never move the goal
  marginal through the readout.

No belief input anywhere (D8 §5 rail): every quantity here is a function of the
current fused record and frozen weights.
"""
from __future__ import annotations

import json
import math
import os

from ..contracts.b1 import GOALS, MacroAction
from ..contracts.b3 import FusedEvidence
from . import arm1, features
from .tracker import Belief, TrackerParams

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MAX_STACKS = 36     # a player inventory holds at most 36 stacks


def _relu(x):
    return x * (x > 0)


def _softmax(logits):
    import numpy as np

    exps = np.exp(logits - logits.max())
    return exps / exps.sum()


class TrainedHeadsV2:
    """One trained v2 model set (models/<name>.npz + .json), lazily loaded."""

    def __init__(self, name: str = "heads_v2"):
        self.name = name
        self._weights = os.path.join(_ROOT, "models", f"{name}.npz")
        self._meta = os.path.join(_ROOT, "models", f"{name}.json")
        self._cache = None

    def available(self) -> bool:
        return (os.path.exists(self._weights) and os.path.exists(self._meta)
                and arm1.available())

    def _load(self):
        if self._cache is None:
            import numpy as np

            if not self.available():
                raise FileNotFoundError(
                    f"no trained heads '{self.name}' (or no arm1 readout backbone) on "
                    "disk — run scripts/train_heads_v2.py first")
            arrays = dict(np.load(self._weights))
            with open(self._meta, encoding="utf-8") as handle:
                meta = json.load(handle)
            if tuple(meta["goals"]) != GOALS:
                raise ValueError(f"{self.name} was trained for a different goal taxonomy")
            self._cache = (arrays, meta)
        return self._cache

    # ------------------------------------------------------------ stream tables

    def _embed_2d(self, fused: FusedEvidence, arrays, meta):
        import numpy as np

        vocab = tuple(meta["vocab"])
        held = arrays["item_embed"][features.held_index(fused.state_feats.held_item, vocab)]
        pool = np.zeros(arrays["item_embed"].shape[1], dtype=np.float32)
        items = features.inventory_items(fused, vocab)[:_MAX_STACKS]
        if items:
            total = sum(weight for _, weight in items)
            if total > 1e-6:
                for index, weight in items:
                    pool += arrays["item_embed"][index] * weight
                pool /= total
        inv_scalars, inv_flag = features.inventory_dense(fused)
        shared = np.asarray(features.shared_dense(fused), dtype=np.float32)
        state = _relu(np.concatenate([held, shared, pool,
                                      np.asarray(inv_scalars, dtype=np.float32),
                                      np.asarray([inv_flag], dtype=np.float32)])
                      @ arrays["w_state"] + arrays["b_state"])
        h2d, f2 = features.channel(fused.h2d, features.H2D_DIM)
        trunk_in = np.concatenate([np.asarray(h2d, dtype=np.float32), state,
                                   np.asarray([f2], dtype=np.float32)])
        return _relu(_relu(trunk_in @ arrays["w_t2a"] + arrays["b_t2a"])
                     @ arrays["w_t2b"] + arrays["b_t2b"])

    def _embed_3d(self, fused: FusedEvidence, arrays):
        import numpy as np

        h3d, f3 = features.channel(fused.h3d, features.H3D_DIM)
        trunk_in = np.concatenate([np.asarray(h3d, dtype=np.float32),
                                   np.asarray(features.global_dense(fused), dtype=np.float32),
                                   np.asarray([f3], dtype=np.float32)])
        return _relu(_relu(trunk_in @ arrays["w_t3a"] + arrays["b_t3a"])
                     @ arrays["w_t3b"] + arrays["b_t3b"])

    def stream_tables(self, fused: FusedEvidence):
        """((delib_2d[goal], heur_2d), (delib_3d[goal], heur_3d)) — each a tempered
        action distribution; RAW (unfloored), per the F2 ordering pin."""
        import numpy as np

        arrays, meta = self._load()
        e2d = self._embed_2d(fused, arrays, meta)
        e3d = self._embed_3d(fused, arrays)
        delib_2d, delib_3d = {}, {}
        for index, goal in enumerate(GOALS):
            onehot = np.zeros(len(GOALS), dtype=np.float32)
            onehot[index] = 1.0
            block2 = np.asarray(features.goal_dense_2d(fused, goal), dtype=np.float32)
            logits2 = (_relu(np.concatenate([e2d, block2, onehot])
                             @ arrays["w_d2a"] + arrays["b_d2a"])
                       @ arrays["w_d2b"] + arrays["b_d2b"])
            delib_2d[goal] = _softmax(logits2 / meta["temperature_2d_delib"])
            block3 = np.asarray(features.goal_dense_3d(fused, goal), dtype=np.float32)
            logits3 = (_relu(np.concatenate([e3d, block3, onehot])
                             @ arrays["w_d3a"] + arrays["b_d3a"])
                       @ arrays["w_d3b"] + arrays["b_d3b"])
            delib_3d[goal] = _softmax(logits3 / meta["temperature_3d_delib"])
        heur_2d = _softmax((_relu(e2d @ arrays["w_h2a"] + arrays["b_h2a"])
                            @ arrays["w_h2b"] + arrays["b_h2b"])
                           / meta["temperature_2d_heur"])
        heur_3d = _softmax((_relu(e3d @ arrays["w_h3a"] + arrays["b_h3a"])
                            @ arrays["w_h3b"] + arrays["b_h3b"])
                           / meta["temperature_3d_heur"])
        return (delib_2d, heur_2d), (delib_3d, heur_3d)

    # ------------------------------------------------------------ the readout

    def readout_factor(self, fused: FusedEvidence) -> dict[str, float]:
        """f̃(g): offset-corrected classifier over smoothed prior, tempered by γ,
        GM-normalized, ceiling-clipped. Returns all-ones when γ = 0."""
        _, meta = self._load()
        gamma = meta["gamma"]
        if gamma == 0.0:
            return {goal: 1.0 for goal in GOALS}
        posterior = arm1.distribution(fused)
        offsets = meta["goal_offsets"]
        adjusted = {g: posterior[g] * math.exp(offsets[g]) for g in GOALS}
        total = sum(adjusted.values())
        p_hat = meta["p_hat"]
        raw = {g: ((adjusted[g] / total) / p_hat[g]) ** gamma for g in GOALS}
        log_gm = sum(math.log(v) for v in raw.values()) / len(raw)
        ceiling = meta["c_gamma"]
        factor = {}
        for goal in GOALS:
            value = math.exp(math.log(raw[goal]) - log_gm)
            factor[goal] = min(max(value, 1.0 / ceiling), ceiling)
        return factor

    # ------------------------------------------------------------ the interface

    def likelihood(self, evidence: FusedEvidence, action: MacroAction) -> Belief:
        """The fused L_k(g, z) — raw; the tracker's correct() floors it once."""
        _, meta = self._load()
        w2, w3 = meta["w2"], meta["w3"]
        index = features.action_index(action)
        (delib_2d, heur_2d), (delib_3d, heur_3d) = self.stream_tables(evidence)
        heur = (float(heur_2d[index]) ** w2) * (float(heur_3d[index]) ** w3)
        gated = bool(evidence.event_ids)          # the F1 event gate
        factor = self.readout_factor(evidence) if gated else None
        result = {}
        for goal in GOALS:
            delib = ((float(delib_2d[goal][index]) ** w2)
                     * (float(delib_3d[goal][index]) ** w3))
            if factor is not None:
                delib *= factor[goal]
            result[(goal, 0)] = delib
            result[(goal, 1)] = heur              # constant in g, by construction
        return result

    def tracker_params(self) -> TrackerParams:
        _, meta = self._load()
        return TrackerParams(lambda_g=meta["lambda_g"], lambda_z=meta["lambda_z"],
                             epsilon=meta["epsilon"])


_default = TrainedHeadsV2()


def available() -> bool:
    return _default.available()


def likelihood(evidence: FusedEvidence, action: MacroAction) -> Belief:
    return _default.likelihood(evidence, action)


def tracker_params() -> TrackerParams:
    return _default.tracker_params()
