"""Turning one fused evidence record into the number lists the trained model reads.

This is the single definition both sides use — scripts/train_heads.py builds its
training tensors from these functions, and heads_v1.py rebuilds the exact same vectors
at inference time — so a feature can never mean one thing in training and another in
the live filter.

The goal-symmetry rule shapes the split (contracts/b3.py):
  shared_dense / global_dense / the h2d+h3d channels are goal-free — both heads and
  the adapter may read them.
  goal_dense(goal) carries the g-indexed signals (per-goal structure pair, s_goal
  cosine) — DELIBERATIVE head only. Feeding it to the heuristic head would break mode
  identifiability, so no heuristic code path ever calls it.

a_hat_conf is deliberately absent everywhere (D3 constraint N1: rule-certainty
metadata, not a likelihood factor). Ranges are squashed to roughly [0, 1] with fixed
constants so no training-set statistics need to ship with the model.
"""
from __future__ import annotations

import math

from ..contracts.b1 import GOALS, MacroAction
from ..contracts.b3 import FusedEvidence

# The emitted action classes, in the model's output order (SCAFFOLD folds into PLACE,
# same as heads_v0.ACTIONS).
ACTION_ORDER = (MacroAction.PLACE, MacroAction.BREAK, MacroAction.NAVIGATE,
                MacroAction.INSPECT, MacroAction.IDLE)
# recent_actions strings may still carry "scaffold"; it gets its own histogram slot.
_RECENT_ORDER = ("place", "break", "navigate", "inspect", "idle", "scaffold")

H2D_DIM = 1024
H3D_DIM = 1024
SHARED_DIM = 14
GLOBAL_DIM = 9
GOAL_DIM = 7


def action_index(a_hat: MacroAction) -> int:
    """The target class of a sample. SCAFFOLD reports as PLACE in v1."""
    if a_hat == MacroAction.SCAFFOLD:
        a_hat = MacroAction.PLACE
    return ACTION_ORDER.index(a_hat)


def held_index(held_item: str, vocab: tuple[str, ...]) -> int:
    """Index into the learned held-item embedding; 0 is the unknown-item slot."""
    try:
        return vocab.index(held_item)
    except ValueError:
        return 0


def shared_dense(fused: FusedEvidence) -> list[float]:
    """Goal-free player state: what were they just doing, how did they move and look."""
    recent = fused.state_feats.recent_actions
    hist = [(sum(1 for a in recent if a == name) / len(recent)) if recent else 0.0
            for name in _RECENT_ORDER]
    px, py, pz = fused.state_feats.pos_delta
    return hist + [
        max(-1.0, min(1.0, px / 8.0)),
        max(-1.0, min(1.0, py / 8.0)),
        max(-1.0, min(1.0, pz / 8.0)),
        max(-1.0, min(1.0, fused.state_feats.yaw_delta / 180.0)),
        max(-1.0, min(1.0, fused.state_feats.pitch_delta / 90.0)),
        min(fused.focus.dwell_ticks, 20) / 20.0 if fused.focus.block else 0.0,
        1.0 if fused.focus.block else 0.0,
        1.0 if fused.idle else 0.0,
    ]


def global_dense(fused: FusedEvidence) -> list[float]:
    """Goal-free structure summary: how much is built and what shape facts hold."""
    g = fused.global_feats
    if g.bbox:
        dims = [min(abs(float(v)), 16.0) / 16.0 for v in g.bbox[:3]]
    else:
        dims = [0.0, 0.0, 0.0]
    return [
        min(math.log1p(g.built_count) / 5.0, 1.0),
        min(g.planar_runs, 10) / 10.0,
        1.0 if g.has_enclosure else 0.0,
        g.symmetry if g.symmetry is not None else 0.0,
        min(g.symmetry_support, 10) / 10.0,
        # None until there are edits to be local to — same absent-vs-zero care as the
        # dense channels, but a scalar: absent reads as 0 here (locality unknown).
        g.edit_locality if g.edit_locality is not None else 0.0,
    ] + dims


def goal_dense(fused: FusedEvidence, goal: str) -> list[float]:
    """The g-indexed block — deliberative head ONLY. comp never travels without fit
    (D3 constraint 4); delta_comp is exempt from the pair rule (12-F7)."""
    feats = fused.per_goal[goal]
    if fused.s_goal is not None:
        s_g, s_present = fused.s_goal[GOALS.index(goal)], 1.0
    else:
        s_g, s_present = 0.0, 0.0
    return [
        feats.fit,
        feats.comp,
        feats.fit * feats.comp,
        max(-1.0, min(1.0, feats.delta_comp * 3.0)),
        min(feats.edit_distance, 50) / 50.0,
        s_g,
        s_present,
    ]


INVENTORY_DIM = 4


def inventory_dense(fused: FusedEvidence) -> tuple[list[float], float]:
    """Goal-free inventory summary + presence flag — heads v2 ONLY (D7 §3).

    NOT wired into the v1 vectors above: heads_v1 shipped against those exact
    shapes, and a loaded model must never meet a feature it was not trained on.
    The v2 trainer consumes this alongside a learned per-item embedding pooled by
    log-count, and trains with held-item dropout (~30% of samples zero the held
    embedding) so the full-inventory signal must carry weight — the D7 lever
    against the held-item shortcut. The leakage rule is upstream: state_feats
    carries only a sample from before the record's own run (asserted per record
    by the validator and per sample by the v2 trainer, review F1).

    Four scalars, squashed like every other channel: total stock, distinct item
    kinds, placeable-block share, and how deep the largest single stack is."""
    inventory = fused.state_feats.inventory
    if inventory is None:
        return [0.0] * INVENTORY_DIM, 0.0
    total = sum(count for _, count in inventory)
    blocks = sum(count for item, count in inventory
                 if "sword" not in item and "pickaxe" not in item and "axe" not in item
                 and "shovel" not in item and "hoe" not in item and "bucket" not in item)
    largest = max((count for _, count in inventory), default=0)
    return [
        min(math.log1p(total) / 7.0, 1.0),
        min(len(inventory), 27) / 27.0,
        (blocks / total) if total else 0.0,
        min(largest, 64) / 64.0,
    ], 1.0


def channel(vec: tuple[float, ...] | None, dim: int) -> tuple[list[float], float]:
    """A dense model channel plus its presence flag: zeros when the frozen model did
    not run (no frames in the window, nothing built yet). The flag lets the trained
    trunk tell 'absent' from 'genuinely zero'."""
    if vec is None:
        return [0.0] * dim, 0.0
    return list(vec), 1.0
