"""The heads-v2 trained stack: TWO per-stream likelihood heads, in torch (D8 §2).

TRAINING-SIDE ONLY, like adapter.py — scripts/train_heads_v2.py imports this; the
runtime never does. Inference runs through heads_v2.py (numpy).

Late fusion is the whole point, so the architecture enforces the channel partition
structurally (review F5): the 2D module's signature can only receive behavior-
provenance channels (held item, shared state, h2d, the pooled inventory, s_goal[g]),
the 3D module's only structure-provenance ones (h3d, global feats, the per-goal
structure block). There is no shared trunk — each stream learns its own likelihood
table and the fusion weights w2/w3 combine them AFTER training, fitted on held-out
sessions through the actual filter.

Each stream carries the same delib/heur asymmetry as v1 (09-F4): the heuristic
sub-head has no code path that could see a goal.

The inventory channel (D7 §3) enters the 2D module: pooled learned item embeddings
weighted by log-count + four squashed scalars + a presence flag. Training applies
held-item dropout (the held embedding zeroed on ~30% of samples) so the model must
learn the full-inventory signal instead of the held-item shortcut.
"""
from __future__ import annotations

import torch as th
from torch import nn

from ..contracts.b1 import GOALS
from .features import (
    GLOBAL_DIM, GOAL_2D_DIM, GOAL_3D_DIM, H2D_DIM, H3D_DIM, INVENTORY_DIM, SHARED_DIM,
)

EMBED_DIM = 8        # learned item embedding (held slot and inventory pool share it)
STATE_DIM = 40       # encoded 2D player state (held + shared + inventory summary)
STREAM_DIM = 48      # each stream's own evidence embedding
_TRUNK_HIDDEN = 192
_DELIB_HIDDEN = 48
_HEUR_HIDDEN = 24
_ACTIONS = 5


class HeadsV2Model(nn.Module):
    """Two independent stream modules; nothing is shared between them except the
    item-embedding table (one vocabulary, used only on the 2D side)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.item_embed = nn.Embedding(vocab_size, EMBED_DIM)
        # --- the 2D (behavior) stream ---
        self.state_encoder = nn.Linear(
            EMBED_DIM + SHARED_DIM + EMBED_DIM + INVENTORY_DIM + 1, STATE_DIM)
        self.trunk_2d = nn.Sequential(
            nn.Linear(H2D_DIM + STATE_DIM + 1, _TRUNK_HIDDEN), nn.ReLU(),
            nn.Linear(_TRUNK_HIDDEN, STREAM_DIM), nn.ReLU(),
        )
        self.delib_2d = nn.Sequential(
            nn.Linear(STREAM_DIM + GOAL_2D_DIM + len(GOALS), _DELIB_HIDDEN),
            nn.ReLU(), nn.Linear(_DELIB_HIDDEN, _ACTIONS),
        )
        self.heur_2d = nn.Sequential(
            nn.Linear(STREAM_DIM, _HEUR_HIDDEN),
            nn.ReLU(), nn.Linear(_HEUR_HIDDEN, _ACTIONS),
        )
        # --- the 3D (structure) stream ---
        self.trunk_3d = nn.Sequential(
            nn.Linear(H3D_DIM + GLOBAL_DIM + 1, _TRUNK_HIDDEN), nn.ReLU(),
            nn.Linear(_TRUNK_HIDDEN, STREAM_DIM), nn.ReLU(),
        )
        self.delib_3d = nn.Sequential(
            nn.Linear(STREAM_DIM + GOAL_3D_DIM + len(GOALS), _DELIB_HIDDEN),
            nn.ReLU(), nn.Linear(_DELIB_HIDDEN, _ACTIONS),
        )
        self.heur_3d = nn.Sequential(
            nn.Linear(STREAM_DIM, _HEUR_HIDDEN),
            nn.ReLU(), nn.Linear(_HEUR_HIDDEN, _ACTIONS),
        )

    def embed_2d(self, held_idx, shared, inv_pool, inv_scalars, inv_flag,
                 h2d, h2d_flag, held_drop=None):
        """The 2D stream's evidence embedding. `held_drop` (training only) zeroes
        the held-item embedding per sample — the D7 anti-shortcut lever."""
        held = self.item_embed(held_idx)
        if held_drop is not None:
            held = held * (~held_drop).float().unsqueeze(-1)
        state = th.relu(self.state_encoder(th.cat(
            [held, shared, inv_pool, inv_scalars, inv_flag.unsqueeze(-1)], dim=-1)))
        return self.trunk_2d(th.cat([h2d, state, h2d_flag.unsqueeze(-1)], dim=-1))

    def embed_3d(self, h3d, h3d_flag, global_feats):
        return self.trunk_3d(th.cat(
            [h3d, global_feats, h3d_flag.unsqueeze(-1)], dim=-1))

    def pool_inventory(self, inv_idx, inv_weight):
        """log-count-weighted mean of item embeddings; zero rows pool to zero.
        inv_idx/inv_weight are (batch, MAX_STACKS) with weight 0 on padding."""
        embedded = self.item_embed(inv_idx)                      # (b, s, d)
        weighted = embedded * inv_weight.unsqueeze(-1)
        total = inv_weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return weighted.sum(dim=1) / total

    def delib_logits_2d(self, e2d, goal_block_2d, goal_onehot):
        return self.delib_2d(th.cat([e2d, goal_block_2d, goal_onehot], dim=-1))

    def heur_logits_2d(self, e2d):
        return self.heur_2d(e2d)

    def delib_logits_3d(self, e3d, goal_block_3d, goal_onehot):
        return self.delib_3d(th.cat([e3d, goal_block_3d, goal_onehot], dim=-1))

    def heur_logits_3d(self, e3d):
        return self.heur_3d(e3d)

    def export_arrays(self) -> dict:
        """Weights as plain numpy arrays for heads_v2.py; (in, out)-shaped."""
        def linear(layer):
            return layer.weight.detach().cpu().numpy().T, layer.bias.detach().cpu().numpy()

        arrays = {"item_embed": self.item_embed.weight.detach().cpu().numpy()}
        for name, layer in (("state", self.state_encoder),
                            ("t2a", self.trunk_2d[0]), ("t2b", self.trunk_2d[2]),
                            ("d2a", self.delib_2d[0]), ("d2b", self.delib_2d[2]),
                            ("h2a", self.heur_2d[0]), ("h2b", self.heur_2d[2]),
                            ("t3a", self.trunk_3d[0]), ("t3b", self.trunk_3d[2]),
                            ("d3a", self.delib_3d[0]), ("d3b", self.delib_3d[2]),
                            ("h3a", self.heur_3d[0]), ("h3b", self.heur_3d[2])):
            w, b = linear(layer)
            arrays[f"w_{name}"], arrays[f"b_{name}"] = w, b
        return arrays
