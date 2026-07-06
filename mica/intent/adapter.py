"""The Phase-E trained stack: adapter A_phi and the two likelihood heads, in torch.

This module is TRAINING-SIDE ONLY — scripts/train_heads.py imports it; the runtime
never does. At inference the exported weights run through heads_v1.py (numpy), so the
intent package stays torch-free in the live path. The architecture is the one pinned
in the D3 note's open items (2026-07-04):

  e_k = A_phi( h2d, encode(state_feats), h3d )        concat -> MLP, observation-side
  Q_delib(e_k, g, a)   category-conditioned; ALSO reads the per-goal structure pair,
                       delta_comp and s_goal[g] directly (the B3 passthrough fields)
  Q_heur(e_k, a)       goal-free inputs only — the 09-F4 signature asymmetry is
                       structural: this head has no code path that could see a goal

Missing dense channels (scripted sessions have no pixels; nothing is built before the
first placement) enter as zero vectors with a presence flag, per features.channel().
The adapter never consumes belief — inputs are evidence fields only, by construction.
"""
from __future__ import annotations

import torch as th
from torch import nn

from ..contracts.b1 import GOALS
from .features import GLOBAL_DIM, GOAL_DIM, H2D_DIM, H3D_DIM, SHARED_DIM

EMBED_DIM = 8        # learned held-item embedding
STATE_DIM = 32       # encoded state_feats
FUSED_DIM = 64       # e_k
_ADAPTER_HIDDEN = 256
_DELIB_HIDDEN = 64
_HEUR_HIDDEN = 32
_ACTIONS = 5


class HeadsV1Model(nn.Module):
    """Adapter + both heads as one trainable module (the adapter trunk is shared, so
    both losses shape e_k — that is the point of a fused representation)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.held_embed = nn.Embedding(vocab_size, EMBED_DIM)
        self.state_encoder = nn.Linear(EMBED_DIM + SHARED_DIM, STATE_DIM)
        self.adapter = nn.Sequential(
            nn.Linear(H2D_DIM + STATE_DIM + H3D_DIM + 2, _ADAPTER_HIDDEN), nn.ReLU(),
            nn.Linear(_ADAPTER_HIDDEN, FUSED_DIM), nn.ReLU(),
        )
        self.delib = nn.Sequential(
            nn.Linear(FUSED_DIM + GLOBAL_DIM + GOAL_DIM + len(GOALS), _DELIB_HIDDEN),
            nn.ReLU(), nn.Linear(_DELIB_HIDDEN, _ACTIONS),
        )
        self.heur = nn.Sequential(
            nn.Linear(FUSED_DIM + GLOBAL_DIM, _HEUR_HIDDEN),
            nn.ReLU(), nn.Linear(_HEUR_HIDDEN, _ACTIONS),
        )

    def fuse(self, held_idx, shared, h2d, h2d_flag, h3d, h3d_flag):
        """e_k for a batch. All inputs are evidence; nothing here may carry belief."""
        state = th.relu(self.state_encoder(
            th.cat([self.held_embed(held_idx), shared], dim=-1)))
        return self.adapter(th.cat(
            [h2d, state, h3d, h2d_flag.unsqueeze(-1), h3d_flag.unsqueeze(-1)], dim=-1))

    def delib_logits(self, e, global_feats, goal_block, goal_onehot):
        """Action logits for one (evidence, goal) pairing — the g-conditioned head."""
        return self.delib(th.cat([e, global_feats, goal_block, goal_onehot], dim=-1))

    def heur_logits(self, e, global_feats):
        """Action logits with NO goal anywhere in the signature (09-F4)."""
        return self.heur(th.cat([e, global_feats], dim=-1))

    def export_arrays(self) -> dict:
        """Weights as plain numpy arrays for the heads_v1 inference path. Linear
        layers export as (in, out)-shaped matrices so inference is x @ W + b."""
        def linear(layer):
            return layer.weight.detach().cpu().numpy().T, layer.bias.detach().cpu().numpy()

        arrays = {"held_embed": self.held_embed.weight.detach().cpu().numpy()}
        for name, layer in (("state", self.state_encoder),
                            ("a1", self.adapter[0]), ("a2", self.adapter[2]),
                            ("d1", self.delib[0]), ("d2", self.delib[2]),
                            ("h1", self.heur[0]), ("h2", self.heur[2])):
            w, b = linear(layer)
            arrays[f"w_{name}"], arrays[f"b_{name}"] = w, b
        return arrays
