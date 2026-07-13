"""Frozen MineCLIP head: a POV clip -> s_goal, the per-goal video-text similarity vector.

MineCLIP (MineDojo) was trained to align video clips with task language, so we read its
similarity as goal-progress evidence: for each goal g in G, how much does the recent clip
look like pursuing g. This is the channel that arrives already factored by hypothesis —
the shape the likelihood P(a|e,g,z) wants — and the one that lets goals separate early.
Loaded once, frozen; the goal texts are embedded once. Nothing trains here.

The load recipe (Layer-2 spike): the 'attn' checkpoint in models/, x_transformers==0.27.1 +
kornia, and a two-submodule stub — MineCLIP's __init__ otherwise drags in MineDojo's sim via
its RL reward wrappers, which the encoder never needs.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import torch as th
from PIL import Image

from ..contracts.b1 import GOALS   # the contract goal set (|G|) — fixes the s_goal vector length
from ..contracts.goals import PHRASES, TAXONOMY   # category scores pool subtype-phrase prompts

# A small prompt ensemble per goal (the D1 spec's first s_goal-flatness mitigation): a few phrasings
# averaged at load, with the article picked per goal so it reads "an animal pen" / "a house".
# Public because these define s_goal — run_d1 pins them in the provenance sidecar.
PROMPT_TEMPLATES = ("building {a} {goal}", "a player building {a} {goal}", "constructing {a} {goal} in Minecraft")


def _article(goal: str) -> str:
    return "an" if goal[:1].lower() in "aeiou" else "a"
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CKPT = os.path.join(_ROOT, "models", "mineclip_attn.pth")
_CLIP_LEN = 16          # MineCLIP's native clip length (~0.8 s)
_H, _W = 160, 256       # MineCLIP's native input resolution (H x W)


def _stub_rl_submodules() -> None:
    # MineCLIP's package __init__ imports RL reward wrappers (gym/MineDojo sim). Pre-register
    # those submodules as stubs so only the encoder model loads.
    dense = types.ModuleType("mineclip.dense_reward")
    for name in ("AnimalZooDenseRewardWrapper", "HuntCowDenseRewardEnv",
                 "MobCombatDenseRewardWrapper", "CombatSpiderDenseRewardEnv"):
        setattr(dense, name, type(name, (), {}))
    sys.modules["mineclip.dense_reward"] = dense
    agent = types.ModuleType("mineclip.mineagent")
    agent.__all__ = []
    sys.modules["mineclip.mineagent"] = agent


def _to_chw(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # Pre-size to MineCLIP's native 160x256 so the model's own resize path never runs
    # (it would kornia-resize any other shape). Values stay raw 0-255 uint8: the model
    # divides by 255 and applies its Minecraft mean/std itself.
    arr = np.asarray(image.convert("RGB").resize((_W, _H), Image.BILINEAR), dtype=np.uint8)   # [H, W, 3]
    return arr.transpose(2, 0, 1)                                             # [3, H, W]


class MineClipHead:
    """Loads frozen MineCLIP once, embeds the goal texts, and scores a clip against each goal."""

    def __init__(self, device: str = "cuda"):
        _stub_rl_submodules()
        from mineclip import MineCLIP

        # Config is tied to the 'attn' checkpoint (pool_type=attn...); the 'avg' variant would need a
        # different pool_type. resolution must equal (_H, _W).
        cfg = dict(arch="vit_base_p16_fz.v2.t2", hidden_dim=512, image_feature_dim=512,
                   mlp_adapter_spec="v0-2.t0", pool_type="attn.d2.nh8.glusw", resolution=[_H, _W])
        model = MineCLIP(**cfg).to(device).eval()
        model.load_ckpt(_CKPT, strict=True)
        self.model = model
        self.device = device
        with th.no_grad():
            feats = []
            for goal in GOALS:
                # A category's text anchor is the average of its subtypes' prompt
                # embeddings — "building a cabin", "building a treehouse", ... — because
                # the category names themselves ("habitation") are not things anyone
                # builds, and the pooling is exactly the coarse level of the taxonomy.
                prompts = [t.format(a=_article(PHRASES[s]), goal=PHRASES[s])
                           for s in TAXONOMY[goal] for t in PROMPT_TEMPLATES]
                emb = model.encode_text(prompts)                      # [n_templates, 512]
                emb = emb / emb.norm(dim=-1, keepdim=True)
                feats.append(emb.mean(dim=0))                         # average the templates for this goal
            stacked = th.stack(feats)                                 # [|G|, 512]
            self.goal_feats = stacked / stacked.norm(dim=-1, keepdim=True)

    @th.no_grad()
    def score(self, frames: list, stride: int = 1) -> tuple[float, ...] | None:
        """A POV window -> per-goal cosine similarity through the checkpoint's TRAINED path.

        The clip is the last 16 frames — or, with stride > 1, every stride-th frame
        counted back from the last, so a multi-second pre-action span can be tested
        (MineCLIP trained on clips spanning seconds, not 0.8 s). A short clip pads by
        repeating its earliest frame.
        """
        if not frames:
            return None
        clip = frames[::-1][::stride][:_CLIP_LEN][::-1]
        while len(clip) < _CLIP_LEN:
            clip.insert(0, clip[0])
        arr = np.stack([_to_chw(f) for f in clip])                # [16, 3, 160, 256] uint8
        video = th.from_numpy(arr)[None].to(self.device)          # [1, 16, 3, 160, 256]
        feats = self.model.encode_video(video)
        # The checkpoint's video-text alignment was trained THROUGH the reward head's
        # video adapter and its learned residual gate; similarity read before them
        # lives in a space the text was never aligned to. Apply the trained path but
        # keep the raw cosine — temperature/calibration belong to D3.
        head = self.model.reward_head
        gate = th.sigmoid(head.video_residual_weight)
        feats = gate * feats + (1.0 - gate) * head.video_adapter(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        sim = (feats @ self.goal_feats.T).squeeze(0)              # [5]
        return tuple(sim.float().cpu().tolist())
