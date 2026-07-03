"""MICA's thin loader for Uni3D — stubs the one compiled dependency, builds the model.

Uni3D's point tokenizer imports `pointnet2_ops` (a CUDA extension that is painful to
compile on Windows) for exactly one thing: farthest point sampling. For offline batches
of a few thousand points a pure-torch FPS is plenty, so we register a stub module BEFORE
importing the encoder — the same pattern MICA uses for minerl and mineclip stubs.

Config matches the repo's inference script for uni3d-b: timm `eva02_base_patch14_448`
backbone, feat dim 768, CLIP-teacher embed dim 1024 (EVA02-E-14-plus), 512 groups of 64.
Weights come from the released checkpoint, so timm builds the backbone uninitialized.
"""
from __future__ import annotations

import sys
import types

import torch


def _pure_torch_fps(xyz: torch.Tensor, count: int) -> torch.Tensor:
    """Farthest point sampling, batch 1..B, O(count * N) — fine for offline scoring."""
    batch, n, _ = xyz.shape
    indices = torch.zeros(batch, count, dtype=torch.long, device=xyz.device)
    distances = torch.full((batch, n), float("inf"), device=xyz.device)
    farthest = torch.zeros(batch, dtype=torch.long, device=xyz.device)
    for i in range(count):
        indices[:, i] = farthest
        chosen = xyz[torch.arange(batch), farthest].unsqueeze(1)      # [B, 1, 3]
        distances = torch.minimum(distances, ((xyz - chosen) ** 2).sum(-1))
        farthest = distances.argmax(-1)
    return indices.to(torch.int32)


def _gather(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # points [B, C, N], indices [B, M] -> [B, C, M]  (pointnet2's gather_operation shape)
    expanded = indices.long().unsqueeze(1).expand(-1, points.size(1), -1)
    return points.gather(2, expanded)


def _stub_pointnet2() -> None:
    ops = types.ModuleType("pointnet2_ops")
    utils = types.ModuleType("pointnet2_ops.pointnet2_utils")
    utils.furthest_point_sample = _pure_torch_fps
    utils.gather_operation = _gather
    ops.pointnet2_utils = utils
    sys.modules["pointnet2_ops"] = ops
    sys.modules["pointnet2_ops.pointnet2_utils"] = utils

    # models/uni3d.py does `from . import losses`, which drags in the repo's whole
    # training stack (h5py, dataset loaders). Inference never touches it, so satisfy
    # the import with an empty placeholder instead of installing training deps.
    losses = types.ModuleType("models.losses")
    losses.Uni3d_Text_Image_Loss = None
    sys.modules["models.losses"] = losses


class _Args:
    """The argparse namespace the repo's constructors expect, pinned for uni3d-b."""

    pc_model = "eva02_base_patch14_448"
    pretrained_pc = ""          # backbone starts blank; the released checkpoint fills it
    drop_path_rate = 0.0
    pc_feat_dim = 768
    embed_dim = 1024            # EVA02-E-14-plus embedding width
    group_size = 64
    num_group = 512
    pc_encoder_dim = 512    # released uni3d-b weights use 512, not the code comment's 256
    patch_dropout = 0.0


def load_uni3d(checkpoint_path: str, device: str = "cuda"):
    _stub_pointnet2()
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.uni3d import create_uni3d

    model = create_uni3d(_Args())
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("module", checkpoint)
    state = { (k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items() }
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.startswith("point_encoder.visual.head")]
    if real_missing or unexpected:
        raise RuntimeError(f"uni3d checkpoint mismatch: missing {real_missing[:5]},"
                           f" unexpected {unexpected[:5]}")
    return model.to(device).eval()
