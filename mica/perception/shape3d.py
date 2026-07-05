"""Uni3D shape head: the player's build as a point cloud -> per-category CLIP-space
similarity (s_shape) — the pretrained-3D analog of s_goal.

Uni3D (BAAI, ICLR 2024) is a ViT point-cloud encoder contrastively aligned to a CLIP
teacher's text/image space, pretrained on ~1M Objaverse-scale shapes. We repurpose its
zero-shot classification exactly as MineCLIP was repurposed: embed the goal taxonomy's
subtype prompts once with the PAIRED CLIP text tower, embed the build's point cloud per
correction, and read the cosine per category (subtype prompts pooled). The output is
category-factored — DELIBERATIVE HEAD ONLY under the goal-symmetry rule — and it had to
EARN fusion by beating the symbolic floor on the labeled corpus first.

Probe verdicts (2026-07-03, scripted corpus, session-held-out where trained):
  * score() zero-shot cosines FAILED the gate — near chance during construction, 0.5 at
    completion vs the symbolic channel's 0.967 (capture/scripted/shape3d_probe.json).
    Not an artifact: rotating the cloud collapses accuracy (y-up is right) and plain
    one-noun prompts score the same. Blocky partial builds are simply far from what the
    text alignment saw in training. So s_shape is NOT a B2 field and reaches no head.
  * embed(), the raw 1024-dim embedding, PASSED — a linear readout trained on Source-A
    labels beats the symbolic floor at every progress bin through 70% and only trails
    once templates lock on late (capture/scripted/h3d_linear_probe.json). The shape
    signal is in the embedding; the text tower just cannot read it out of distribution.
    That licenses embed() as the h3d producer for D3's TRAINED adapter (Phase E), where
    learned readouts live by design.

The build becomes a point cloud the way Uni3D's training shapes did — points on the
surface: we sample the exposed faces of player-built cells (a face is exposed when no
built cell covers it), color them from a block palette, and normalize to the unit ball.
Only PLAYER-built cells enter the cloud (the D2 ground rules: shape identity is what
the builder made, not the hill they made it on).
"""
from __future__ import annotations

import os
import random

from ..contracts.b1 import GOALS
from ..contracts.goals import TAXONOMY

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CKPT = os.path.join(_ROOT, "models", "uni3d-b.pt")
_TEXT_CACHE = os.path.join(_ROOT, "models", "uni3d_goal_prompts.pt")
_REPO = os.path.join(_ROOT, "vendor", "uni3d")

_POINTS = 4096          # cloud size fed to the encoder (Uni3D evals use 2k-10k)
_FACE_SAMPLES = 8       # sampled points per exposed block face

# Representative colors per block family, normalized [0,1] — Uni3D trains on colored
# clouds; unknown blocks fall back to neutral gray rather than failing.
_BLOCK_COLORS = {
    "oak": (0.65, 0.52, 0.31), "spruce": (0.45, 0.33, 0.19), "birch": (0.77, 0.71, 0.48),
    "stone": (0.49, 0.49, 0.49), "cobblestone": (0.42, 0.42, 0.42),
    "brick": (0.56, 0.35, 0.30), "sand": (0.85, 0.80, 0.60), "dirt": (0.52, 0.38, 0.26),
    "grass": (0.36, 0.56, 0.25), "poppy": (0.80, 0.15, 0.15), "fence": (0.65, 0.52, 0.31),
    "glass": (0.85, 0.92, 0.95), "log": (0.42, 0.33, 0.20), "planks": (0.65, 0.52, 0.31),
    "slab": (0.65, 0.52, 0.31), "wool": (0.90, 0.90, 0.90),
}
_GRAY = (0.6, 0.6, 0.6)

_FACES = (((1, 0, 0), (0.5, 0.0, 0.0)), ((-1, 0, 0), (-0.5, 0.0, 0.0)),
          ((0, 1, 0), (0.0, 0.5, 0.0)), ((0, -1, 0), (0.0, -0.5, 0.0)),
          ((0, 0, 1), (0.0, 0.0, 0.5)), ((0, 0, -1), (0.0, 0.0, -0.5)))


def _block_color(block: str) -> tuple[float, float, float]:
    name = block.split(":")[-1]
    for family, color in _BLOCK_COLORS.items():
        if family in name:
            return color
    return _GRAY


def build_point_cloud(built: dict, rng: random.Random) -> list[tuple[float, ...]]:
    """Player-built cells -> a surface point cloud (x, y, z, r, g, b), unit-ball
    normalized. Deterministic given the rng — probes must be reproducible."""
    if not built:
        return []
    points = []
    for (x, y, z), block in built.items():
        color = _block_color(block)
        for (nx, ny, nz), (ox, oy, oz) in _FACES:
            if (x + nx, y + ny, z + nz) in built:
                continue   # covered face — no surface there
            for _ in range(_FACE_SAMPLES):
                u, v = rng.random() - 0.5, rng.random() - 0.5
                if nx != 0:
                    point = (x + ox + 0.0, y + u, z + v)
                elif ny != 0:
                    point = (x + u, y + oy + 0.0, z + v)
                else:
                    point = (x + u, y + v, z + oz + 0.0)
                points.append((*point, *color))
    # resample to a fixed count, then normalize into the unit ball (Uni3D's preprocessing)
    while len(points) < _POINTS:
        points.append(points[rng.randrange(len(points))])
    if len(points) > _POINTS:
        points = rng.sample(points, _POINTS)
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    cz = sum(p[2] for p in points) / len(points)
    radius = max(((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2) ** 0.5
                 for p in points) or 1.0
    return [((p[0] - cx) / radius, (p[1] - cy) / radius, (p[2] - cz) / radius,
             p[3], p[4], p[5]) for p in points]


def assets_ready() -> tuple[bool, str]:
    """Whether the encoder checkpoint, vendored repo, and cached prompt embeddings
    exist — the probe reports what is missing instead of crashing."""
    missing = [path for path in (_CKPT, _TEXT_CACHE, _REPO) if not os.path.exists(path)]
    return (not missing, "; ".join(os.path.relpath(m, _ROOT) for m in missing))


class Uni3DShapeHead:
    """Loads frozen Uni3D once plus the pre-cached goal-prompt text embeddings, and
    scores a build's point cloud against each category (subtype prompts pooled).

    The text embeddings are cached to disk by scripts/setup_uni3d.py using the PAIRED
    CLIP teacher — pairing matters: a cosine against the wrong text tower is noise.
    """

    def __init__(self, device: str = "cuda"):
        import sys

        import torch as th

        if _REPO not in sys.path:
            sys.path.insert(0, _REPO)
        from mica_uni3d_loader import load_uni3d   # thin loader vendored next to the repo

        self.torch = th
        self.device = device
        self.model = load_uni3d(_CKPT, device)
        cache = th.load(_TEXT_CACHE, map_location=device)
        self.goal_feats = cache["goal_feats"].to(device)      # [|G|, D], normalized,
        self.goals = tuple(cache["goals"])                    # subtype prompts pooled
        assert self.goals == GOALS, "prompt cache was built for a different taxonomy"

    def embed(self, built: dict, seed: int = 0):
        """The build's normalized 1024-dim Uni3D embedding — the h3d channel for the
        trained adapter. None when nothing is built. This is the reading of the encoder
        that survived the probe; the zero-shot cosines in score() did not."""
        cloud = build_point_cloud(built, random.Random(seed))
        if not cloud:
            return None
        th = self.torch
        with th.no_grad():
            points = th.tensor(cloud, dtype=th.float32, device=self.device)[None]
            feats = self.model.encode_pc(points)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).float()

    def h3d(self, built: dict, seed: int = 0) -> tuple[float, ...] | None:
        """embed() as the plain float tuple the B2 contract carries (None when nothing
        is built) — the exact shape of Evidence3D.h3d, so pipeline code that must stay
        torch-free can take this method as its `h3d_fn` without seeing a tensor.
        Seed 0 is the pipeline's fixed cloud seed (recorded in d2 provenance)."""
        feats = self.embed(built, seed)
        if feats is None:
            return None
        return tuple(feats.cpu().tolist())

    def score(self, built: dict, seed: int = 0) -> tuple[float, ...] | None:
        """Per-category cosine in CLIP space for the current build; None when empty.
        Kept as the probe's measurement head — it failed the fusion gate (see module
        docstring) and is not consumed by the pipeline."""
        feats = self.embed(built, seed)
        if feats is None:
            return None
        sims = feats.to(self.goal_feats.device) @ self.goal_feats.T
        return tuple(sims.cpu().tolist())


def prompt_texts() -> dict[str, list[str]]:
    """The taxonomy prompts the text cache is built from — one list per category,
    subtype phrasings pooled, mirroring the MineCLIP head's ensembling."""
    templates = ("a {s}", "a minecraft {s}", "a voxel model of a {s}", "a blocky {s}")
    return {goal: [t.format(s=subtype) for subtype in TAXONOMY[goal] for t in templates]
            for goal in GOALS}
