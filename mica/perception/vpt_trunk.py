"""Frozen VPT trunk: a window of POV frames -> h2d, the behavior embedding the adapter reads.

VPT (OpenAI Video-Pre-Training) is a policy trained to predict the next human action from
pixels, so its trunk encodes "what playing looks like" — the prior the deliberative head
scores observed actions against. We load it once, frozen, and read the latent just below
the action head (1024-d). Nothing trains here; this is inference only.

The load recipe (found by the Layer-2 compatibility spike): the VPT repo cloned at
vendor/vpt, the 1x foundation checkpoint in models/, and a one-constant minerl stub — VPT's
action code imports minerl for an item map the trunk never touches.
"""
from __future__ import annotations

import os
import pickle
import sys
import types

import numpy as np
import torch as th
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_REPO = os.path.join(_ROOT, "vendor", "vpt")
_MODEL = os.path.join(_ROOT, "models", "vpt-1x.model")
_WEIGHTS = os.path.join(_ROOT, "models", "vpt-1x.weights")
_RES = 128   # VPT's native input resolution (square)


def _stub_minerl() -> None:
    # VPT's actions.py does `import minerl.herobraine.hero.mc` only for MINERL_ITEM_MAP,
    # which the trunk never reads. Stub the chain so we don't pull in the MineRL env.
    for name in ("minerl", "minerl.herobraine", "minerl.herobraine.hero", "minerl.herobraine.hero.mc"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
    sys.modules["minerl.herobraine.hero.mc"].MINERL_ITEM_MAP = {}


def _to_frame(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    # VPT trains on a plain bilinear squish to 128x128 (vendor/vpt agent.resize_image uses
    # cv2.INTER_LINEAR); we match it. The model normalizes internally.
    return np.asarray(image.convert("RGB").resize((_RES, _RES), Image.BILINEAR), dtype=np.uint8)


class VptTrunk:
    """Loads the frozen VPT policy once and turns a frame window into the 1024-d embedding."""

    def __init__(self, device: str = "cuda"):
        _stub_minerl()
        if _REPO not in sys.path:
            sys.path.insert(0, _REPO)
        from lib.action_mapping import CameraHierarchicalMapping
        from lib.policy import MinecraftAgentPolicy
        from gym3.types import DictType

        params = pickle.load(open(_MODEL, "rb"))
        net_kwargs = params["model"]["args"]["net"]["args"]
        pi_kwargs = params["model"]["args"]["pi_head_opts"]
        pi_kwargs["temperature"] = float(pi_kwargs["temperature"])
        action_space = DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update())

        policy = MinecraftAgentPolicy(policy_kwargs=net_kwargs, pi_head_kwargs=pi_kwargs, action_space=action_space)
        # strict=True: the 1x checkpoint matches the agent policy exactly (the spike showed 0 missing /
        # 0 unexpected), so a strict load surfaces any future mismatch instead of hiding it.
        policy.load_state_dict(th.load(_WEIGHTS, map_location="cpu"), strict=True)
        self.policy = policy.to(device).eval()
        self.device = device

    @th.no_grad()
    def embed(self, frames: list) -> tuple[float, ...] | None:
        """A window of frames (PIL or HxWx3 arrays) -> the behavior embedding after the window."""
        if not frames:
            return None
        imgs = np.stack([_to_frame(f) for f in frames])             # [T, 128, 128, 3] uint8; model scales /255 itself
        obs = {"img": th.from_numpy(imgs)[None].to(self.device)}    # [1, T, 128, 128, 3]
        first = th.zeros(1, len(frames), dtype=th.bool, device=self.device)
        # Fresh state + a reset flag at position 0: the embedding depends on this window's
        # frames and nothing else. The belief filter needs that purity — evidence from one
        # step must not secretly carry earlier steps inside it.
        first[0, 0] = True
        (pi_latent, _), _ = self.policy.net(obs, self.policy.initial_state(1), context={"first": first})
        return tuple(pi_latent[0, -1].float().cpu().tolist())       # latent after the last frame
