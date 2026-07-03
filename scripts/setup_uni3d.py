"""One-time Uni3D asset setup — everything the shape head needs on disk.

    python scripts/setup_uni3d.py

Three steps, each skipped if its output already exists:
  1. Download the released uni3d-b checkpoint (HF BAAI/Uni3D, modelzoo/uni3d-b/model.pt)
     to models/uni3d-b.pt.
  2. Embed the goal taxonomy's subtype prompts with Uni3D's PAIRED text tower
     (open_clip EVA02-E-14-plus, laion2b_s9b_b144k — the teacher every Uni3D size was
     aligned to) and cache the pooled per-category features to
     models/uni3d_goal_prompts.pt. Pairing matters: cosines against any other text
     tower would be noise. This is the big download (~10 GB, one time); only the tiny
     embedding cache is kept in the loop afterwards.
  3. Smoke-test the full load path: build the model through vendor/uni3d's loader shim,
     encode a dummy cloud, check the embedding is finite and text-cache-compatible.

After this, scripts/probe_shape3d.py runs the earn-its-place probe.
"""
from __future__ import annotations

import gc
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from mica.perception.shape3d import _CKPT, _TEXT_CACHE, prompt_texts   # noqa: E402
from mica.contracts.b1 import GOALS                                    # noqa: E402

_HF_REPO = "BAAI/Uni3D"
_HF_FILE = "modelzoo/uni3d-b/model.pt"
_TEACHER = "EVA02-E-14-plus"
_TEACHER_TAG = "laion2b_s9b_b144k"


def fetch_checkpoint() -> None:
    if os.path.exists(_CKPT):
        print(f"checkpoint already at {os.path.relpath(_CKPT, _ROOT)}")
        return
    from huggingface_hub import hf_hub_download
    print(f"downloading {_HF_REPO}/{_HF_FILE} ...")
    got = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)
    os.makedirs(os.path.dirname(_CKPT), exist_ok=True)
    import shutil
    shutil.copyfile(got, _CKPT)
    print(f"  -> {os.path.relpath(_CKPT, _ROOT)}")


def cache_text_embeddings() -> None:
    if os.path.exists(_TEXT_CACHE):
        print(f"prompt cache already at {os.path.relpath(_TEXT_CACHE, _ROOT)}")
        return
    import torch
    import open_clip

    print(f"loading paired text tower {_TEACHER} ({_TEACHER_TAG}) — large, one time ...")
    model, _, _ = open_clip.create_model_and_transforms(_TEACHER, pretrained=_TEACHER_TAG)
    tokenizer = open_clip.get_tokenizer(_TEACHER)
    model.visual = None          # only the text tower is needed; free the image half
    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = model.to(device)
    except torch.cuda.OutOfMemoryError:
        device = "cpu"
    model = model.to(device).eval()

    prompts = prompt_texts()
    feats = []
    with torch.no_grad():
        for goal in GOALS:
            tokens = tokenizer(prompts[goal]).to(device)
            embedded = model.encode_text(tokens)
            embedded = embedded / embedded.norm(dim=-1, keepdim=True)
            pooled = embedded.mean(dim=0)                 # pool subtype phrasings
            feats.append(pooled / pooled.norm())
    os.makedirs(os.path.dirname(_TEXT_CACHE), exist_ok=True)
    torch.save({"goals": tuple(GOALS), "goal_feats": torch.stack(feats).float().cpu(),
                "teacher": f"{_TEACHER}/{_TEACHER_TAG}"}, _TEXT_CACHE)
    print(f"  -> {os.path.relpath(_TEXT_CACHE, _ROOT)} "
          f"({len(GOALS)} categories, dim {feats[0].shape[0]})")
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def smoke_test() -> None:
    import random
    import torch
    from mica.perception.shape3d import Uni3DShapeHead

    head = Uni3DShapeHead("cuda" if torch.cuda.is_available() else "cpu")
    build = {(x, 64, z): "minecraft:oak_planks" for x in range(4) for z in range(4)}
    sims = head.score(build, seed=1)
    assert sims is not None and len(sims) == len(GOALS)
    assert all(-1.0 <= s <= 1.0 for s in sims), sims
    print("smoke test: encode + score OK ->",
          {g: round(s, 3) for g, s in zip(GOALS, sims)})


if __name__ == "__main__":
    fetch_checkpoint()
    cache_text_embeddings()
    smoke_test()
    print("uni3d assets ready — run scripts/probe_shape3d.py")
