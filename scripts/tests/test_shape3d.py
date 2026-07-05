"""The build->point-cloud converter: deterministic, surface-only, normalized — the part
of the Uni3D adaptation that runs without any model on disk."""
import random

from mica.perception.shape3d import _POINTS, build_point_cloud, prompt_texts
from mica.contracts.b1 import GOALS


def _pen_cells():
    fence = "minecraft:oak_fence"
    ring = [(x, 64, 0) for x in range(5)] + [(x, 64, 4) for x in range(5)] \
         + [(0, 64, z) for z in range(1, 4)] + [(4, 64, z) for z in range(1, 4)]
    return {cell: fence for cell in ring}


def test_cloud_is_fixed_size_normalized_and_deterministic():
    cloud_a = build_point_cloud(_pen_cells(), random.Random(3))
    cloud_b = build_point_cloud(_pen_cells(), random.Random(3))
    assert cloud_a == cloud_b                        # probes must be reproducible
    assert len(cloud_a) == _POINTS
    assert max((p[0] ** 2 + p[1] ** 2 + p[2] ** 2) ** 0.5 for p in cloud_a) <= 1.0 + 1e-9
    assert all(0.0 <= c <= 1.0 for p in cloud_a for c in p[3:])   # colors in range


def test_covered_faces_produce_no_points():
    # two glued cubes: the touching faces are interior — a lone cube samples 6 faces,
    # the pair samples 10, never 12
    lone = {(0, 0, 0): "minecraft:stone"}
    pair = {(0, 0, 0): "minecraft:stone", (1, 0, 0): "minecraft:stone"}
    raw_faces = lambda cells: sum(
        1 for (x, y, z) in cells for dx, dy, dz in
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        if (x + dx, y + dy, z + dz) not in cells)
    assert raw_faces(lone) == 6
    assert raw_faces(pair) == 10


def test_empty_build_yields_no_cloud():
    assert build_point_cloud({}, random.Random(0)) == []


def test_prompts_cover_every_category_via_its_subtypes():
    prompts = prompt_texts()
    assert set(prompts) == set(GOALS)
    assert all(len(texts) >= 8 for texts in prompts.values())
    assert any("treehouse" in t for t in prompts["habitation"])


def test_fps_shim_matches_brute_force_reference():
    # the loader shim replaces Uni3D's compiled farthest-point-sampling op with pure
    # torch; both are greedy and deterministic, so they must agree index for index
    import sys
    import torch

    sys.path.insert(0, r"D:\2026projects\MICA\vendor\uni3d")
    from mica_uni3d_loader import _gather, _pure_torch_fps

    torch.manual_seed(11)
    xyz = torch.rand(2, 200, 3)
    count = 16
    got = _pure_torch_fps(xyz, count)

    for b in range(xyz.size(0)):
        chosen = [0]
        dist = ((xyz[b] - xyz[b][0]) ** 2).sum(-1)
        for _ in range(count - 1):
            nxt = int(dist.argmax())
            chosen.append(nxt)
            dist = torch.minimum(dist, ((xyz[b] - xyz[b][nxt]) ** 2).sum(-1))
        assert got[b].tolist() == chosen

    # gather_operation semantics: [B, C, N] indexed by [B, M] -> [B, C, M]
    points = xyz.transpose(1, 2).contiguous()
    picked = _gather(points, got)
    assert picked.shape == (2, 3, count)
    assert torch.equal(picked[0, :, 0], points[0, :, got[0, 0].long()])
