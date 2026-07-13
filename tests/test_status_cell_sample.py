"""The live-status cell feed: small builds travel whole; big builds travel as a
fresh random sample per write, so the agent's patrol is never permanently blind to
cells past the cap (the old sorted-prefix bug — patrol v2, 2026-07-11)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
from run_live import _status_cell_sample  # noqa: E402


def test_small_builds_travel_whole_and_sorted():
    built = {(3, 64, 1): "minecraft:stone", (0, 64, 0): "minecraft:stone"}
    assert _status_cell_sample(built, cap=512) == [[0, 64, 0], [3, 64, 1]]


def test_big_builds_sample_fresh_and_cover_everything():
    built = {(x, 64, z): "minecraft:stone" for x in range(40) for z in range(20)}  # 800 cells
    seen = set()
    for _ in range(30):
        sample = _status_cell_sample(built, cap=512)
        assert len(sample) == 512                     # the cap holds every write
        seen.update(tuple(cell) for cell in sample)
    # A sorted prefix would show the same 512 low cells forever; random sampling
    # must reach every cell across draws (miss chance ~(0.36)^30 per cell).
    assert seen == set(built)
