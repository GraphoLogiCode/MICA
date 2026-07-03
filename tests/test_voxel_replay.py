"""The replay engine: built-cell semantics, crop escapes, snapshot decode, and the
divergence taxonomy — the pieces Proposition 1's per-session check stands on."""
import json

from mica.contracts.b0 import BlockEvent, BlockOp, BlockPos
from mica.perception.voxel_replay import (
    Region, ReplayWorld, classify_divergence, compare_to_snapshot, load_snapshot,
)

_REGION = Region(-4, 0, -4, 4, 4, 4)


def _place(eid, x, y, z, block="minecraft:oak_planks"):
    return BlockEvent(event_id=eid, pos=BlockPos(x, y, z), block_type=block,
                      op=BlockOp.PLACE, actor="tester")


def _break(eid, x, y, z, block="minecraft:stone"):
    return BlockEvent(event_id=eid, pos=BlockPos(x, y, z), block_type=block,
                      op=BlockOp.BREAK, actor="tester")


def test_built_means_player_placed_and_surviving():
    # natural stone sits in the base; breaking it must not create a "built" cell,
    # and placing-then-breaking must leave nothing behind
    world = ReplayWorld(_REGION, {(0, 0, 0): "minecraft:stone"})
    world.apply(_break(0, 0, 0, 0))
    world.apply(_place(1, 1, 0, 0))
    world.apply(_place(2, 2, 0, 0))
    world.apply(_break(3, 2, 0, 0))
    assert world.built() == {(1, 0, 0): "minecraft:oak_planks"}
    assert world.block_at((0, 0, 0)) == "minecraft:air"


def test_event_outside_region_is_an_escape_not_a_change():
    world = ReplayWorld(_REGION, {})
    world.apply(_place(0, 99, 0, 0))
    assert len(world.escaped) == 1
    assert world.built() == {}


def test_snapshot_roundtrip(tmp_path):
    # a 2x1x2 region: [stone, air, air, planks] in y->z->x scan order
    snap = {"tick": 7, "region": [0, 0, 0, 1, 0, 1],
            "palette": ["minecraft:stone", "minecraft:air", "minecraft:oak_planks"],
            "runs": [[1, 0], [2, 1], [1, 2]]}
    path = tmp_path / "7.json"
    path.write_text(json.dumps(snap), encoding="utf-8")
    tick, region, cells = load_snapshot(str(path))
    assert tick == 7 and region == Region(0, 0, 0, 1, 0, 1)
    assert cells == {(0, 0, 0): "minecraft:stone", (1, 0, 1): "minecraft:oak_planks"}


def test_divergence_classes():
    water = classify_divergence((0, 0, 0), "minecraft:air", "minecraft:water")
    door = classify_divergence((0, 1, 0), "minecraft:air", "minecraft:oak_door")
    phantom = classify_divergence((0, 0, 0), "minecraft:oak_planks", "minecraft:air")
    assert (water.taxonomy_class, door.taxonomy_class, phantom.taxonomy_class) == ("a", "b", "c")


def test_compare_reports_only_real_disagreements():
    world = ReplayWorld(_REGION, {(0, 0, 0): "minecraft:stone"})
    world.apply(_place(0, 1, 0, 0))
    snapshot = {(0, 0, 0): "minecraft:stone", (1, 0, 0): "minecraft:oak_planks",
                (2, 0, 0): "minecraft:water"}   # water the replay knows nothing about
    divergences = compare_to_snapshot(world, snapshot)
    assert len(divergences) == 1 and divergences[0].taxonomy_class == "a"
