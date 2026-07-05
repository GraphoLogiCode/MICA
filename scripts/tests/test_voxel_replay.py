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
    # taxonomy v2: natural growth is class (a) — the cactus that quarantined a live
    # session (2026-07-04) must read as world dynamics, not as a phantom event.
    cactus = classify_divergence((1, 65, 1), "minecraft:air", "minecraft:cactus")
    grass = classify_divergence((2, 64, 0), "minecraft:dirt", "minecraft:grass_block")
    log = classify_divergence((3, 64, 0), "minecraft:air", "minecraft:oak_log")
    assert (cactus.taxonomy_class, grass.taxonomy_class, log.taxonomy_class) == ("a", "a", "c")
    # taxonomy v3: support-popped plants — dig the block under a grass plant and the
    # plant vanishes with no break event of its own (quarantined a real session,
    # 2026-07-05: replay grass vs world air). Logs still deliberately class (c).
    popped = classify_divergence((4, 65, 0), "minecraft:grass", "minecraft:air")
    flower = classify_divergence((5, 65, 0), "minecraft:poppy", "minecraft:air")
    assert (popped.taxonomy_class, flower.taxonomy_class) == ("a", "a")


def test_monitor_holds_live_events_newer_than_the_snapshot():
    # The false-quarantine race (2026-07-04, session 223052): a snapshot file
    # reaches the live comparison ~1 s after the world state it froze, and a busy
    # builder keeps placing in that second. Events newer than the snapshot's tick
    # must WAIT — comparing a world that is ahead of the snapshot reads the
    # builder's own new blocks as phantom class-c divergences.
    from types import SimpleNamespace

    from mica.perception.voxel_replay import SnapshotMonitor

    def packet(tick, *events):
        return SimpleNamespace(tick=tick, server=SimpleNamespace(block_events=events))

    monitor = SnapshotMonitor(_REGION, {})
    monitor.apply_packet(packet(50, _place(0, 0, 0, 0)))
    monitor.apply_packet(packet(120, _place(1, 1, 0, 0)))   # lands AFTER the snapshot
    found = monitor.on_snapshot(100, {(0, 0, 0): "minecraft:oak_planks"})
    assert found == [] and not monitor.quarantined          # the tick-120 event waited
    assert monitor.world.block_at((1, 0, 0)) == "minecraft:air"
    monitor.drain_pending()                                  # stream over: it applies now
    assert monitor.world.block_at((1, 0, 0)) == "minecraft:oak_planks"

    # escapes are judged at ADJUDICATION (region v3: the frame may grow between an
    # event's arrival and its judgment) — held until a compare or the final drain,
    # then an event still outside the frame escapes and condemns the session
    escaper = SnapshotMonitor(_REGION, {})
    escaper.apply_packet(packet(10, _place(7, 99, 0, 0)))
    assert not escaper.quarantined                    # not judged yet
    escaper.drain_pending()
    assert len(escaper.world.escaped) == 1 and escaper.quarantined


def test_compare_reports_only_real_disagreements():
    world = ReplayWorld(_REGION, {(0, 0, 0): "minecraft:stone"})
    world.apply(_place(0, 1, 0, 0))
    snapshot = {(0, 0, 0): "minecraft:stone", (1, 0, 0): "minecraft:oak_planks",
                (2, 0, 0): "minecraft:water"}   # water the replay knows nothing about
    divergences = compare_to_snapshot(world, snapshot)
    assert len(divergences) == 1 and divergences[0].taxonomy_class == "a"
