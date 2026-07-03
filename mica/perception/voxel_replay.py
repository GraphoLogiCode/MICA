"""Rebuild the world's build region by replaying block events onto a starting snapshot.

This is D2's Proposition 1 made runnable. The replayed state is what every 3D feature is
computed from, and the replay-vs-snapshot comparison is what earns the right to trust it:
the capture's later snapshots are the world's own answer sheet. Divergences are sorted by
the review's taxonomy — world dynamics (water flowing back) and multi-block items (door
tops) are documented and tolerated; anything else means a missed or phantom event and the
session cannot be trusted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..contracts.b0 import BlockEvent, BlockOp
from ..contracts.manifest import CapturedSession

TAXONOMY_VERSION = "1"
AIR = "minecraft:air"

# Class (a): blocks the world moves on its own right next to where the player worked.
_DYNAMIC = frozenset((
    "minecraft:water", "minecraft:lava", "minecraft:sand", "minecraft:gravel",
    "minecraft:red_sand", "minecraft:fire",
))
# Class (b): the un-recorded second halves of items that place two blocks from one click.
_SECOND_HALVES = frozenset((
    "minecraft:oak_door", "minecraft:spruce_door", "minecraft:birch_door",
    "minecraft:jungle_door", "minecraft:acacia_door", "minecraft:dark_oak_door",
    "minecraft:iron_door", "minecraft:white_bed", "minecraft:red_bed",
    "minecraft:tall_grass", "minecraft:sunflower", "minecraft:lilac",
))


@dataclass(frozen=True)
class Region:
    """The fixed box every snapshot and every spatial feature is framed in."""

    x0: int
    y0: int
    z0: int
    x1: int
    y1: int
    z1: int

    def contains(self, x: int, y: int, z: int) -> bool:
        return (self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1
                and self.z0 <= z <= self.z1)


def region_from_manifest(session: CapturedSession) -> Region | None:
    r = session.manifest.snapshot_region
    return Region(*r) if len(r) == 6 else None


def region_around_events(session: CapturedSession, margin: int = 8) -> Region:
    """A synthetic session has no capture region, so frame the events themselves."""
    positions = [e.pos for p in session.packets for e in p.server.block_events]
    if not positions:
        return Region(-margin, 0, -margin, margin, margin, margin)
    xs, ys, zs = [p.x for p in positions], [p.y for p in positions], [p.z for p in positions]
    return Region(min(xs) - margin, max(0, min(ys) - margin), min(zs) - margin,
                  max(xs) + margin, min(255, max(ys) + margin), max(zs) + margin)


class ReplayWorld:
    """The region's state, event-sourced: a base snapshot plus every change on top of it.

    `base` holds the starting state (empty for synthetic void worlds). Only events mutate
    it afterward, so "player-built" is exactly "differs from base and is not air" — the
    D2 ground rules' built-cell set falls out for free, actor-filterable when the agent
    later shares the world.
    """

    def __init__(self, region: Region, base: dict[tuple[int, int, int], str]):
        self.region = region
        self.base = base
        self.changed: dict[tuple[int, int, int], str] = {}
        self.escaped: list[BlockEvent] = []   # events outside the frame — a quarantine sign

    def block_at(self, cell: tuple[int, int, int]) -> str:
        current = self.changed.get(cell)
        if current is not None:
            return current
        return self.base.get(cell, AIR)

    def apply(self, event: BlockEvent) -> None:
        cell = (event.pos.x, event.pos.y, event.pos.z)
        if not self.region.contains(*cell):
            self.escaped.append(event)
            return
        self.changed[cell] = event.block_type if event.op is BlockOp.PLACE else AIR

    def built(self) -> dict[tuple[int, int, int], str]:
        """Player-placed-and-surviving cells: changed from the base, and not to air."""
        return {cell: block for cell, block in self.changed.items()
                if block != AIR and self.base.get(cell, AIR) != block}


def load_snapshot(path: str) -> tuple[int, Region, dict[tuple[int, int, int], str]]:
    """Decode one palette+run-length snapshot file into a sparse non-air cell map.

    The mod writes runs in scan order y -> z -> x; walking any other order scrambles
    the world. Air is left out of the map so a mostly-empty box stays small in memory.
    """
    with open(path, encoding="utf-8") as handle:
        snap = json.load(handle)
    region = Region(*snap["region"])
    palette = snap["palette"]
    cells: dict[tuple[int, int, int], str] = {}
    x_count = region.x1 - region.x0 + 1
    z_count = region.z1 - region.z0 + 1
    flat = 0
    for count, index in snap["runs"]:
        block = palette[index]
        if block != AIR:
            for offset in range(count):
                position = flat + offset
                y, rest = divmod(position, z_count * x_count)
                z, x = divmod(rest, x_count)
                cells[(region.x0 + x, region.y0 + y, region.z0 + z)] = block
        flat += count
    expected = (region.x1 - region.x0 + 1) * (region.y1 - region.y0 + 1) * z_count
    if flat != expected:
        raise ValueError(f"snapshot {path}: runs decode to {flat} cells, region holds {expected}")
    return snap["tick"], region, cells


@dataclass(frozen=True)
class Divergence:
    cell: tuple[int, int, int]
    replay_block: str
    world_block: str
    taxonomy_class: str   # "a" world dynamics | "b" multi-block item | "c" unexplained


def classify_divergence(cell, replay_block: str, world_block: str) -> Divergence:
    if world_block in _DYNAMIC or replay_block in _DYNAMIC:
        taxonomy_class = "a"
    elif world_block in _SECOND_HALVES or replay_block in _SECOND_HALVES:
        taxonomy_class = "b"
    else:
        taxonomy_class = "c"
    return Divergence(cell=cell, replay_block=replay_block, world_block=world_block,
                      taxonomy_class=taxonomy_class)


def compare_to_snapshot(world: ReplayWorld, snapshot_cells: dict) -> list[Divergence]:
    """Every cell where the replayed state disagrees with the world's own snapshot."""
    divergences = []
    for cell in set(snapshot_cells) | set(world.changed) | set(world.base):
        if not world.region.contains(*cell):
            continue
        replayed = world.block_at(cell)
        actual = snapshot_cells.get(cell, AIR)
        if replayed != actual:
            divergences.append(classify_divergence(cell, replayed, actual))
    return divergences
