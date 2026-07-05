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

TAXONOMY_VERSION = "3"
AIR = "minecraft:air"

# Class (a): blocks the world moves on its own right next to where the player worked.
# Version 2 (2026-07-04): NATURAL GROWTH joined the set after a live session was
# quarantined by a single cactus growing beside the build (replay air vs world
# cactus — class c under v1). Growth-stage plants, spreading ground covers, decaying
# leaves, and snowfall/ice are the world acting alone, exactly like flowing water.
# Version 3 (2026-07-05): SUPPORT-POPPED PLANTS joined after the next real session
# quarantined on `grass` (the short grass plant) reading air — dig the block UNDER
# a plant and the plant pops off as a consequence, with no break event of its own
# (the same family as the door-top gap). Short plants, flowers, saplings, and water
# plants are in; they pop, they grow back, and players almost never place them.
# Deliberately NOT included: logs (tree growth from a sapling still reads class c —
# a log's spontaneous appearance is indistinguishable from a missed placement, and
# players place logs constantly; conservatism wins there).
_DYNAMIC = frozenset((
    "minecraft:water", "minecraft:lava", "minecraft:sand", "minecraft:gravel",
    "minecraft:red_sand", "minecraft:fire",
    # growth-stage plants
    "minecraft:cactus", "minecraft:sugar_cane", "minecraft:bamboo", "minecraft:bamboo_sapling",
    "minecraft:wheat", "minecraft:carrots", "minecraft:potatoes", "minecraft:beetroots",
    "minecraft:melon_stem", "minecraft:pumpkin_stem",
    "minecraft:attached_melon_stem", "minecraft:attached_pumpkin_stem",
    "minecraft:melon", "minecraft:pumpkin", "minecraft:sweet_berry_bush",
    "minecraft:vine", "minecraft:kelp", "minecraft:kelp_plant",
    "minecraft:brown_mushroom", "minecraft:red_mushroom",
    # support-popped / regrowing short plants (v3)
    "minecraft:grass", "minecraft:fern", "minecraft:large_fern", "minecraft:dead_bush",
    "minecraft:dandelion", "minecraft:poppy", "minecraft:blue_orchid", "minecraft:allium",
    "minecraft:azure_bluet", "minecraft:red_tulip", "minecraft:orange_tulip",
    "minecraft:white_tulip", "minecraft:pink_tulip", "minecraft:oxeye_daisy",
    "minecraft:cornflower", "minecraft:lily_of_the_valley",
    "minecraft:oak_sapling", "minecraft:spruce_sapling", "minecraft:birch_sapling",
    "minecraft:jungle_sapling", "minecraft:acacia_sapling", "minecraft:dark_oak_sapling",
    "minecraft:lily_pad", "minecraft:seagrass", "minecraft:tall_seagrass",
    # spreading / reverting ground covers
    "minecraft:grass_block", "minecraft:dirt", "minecraft:mycelium",
    # leaf decay and weather
    "minecraft:oak_leaves", "minecraft:spruce_leaves", "minecraft:birch_leaves",
    "minecraft:jungle_leaves", "minecraft:acacia_leaves", "minecraft:dark_oak_leaves",
    "minecraft:snow", "minecraft:ice",
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

    def covers(self, other: "Region") -> bool:
        """Does this box contain the whole of `other`? (region v3: frames may only grow,
        and 'covers' is how every consumer tells growth from an illegal move)."""
        return (self.x0 <= other.x0 and self.y0 <= other.y0 and self.z0 <= other.z0
                and self.x1 >= other.x1 and self.y1 >= other.y1 and self.z1 >= other.z1)


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
    D2 ground rules' built-cell set falls out for free. The world itself is actor-blind
    on purpose: WHICH events reach it decides whose blocks it holds. The feature world
    receives human events only (Evidence3DStream.remember filters by A7); the snapshot
    monitor's shadow world receives every event, because the real world it is compared
    against contains the agent's blocks too.
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

    def grow(self, new_region: Region) -> None:
        """Adopt a LARGER frame (region v3). Growth is the only legal change — a frame
        that moved or shrank would silently re-frame existing evidence — so anything
        that neither covers nor is covered refuses loudly. A frame we already cover
        is old news (the manifest can lag behind a newer growth snapshot): no-op."""
        if self.region.covers(new_region):
            return
        assert new_region.covers(self.region), \
            f"region may only GROW: {self.region} -> {new_region}"
        self.region = new_region

    def extend_base(self, cells: dict[tuple[int, int, int], str]) -> None:
        """Base state for freshly adopted cells (a growth snapshot's view of the new
        territory). setdefault-only, and a cell the player already CHANGED is skipped
        outright — the growth snapshot was taken after their block landed there, and
        adopting it as 'base' would swallow a real placement out of built()."""
        for cell, block in cells.items():
            if cell not in self.changed:
                self.base.setdefault(cell, block)


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


class SnapshotMonitor:
    """The replay-vs-snapshot check as a machine: keep a shadow world advanced by every
    event as it arrives, compare it whenever the capture produces a snapshot.

    Offline, run_d2 drives it across a finished recording; live, the runner feeds it
    per packet and hands it each new snapshot file the mod writes mid-session. Either
    way it accumulates the same divergence list and the same verdict: a class-c
    divergence or an event escaping the region means the replayed world can't be
    trusted and the session is quarantined.
    """

    # Keep only the first examples in memory — a session whose region went bad can
    # diverge on every cell (a real one held 823k tuples); the COUNT stays exact.
    _DIVERGENCE_EXAMPLES = 200

    def __init__(self, region: Region, base_cells: dict[tuple[int, int, int], str]):
        self.world = ReplayWorld(region, base_cells)
        self.divergences: list[tuple[int, Divergence]] = []   # first N examples only
        self.divergence_count = 0                             # exact total, always
        self.class_counts = {"a": 0, "b": 0, "c": 0}          # exact per class, always
        self._seen_ids: set[int] = set()                      # every id already applied
        self._pending: list[tuple[int, BlockEvent]] = []      # live events awaiting a compare
        self._base_region = self.world.region                 # how far the BASE knowledge reaches

    def apply_event(self, event: BlockEvent) -> None:
        # Applied at most once per id: a late attach replays the on-disk history
        # first (seed_history), and the live socket may deliver the events at the
        # attach boundary again — a repeat must not double-place a block or double-
        # count an escape.
        if event.event_id in self._seen_ids:
            return
        self._seen_ids.add(event.event_id)
        self.world.apply(event)

    def apply_packet(self, packet) -> None:
        """The live feed's entry point: every event is HELD with its tick. Two rules
        meet here. First, the compare race: a snapshot file reaches the comparison a
        second after the world state it froze, and a busy builder keeps placing in
        that second — applying newer events first once false-quarantined a whole real
        session. Second, region v3: the frame may GROW between an event's arrival and
        its judgment, so an event is an escape iff it lies outside the region in
        force when it is ADJUDICATED (at the next comparison, or the final drain) —
        not the region of the moment it happened."""
        for event in packet.server.block_events:
            self._pending.append((packet.tick, event))

    def notice_region(self, new_region: Region) -> None:
        """The manifest says the frame grew (region v3): widen the escape test NOW.
        The adopted cells' base arrives with the growth snapshot file — until then
        the wider frame only changes where the boundary of 'escape' sits."""
        self.world.grow(new_region)

    def drain_pending(self) -> None:
        """Apply every held event (stream over — no more snapshots are coming)."""
        for _, event in self._pending:
            self.apply_event(event)
        self._pending.clear()

    def seed_history(self, events, snapshots) -> None:
        """Late-attach catch-up for the shadow world: replay EVERY event recorded so
        far (agent blocks included — the real world holds them too), pausing at each
        snapshot already on disk to run the replay-vs-snapshot comparison. This is
        the same walk the offline verifier does over a finished recording, so a
        session that had already gone untrustworthy is caught AT the attach instead
        of surfacing later. `events` is (tick, event) pairs in recorded order;
        `snapshots` is (tick, cells, region) triples in tick order — the frame widens
        BEFORE each snapshot's events apply, so events in freshly grown territory are
        judged against the frame that adopted them (region v3)."""
        ordered = list(events)
        index = 0
        for snap_tick, cells, region in snapshots:
            if region is not None:
                self.notice_region(region)
            while index < len(ordered) and ordered[index][0] <= snap_tick:
                self.apply_event(ordered[index][1])
                index += 1
            self.on_snapshot(snap_tick, cells, region)
        while index < len(ordered):
            self.apply_event(ordered[index][1])
            index += 1

    def on_snapshot(self, snap_tick: int, cells: dict,
                    region: Region | None = None) -> list[Divergence]:
        """Compare the shadow world against one snapshot; remember what differed.

        Order is load-bearing (region v3): the frame widens FIRST (held events in
        freshly grown territory must be judged in-frame), then held events up to the
        snapshot's tick apply (newer ones keep waiting — the world must be AS OF the
        snapshot, not as of right now), and only THEN does the snapshot's view of the
        cells beyond previous base coverage become the adopted base — applying events
        first lets the changed-cell guard keep a margin-race placement out of it."""
        if region is not None:
            self.world.grow(region)
        still_pending = []
        for tick, event in self._pending:
            if tick <= snap_tick:
                self.apply_event(event)
            else:
                still_pending.append((tick, event))
        self._pending = still_pending
        if region is not None and not self._base_region.covers(region):
            self.world.extend_base({cell: block for cell, block in cells.items()
                                    if not self._base_region.contains(*cell)})
            self._base_region = region
        found = compare_to_snapshot(self.world, cells)
        self.divergence_count += len(found)
        for divergence in found:
            self.class_counts[divergence.taxonomy_class] += 1
        room = self._DIVERGENCE_EXAMPLES - len(self.divergences)
        if room > 0:
            self.divergences.extend((snap_tick, divergence) for divergence in found[:room])
        return found

    @property
    def quarantined(self) -> bool:
        # Exact counters, not the capped example list — a class-c divergence past
        # the example cap must still condemn the session.
        return bool(self.world.escaped) or self.class_counts["c"] > 0
