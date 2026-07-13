"""D2 — the 3D structure stream: replayed world state in, B2 (Evidence3D) records out.

One B2 record per correction step, paired one-to-one with B1's scored records by the
event ids they share. Features are computed on the state BEFORE the correction's first
block change applies (the snapshot rule): for each goal, how far the world matches its
template at the best offset-and-rotation; plus goal-free shape facts about what the
player has built. Nothing here trains, nothing sees the belief, and features only
recompute when an event actually changed the region — between changes they are
constant by construction, not approximation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable

from ..contracts.b0 import BlockEvent, ObservationPacket, is_agent_actor
from ..contracts.b1 import GOALS, Evidence2D
from ..contracts.b2 import ROTATIONS, Evidence3D, GlobalStructure, PerGoalStructure, Pose
from .templates import REQ_SOLID, TEMPLATES, Template, footprint, rotate_offset
from .voxel_replay import ReplayWorld

DELTA_COMP_WINDOW = 5   # corrections between the two comp readings (versioned, in provenance)
_RECENT_EDITS = 10      # how many recent changes the locality feature looks at
_MIN_RUN = 3            # a straight line this long counts as a deliberate row


def _satisfied(world: ReplayWorld, cell: tuple[int, int, int], required: str) -> bool:
    # Completion reads the world as it is: terrain may satisfy a cell (ground rules).
    block = world.block_at(cell)
    if required == REQ_SOLID:
        return block != "minecraft:air"
    return block == required


_ROTATED_FP: dict[tuple[str, int], frozenset[tuple[int, int]]] = {}


def _rotated_footprint(template: Template, rot: int) -> frozenset[tuple[int, int]]:
    key = (template.name, rot)
    if key not in _ROTATED_FP:
        _ROTATED_FP[key] = frozenset(
            (dx, dz) for dx, _, dz in (rotate_offset(offset, rot) for offset in template.cells))
    return _ROTATED_FP[key]


def _fit_pose(built_fp: set[tuple[int, int]], template: Template) -> tuple[float, Pose]:
    """Best footprint IoU of the template against the player-built cells, over
    translations x the four yaw rotations. Nothing built means nothing to register.

    One counting pass per rotation: every (built cell, template cell) pair votes for
    the offset that would align them, so the overlap at EVERY candidate offset falls
    out of |built| x |footprint| additions — same answer as testing each offset, far
    fewer steps, and only offsets that can score above zero are ever considered.
    """
    if not built_fp:
        return 0.0, Pose(dx=0, dz=0, rot=0)
    best_iou, best_pose = 0.0, Pose(dx=0, dz=0, rot=0)
    built_count = len(built_fp)
    for rot in ROTATIONS:
        footprint_cells = _rotated_footprint(template, rot)
        overlap_at: dict[tuple[int, int], int] = {}
        for bx, bz in built_fp:
            for fx, fz in footprint_cells:
                offset = (bx - fx, bz - fz)
                overlap_at[offset] = overlap_at.get(offset, 0) + 1
        fp_count = len(footprint_cells)
        for (ox, oz), overlap in overlap_at.items():
            iou = overlap / (fp_count + built_count - overlap)
            if iou > best_iou:
                best_iou, best_pose = iou, Pose(dx=ox, dz=oz, rot=rot)
    return best_iou, best_pose


@dataclass(frozen=True)
class _GoalReading:
    comp: float
    edit_distance: int
    fit: float
    pose: Pose


def _read_goal(world: ReplayWorld, template: Template, built) -> _GoalReading:
    """comp, edit distance, and fit for one template, all at the one winning pose.

    Nothing built means nothing to register against — and evaluating the template at
    an arbitrary origin would let underground stone satisfy its #solid cells. So an
    empty build reads as zero progress toward everything, by definition.
    """
    if not built:
        return _GoalReading(comp=0.0, edit_distance=len(template.cells), fit=0.0,
                            pose=Pose(dx=0, dz=0, rot=0))
    built_fp = {(x, z) for x, _, z in built}
    fit, pose = _fit_pose(built_fp, template)
    ground = min((y for _, y, _ in built), default=0)   # templates sit on the built layer
    satisfied = 0
    template_cells = set()
    for offset, required in template.cells.items():
        dx, dy, dz = rotate_offset(offset, pose.rot)
        cell = (dx + pose.dx, dy + ground, dz + pose.dz)
        template_cells.add(cell)
        if _satisfied(world, cell, required):
            satisfied += 1
    comp = satisfied / len(template.cells)
    template_fp = {(x, z) for x, _, z in template_cells}
    surplus = sum(1 for cell in built
                  if (cell[0], cell[2]) in template_fp and cell not in template_cells)
    edit_distance = (len(template.cells) - satisfied) + surplus
    return _GoalReading(comp=comp, edit_distance=edit_distance, fit=fit, pose=pose)


def _best_instance(world: ReplayWorld, goal: str, built) -> tuple[_GoalReading, str]:
    # A category is backed by its template instances; the category feature is the best
    # instance's reading, and that instance's SUBTYPE is the style read (coarse-to-fine;
    # since taxonomy v3 several instances can share one subtype — cabin and longhouse
    # both report "house").
    #
    # Ranked by fit x comp (D2 gate 2026-07-03 F1), because each factor alone fails:
    # fit is blind to missing upper layers (a bare deck would read "railed bridge" —
    # rails share the deck's footprint), and comp is terrain-inflatable (a flat template
    # soaks up natural ground and wins the style slot on real terrain — observed live).
    # The product demands both: the shape is present AND the player put it there.
    readings = [(_read_goal(world, template, built), template.subtype)
                for template in TEMPLATES[goal]]
    return max(readings, key=lambda pair: (pair[0].fit * pair[0].comp, pair[0].fit))


def read_finished_build(world: ReplayWorld) -> dict[str, tuple[_GoalReading, str]]:
    """The finished-structure matcher — Source B's recognizer (D3), which is exactly
    the stream's own per-category best-instance reading evaluated once on a COMPLETED
    world. At 100% completion the mid-build registration ambiguity is gone: the blocks
    fully constrain the fit, which is what makes the hindsight label trustworthy."""
    built = world.built()
    return {goal: _best_instance(world, goal, built) for goal in GOALS}


def _planar_runs(built) -> int:
    runs = 0
    cells = set(built)
    for x, y, z in cells:
        # Count each maximal run once, from its lowest-coordinate end.
        if (x - 1, y, z) not in cells:
            length = 1
            while (x + length, y, z) in cells:
                length += 1
            runs += length >= _MIN_RUN
        if (x, y, z - 1) not in cells:
            length = 1
            while (x, y, z + length) in cells:
                length += 1
            runs += length >= _MIN_RUN
    return runs


def _has_enclosure(built) -> bool:
    """Does the built footprint close a loop? Flood the empty cells from outside the
    bounding box; any empty cell the flood cannot reach is fenced in."""
    fp = {(x, z) for x, _, z in built}
    if len(fp) < 8:   # the smallest closable ring
        return False
    xs = [x for x, _ in fp]
    zs = [z for _, z in fp]
    x0, x1, z0, z1 = min(xs) - 1, max(xs) + 1, min(zs) - 1, max(zs) + 1
    outside = {(x0, z0)}
    frontier = [(x0, z0)]
    while frontier:
        cx, cz = frontier.pop()
        for nx, nz in ((cx + 1, cz), (cx - 1, cz), (cx, cz + 1), (cx, cz - 1)):
            if x0 <= nx <= x1 and z0 <= nz <= z1 and (nx, nz) not in fp and (nx, nz) not in outside:
                outside.add((nx, nz))
                frontier.append((nx, nz))
    interior_empties = ((x1 - x0 + 1) * (z1 - z0 + 1)) - len(fp) - len(outside)
    return interior_empties > 0


def _symmetry(built) -> tuple[float, int]:
    fp = {(x, z) for x, _, z in built}
    if not fp:
        return 0.0, 0
    # Doubled coordinates keep mirror cells on the grid when the centroid sits between cells.
    cx2 = round(sum(2 * x for x, _ in fp) / len(fp))
    cz2 = round(sum(2 * z for _, z in fp) / len(fp))
    mirrors = (
        lambda x, z: ((cx2 - 2 * x) // 2, z),                     # plane across x
        lambda x, z: (x, (cz2 - 2 * z) // 2),                     # plane across z
        lambda x, z: ((cx2 + 2 * (z - cz2 // 2)) // 2, (cz2 + 2 * (x - cx2 // 2)) // 2),
        lambda x, z: ((cx2 - 2 * (z - cz2 // 2)) // 2, (cz2 - 2 * (x - cx2 // 2)) // 2),
    )
    best = max(sum(1 for x, z in fp if mirror(x, z) in fp) / len(fp) for mirror in mirrors)
    return best, len(fp)


def _edit_locality(recent: tuple[tuple[int, int, int], ...]) -> float | None:
    if len(recent) < 2:
        return None
    cx = sum(x for x, _, _ in recent) / len(recent)
    cz = sum(z for _, _, z in recent) / len(recent)
    spread = sum(max(abs(x - cx), abs(z - cz)) for x, _, z in recent) / len(recent)
    return round(spread, 3)


def _global_feats(world: ReplayWorld, recent_edits) -> GlobalStructure:
    built = world.built()
    if not built:
        return GlobalStructure(built_count=0, bbox=None, centroid=None, planar_runs=0,
                               has_enclosure=False, symmetry=0.0, symmetry_support=0,
                               edit_locality=_edit_locality(recent_edits))
    xs = [x for x, _, _ in built]
    ys = [y for _, y, _ in built]
    zs = [z for _, _, z in built]
    symmetry, support = _symmetry(built)
    return GlobalStructure(
        built_count=len(built),
        bbox=(min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)),
        centroid=(round(sum(xs) / len(xs), 2), round(sum(ys) / len(ys), 2),
                  round(sum(zs) / len(zs), 2)),
        planar_runs=_planar_runs(built),
        has_enclosure=_has_enclosure(built),
        symmetry=round(symmetry, 4),
        symmetry_support=support,
        edit_locality=_edit_locality(recent_edits),
    )


class Evidence3DStream:
    """D2 as a correction-fed machine: block events go in as their ticks arrive, one
    Evidence3D comes out per scored B1 record. This is THE structure stage — the offline
    `build_evidence3d` below just seeds it and loops.

    The snapshot rule is baked into the call order: features are computed on the world
    as it stood BEFORE this correction's first block change, then the correction's own
    events are applied, then the feature cache is dropped so the next record recomputes.
    Between events nothing recomputes — the region only changes through events.
    """

    def __init__(self, world: ReplayWorld,
                 h3d_fn: Callable[[dict], "tuple[float, ...] | None"] | None = None):
        self._world = world
        self._base_region = world.region                # how far the base knowledge reaches
        self._events: dict[int, BlockEvent] = {}        # ids seen but not yet applied
        self._seeded: set[int] = set()                  # ids applied by seed_history (catch-up)
        self._recent_edits: deque[tuple[int, int, int]] = deque(maxlen=_RECENT_EDITS)
        # delta_comp's baseline needs the first-ever comp reading plus the last few —
        # same numbers the old whole-list history produced, in bounded memory.
        self._first_comp: dict[str, float] = {}
        self._last_comps: dict[str, deque[float]] = {
            goal: deque(maxlen=DELTA_COMP_WINDOW) for goal in GOALS}
        self._cached: dict[str, tuple[_GoalReading, str]] | None = None
        self._cached_global: GlobalStructure | None = None
        # Optional learned 3D channel: a callable taking the player-built cells and
        # returning the build's dense shape embedding (Uni3D's, in production) as a
        # plain tuple, or None when nothing is built. Kept behind a callable so this
        # module stays stdlib-only — the torch model lives in shape3d.py.
        self._h3d_fn = h3d_fn
        self._cached_h3d: tuple[float, ...] | None = None

    def harvest(self, packet: ObservationPacket) -> None:
        """Remember this tick's block events; they apply when a correction consumes them."""
        self.remember(packet.server.block_events)

    def remember(self, events: Iterable[BlockEvent]) -> None:
        """Take events into the FEATURE world's pool — humans only. The agent's own
        blocks must never shape the evidence (A7, contracts/b0.py): they are dropped
        here, at the one entry point, so no path can apply them to this world. The
        snapshot monitor keeps its own world and applies EVERY event — the real world
        contains the agent's blocks, and the replay proof must match reality."""
        for event in events:
            if event.event_id in self._seeded:
                continue                     # already in the world from catch-up
            if not is_agent_actor(event.actor):
                self._events[event.event_id] = event

    def seed_history(self, events: Iterable[BlockEvent]) -> int:
        """Late-attach catch-up: apply block events that happened BEFORE this stream
        started listening, straight into the feature world. Human events only — the
        same A7 rule as remember(), because the agent's own blocks must never shape
        the evidence. Every applied id is recorded so the live stream can never apply
        the same event a second time. Returns how many events were applied."""
        applied = 0
        for event in events:
            if is_agent_actor(event.actor):
                continue
            self._world.apply(event)
            self._recent_edits.append((event.pos.x, event.pos.y, event.pos.z))
            self._seeded.add(event.event_id)
            applied += 1
        return applied

    def notice_region(self, region) -> None:
        """Region v3: the capture frame grew. Widen the feature world's escape test
        now; the adopted cells' base arrives with the growth snapshot (extend_world)."""
        self._world.grow(region)

    def extend_world(self, region, cells: dict) -> None:
        """Adopt a growth snapshot: the wider frame plus base state for the cells
        beyond previous base coverage, so built-vs-terrain stays exact in the new
        territory. Existing evidence never changes — every spatial feature derives
        from built cells' absolute coordinates, so growth only ADMITS cells. The
        feature caches are left alone: base extension cannot change any built cell
        that already exists (the changed-cell guard), so cached features stay true."""
        self._world.grow(region)
        if not self._base_region.covers(region):
            self._world.extend_base({cell: block for cell, block in cells.items()
                                     if not self._base_region.contains(*cell)})
            self._base_region = region

    @property
    def pending_event_ids(self) -> tuple[int, ...]:
        """Events seen but not yet consumed by any correction — at session end these
        are the unconsumed ids (a tick-0 action's events, or a bug). Agent events are
        never here: they are filtered at remember()."""
        return tuple(sorted(self._events))

    @property
    def crop_escapes(self) -> int:
        """How many applied events fell OUTSIDE the capture region. Nonzero means the
        build is happening where the region cannot see it — every structure feature
        reads zero and D2 is blind. The one real session that hit this (2026-07-04,
        region anchored at spawn, build at a teleport target) showed 92/92 escapes."""
        return len(self._world.escaped)

    def on_correction(self, correction: Evidence2D) -> Evidence3D | None:
        """One structure record for one scored B1 record (None for context records)."""
        if not correction.scored:
            return None
        if self._cached is None:
            built = self._world.built()
            self._cached = {goal: _best_instance(self._world, goal, built) for goal in GOALS}
            self._cached_global = _global_feats(self._world, tuple(self._recent_edits))
            # The shape embedding describes the same pre-action state, so it shares the
            # cache: recomputed only when an event actually changed the region.
            self._cached_h3d = self._h3d_fn(built) if self._h3d_fn is not None else None
        per_goal = {}
        for goal, (reading, subtype) in self._cached.items():
            last = self._last_comps[goal]
            if len(last) >= DELTA_COMP_WINDOW:
                baseline = last[0]                       # the reading 5 corrections ago
            else:
                baseline = self._first_comp.get(goal, reading.comp)
            per_goal[goal] = PerGoalStructure(
                comp=round(reading.comp, 4),
                edit_distance=reading.edit_distance,
                fit=round(reading.fit, 4),
                pose=reading.pose,
                subtype=subtype,
                delta_comp=round(reading.comp - baseline, 4),
            )
            self._first_comp.setdefault(goal, reading.comp)
            last.append(reading.comp)
        record = Evidence3D(
            tick=correction.tick_range[1] + 1,   # the scored action starts here (B1 contract)
            event_ids=correction.event_ids,
            scored=True,
            per_goal=per_goal,
            global_feats=self._cached_global,
            h3d=self._cached_h3d,
        )
        if correction.event_ids:
            changed = False
            for event_id in correction.event_ids:
                if event_id in self._seeded:
                    continue           # landed during catch-up: the world already holds it
                event = self._events.pop(event_id)   # pop: applied ids never pile up
                self._world.apply(event)
                self._recent_edits.append((event.pos.x, event.pos.y, event.pos.z))
                changed = True
            if changed:
                self._cached = None    # the region changed; the next record must recompute
                self._cached_global = None
                self._cached_h3d = None
        return record


def build_evidence3d(
    world: ReplayWorld,
    corrections: tuple[Evidence2D, ...],
    events_by_id: dict,
    h3d_fn: Callable[[dict], "tuple[float, ...] | None"] | None = None,
    growths: tuple = (),
) -> list[Evidence3D]:
    """One Evidence3D per scored correction, in order; events applied after their record.

    The recording driver: seed the same machine the live path uses with the session's
    events, then feed it every correction. `h3d_fn` (optional) is the learned shape
    channel — see Evidence3DStream. `growths` (region v3) is (tick, region, cells)
    triples for snapshots whose frame grew: each is adopted before the first
    correction past its tick, mirroring when the live runner meets the file.
    """
    stream = Evidence3DStream(world, h3d_fn=h3d_fn)
    stream.remember(events_by_id.values())
    pending_growth = list(growths)
    records: list[Evidence3D] = []
    for correction in corrections:
        while pending_growth and pending_growth[0][0] <= correction.tick_range[1]:
            _, grown_region, grown_cells = pending_growth.pop(0)
            stream.extend_world(grown_region, grown_cells)
        record = stream.on_correction(correction)
        if record is not None:
            records.append(record)
    return records
