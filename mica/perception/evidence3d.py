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

from dataclasses import dataclass

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
    # A category is backed by one instance per style; the category feature is the best
    # instance's reading, and that instance's name is the style read (coarse-to-fine).
    #
    # Ranked by fit x comp (D2 gate 2026-07-03 F1), because each factor alone fails:
    # fit is blind to missing upper layers (a bare deck would read "railed bridge" —
    # rails share the deck's footprint), and comp is terrain-inflatable (a flat template
    # soaks up natural ground and wins the style slot on real terrain — observed live).
    # The product demands both: the shape is present AND the player put it there.
    readings = [(_read_goal(world, template, built), template.name)
                for template in TEMPLATES[goal]]
    return max(readings, key=lambda pair: (pair[0].fit * pair[0].comp, pair[0].fit))


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


def build_evidence3d(
    world: ReplayWorld,
    corrections: tuple[Evidence2D, ...],
    events_by_id: dict,
) -> list[Evidence3D]:
    """One Evidence3D per scored correction, in order; events applied after their record."""
    recent_edits: list[tuple[int, int, int]] = []
    comp_history: dict[str, list[float]] = {goal: [] for goal in GOALS}
    cached: dict[str, tuple[_GoalReading, str]] | None = None
    cached_global: GlobalStructure | None = None
    records: list[Evidence3D] = []
    for correction in corrections:
        if not correction.scored:
            continue
        if cached is None:
            built = world.built()
            cached = {goal: _best_instance(world, goal, built) for goal in GOALS}
            cached_global = _global_feats(world, tuple(recent_edits[-_RECENT_EDITS:]))
        per_goal = {}
        for goal, (reading, subtype) in cached.items():
            history = comp_history[goal]
            baseline = history[-DELTA_COMP_WINDOW] if len(history) >= DELTA_COMP_WINDOW \
                else (history[0] if history else reading.comp)
            per_goal[goal] = PerGoalStructure(
                comp=round(reading.comp, 4),
                edit_distance=reading.edit_distance,
                fit=round(reading.fit, 4),
                pose=reading.pose,
                subtype=subtype,
                delta_comp=round(reading.comp - baseline, 4),
            )
            history.append(reading.comp)
        records.append(Evidence3D(
            tick=correction.tick_range[1] + 1,   # the scored action starts here (B1 contract)
            event_ids=correction.event_ids,
            scored=True,
            per_goal=per_goal,
            global_feats=cached_global,
        ))
        if correction.event_ids:
            for event_id in correction.event_ids:
                event = events_by_id[event_id]
                world.apply(event)
                recent_edits.append((event.pos.x, event.pos.y, event.pos.z))
            cached = None        # the region changed; the next record must recompute
            cached_global = None
    return records
