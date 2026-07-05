"""Source A: scripted goal-labeled builds, generated with deliberate variance.

Every session builds one goal type's template shape — so the label is known because the
generator wrote it — but no two sessions build it the same way. The variance axes exist
to stop a likelihood head from memorizing one fixed action sequence per goal:

  style       which taxonomy subtype gets built (a fence-ring pen vs a post-and-rail
              pen) — within-category variation, so a head must learn the CATEGORY from
              styles that share it rather than memorizing one shape
  pose        the build sits at a random offset and yaw, so position/rotation carry no
              goal information (registration must earn the match)
  order       layer-by-layer, outline-then-fill, wall-by-wall, or scattered — the same
              shape reached through different action sequences
  pacing      gaps between placements vary, with multi-second pauses (idle segments and
              real Δt variance for the tracker's time-aware kernels)
  mode        deliberate (orderly, careful) vs shortcut (scattered, hasty, error-prone) —
              ground-truth labels for the z latent
  mistakes    misplaced blocks that get broken and are never part of the shape — break
              events, non-monotone edit distance, honest mess
  completion  some sessions stop at 85-100% — partially finished builds exist in life
  palette     goals that accept any solid block get built from varied materials
  gaze        the mode's motion signature: deliberate builders walk, aim, linger on
              fresh blocks, and sweep the structure during pauses; shortcut builders
              flick and rush (the MotionStyle presets in capture.motion)

What this deliberately does NOT provide: pixel frames (synthetic sessions have none, so
h2d/s_goal need real in-game captures), free-form non-template builds (that is Source B's
hindsight-labeled data), and real builder diversity (one generator is one "builder";
the holdout discipline needs humans). Scripted data bootstraps the heads and calibrates
the machinery — it is the controlled condition, not the generalization proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random

from ..contracts.b0 import BlockOp, BlockPos
from ..contracts.goals import TAXONOMY
from ..perception.templates import REQ_SOLID, instance, rotate_offset
from .motion import DELIBERATE_GAZE, SHORTCUT_GAZE
from .synthetic import ScriptedBuild, ScriptedPlacement

_GROUND_Y = 64
_ORDERS = ("layers", "outline", "walls", "scattered")
# Solid-cell goals draw from these; the pen's fences are fixed by its template.
_PALETTES = ("minecraft:oak_planks", "minecraft:cobblestone", "minecraft:spruce_planks",
             "minecraft:stone_bricks")


@dataclass(frozen=True)
class BuildPlan:
    """Everything that makes one scripted session the particular session it is."""

    goal: str
    subtype: str                 # which style of the goal gets built
    seed: int
    mode: str                    # "deliberate" | "shortcut"
    origin: tuple[int, int]      # (x, z) world offset of the template's local origin
    rotation: int                # yaw the build is placed at
    order: str
    base_gap_ticks: int          # typical ticks between placements
    completion: float            # fraction of template cells actually placed
    mistake_rate: float          # chance a placement is a misplaced block, later broken
    block: str                   # material used for REQ_SOLID cells


def plan_variants(goal: str, count: int, seed: int) -> list[BuildPlan]:
    """Deterministic spread of plans for one goal — same seed, same corpus, always."""
    rng = Random((goal, seed).__repr__())
    subtypes = TAXONOMY[goal]
    plans = []
    for index in range(count):
        shortcut = index % 3 == 2   # every third session is the hasty kind
        plans.append(BuildPlan(
            goal=goal,
            subtype=subtypes[index % len(subtypes)],   # styles alternate deterministically
            seed=seed * 1000 + index,
            mode="shortcut" if shortcut else "deliberate",
            origin=(rng.randint(-12, 12), rng.randint(-12, 12)),
            rotation=rng.choice((0, 90, 180, 270)),
            order="scattered" if shortcut else rng.choice(_ORDERS[:3]),
            base_gap_ticks=rng.randint(3, 6) if shortcut else rng.randint(6, 14),
            completion=1.0 if index == 0 else round(rng.uniform(0.85, 1.0), 2),
            mistake_rate=round(rng.uniform(0.05, 0.15), 2) if shortcut
                         else round(rng.uniform(0.0, 0.05), 2),
            block=rng.choice(_PALETTES),
        ))
    return plans


def _world_cells(plan: BuildPlan) -> list[tuple[tuple[int, int, int], str]]:
    """The plan's style instance moved to the plan's pose, with the plan's material."""
    template = instance(plan.subtype)
    cells = []
    for offset, required in template.cells.items():
        dx, dy, dz = rotate_offset(offset, plan.rotation)
        cell = (dx + plan.origin[0], dy + _GROUND_Y, dz + plan.origin[1])
        cells.append((cell, plan.block if required == REQ_SOLID else required))
    return cells


def _ordered(cells, plan: BuildPlan, rng: Random):
    """The same shape reached through a different action sequence per strategy."""
    if plan.order == "layers":
        return sorted(cells, key=lambda c: (c[0][1], c[0][2], c[0][0]))
    if plan.order == "outline":
        # each layer's rim before its interior — the "trace the shape first" builder
        def rim_key(entry):
            (x, y, z), _ = entry
            layer = [c for c, _ in cells if c[1] == y]
            xs, zs = [c[0] for c in layer], [c[2] for c in layer]
            on_rim = x in (min(xs), max(xs)) or z in (min(zs), max(zs))
            return (y, not on_rim, z, x)
        return sorted(cells, key=rim_key)
    if plan.order == "walls":
        # finish one side before starting the next — the "wall by wall" builder
        def side(entry):
            (x, y, z), _ = entry
            return (0 if x <= plan.origin[0] else 1 if z <= plan.origin[1] else 2, x, z, y)
        return sorted(cells, key=side)
    shuffled = list(cells)
    rng.shuffle(shuffled)
    return shuffled


def build_from_plan(plan: BuildPlan) -> tuple[ScriptedBuild, dict]:
    """One scripted session plus its ground-truth label record."""
    rng = Random(plan.seed)
    ordered = _ordered(_world_cells(plan), plan, rng)
    kept = ordered[:max(8, round(len(ordered) * plan.completion))]

    placements: list[ScriptedPlacement] = []
    pending_breaks: list[tuple[int, tuple[int, int, int], str]] = []  # (due_index, cell, block)
    tick = 0
    for index, (cell, block) in enumerate(kept):
        tick += plan.base_gap_ticks + rng.randint(0, 4)
        if rng.random() < 0.12:                      # a thinking pause, seconds long
            tick += rng.randint(40, 90)
        if rng.random() < plan.mistake_rate:
            # a misplaced block one cell off the shape — placed now, broken soon after
            wrong = (cell[0] + rng.choice((-1, 1)), cell[1], cell[2] + rng.choice((-1, 1)))
            placements.append(ScriptedPlacement(tick=tick, pos=BlockPos(*wrong), block_type=block))
            pending_breaks.append((index + rng.randint(2, 5), wrong, block))
            tick += plan.base_gap_ticks
        placements.append(ScriptedPlacement(tick=tick, pos=BlockPos(*cell), block_type=block))
        due = [b for b in pending_breaks if b[0] <= index]
        for _, wrong_cell, wrong_block in due:
            tick += 2
            placements.append(ScriptedPlacement(
                tick=tick, pos=BlockPos(*wrong_cell), block_type=wrong_block, op=BlockOp.BREAK))
        pending_breaks = [b for b in pending_breaks if b[0] > index]

    build = ScriptedBuild(
        session_id=f"scripted-{plan.subtype.replace(' ', '-')}-{plan.seed}",
        held_item=plan.block,
        placements=tuple(placements),
        motion=DELIBERATE_GAZE if plan.mode == "deliberate" else SHORTCUT_GAZE,
    )
    label = {
        "goal": plan.goal, "subtype": plan.subtype, "mode": plan.mode,
        "origin": list(plan.origin), "rotation": plan.rotation, "order": plan.order,
        "completion": plan.completion, "mistake_rate": plan.mistake_rate,
        "block": plan.block, "seed": plan.seed, "events": len(placements),
    }
    return build, label
