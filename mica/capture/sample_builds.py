"""Ready-made pretend recordings, kept out of the engine so the engine stays clean.

The tests and the proof file build on these instead of hand-writing block changes.
"""
from __future__ import annotations

from ..contracts.b0 import BlockOp, BlockPos
from .synthetic import ScriptedBuild, ScriptedPlacement

_ROW_Y = 64  # the height these sample rows are built at
_ROW_LENGTH = 5  # blocks in the wall row
_TICKS_PER_PLACEMENT = 5  # steps between each placement


def wall_row_build(session_id: str = "synthetic-wall-row") -> ScriptedBuild:
    """A tiny sample build — five blocks in a row, one every five steps."""
    block_type = "minecraft:oak_planks"
    placements = tuple(
        ScriptedPlacement(
            tick=_TICKS_PER_PLACEMENT * (i + 1),
            pos=BlockPos(x=i, y=_ROW_Y, z=0),
            block_type=block_type,
        )
        for i in range(_ROW_LENGTH)
    )
    return ScriptedBuild(session_id=session_id, held_item=block_type, placements=placements)


def walk_only_build(session_id: str = "synthetic-walk-only") -> ScriptedBuild:
    """A recording where the player only walks around — no blocks changed."""
    return ScriptedBuild(session_id=session_id, held_item="minecraft:air", placements=())


def mixed_build(session_id: str = "synthetic-mixed") -> ScriptedBuild:
    """Places two blocks in the same moment and later breaks one — exercises every kind of change."""
    block_type = "minecraft:oak_planks"
    placements = (
        ScriptedPlacement(tick=5, pos=BlockPos(0, _ROW_Y, 0), block_type=block_type),
        ScriptedPlacement(tick=10, pos=BlockPos(1, _ROW_Y, 0), block_type=block_type),
        ScriptedPlacement(tick=10, pos=BlockPos(2, _ROW_Y, 0), block_type=block_type),
        ScriptedPlacement(tick=15, pos=BlockPos(0, _ROW_Y, 0), block_type=block_type, op=BlockOp.BREAK),
    )
    return ScriptedBuild(session_id=session_id, held_item=block_type, placements=placements)


def sparse_build(session_id: str = "synthetic-sparse") -> ScriptedBuild:
    """Two placements with a long pause between — long enough that the ~1 Hz unscored
    context records must appear between the two corrections."""
    block_type = "minecraft:oak_planks"
    placements = (
        ScriptedPlacement(tick=5, pos=BlockPos(0, _ROW_Y, 0), block_type=block_type),
        ScriptedPlacement(tick=70, pos=BlockPos(1, _ROW_Y, 0), block_type=block_type),
    )
    return ScriptedBuild(session_id=session_id, held_item=block_type, placements=placements)


def pen_build(session_id: str = "synthetic-pen") -> ScriptedBuild:
    """Builds the animal-pen template exactly: a 5x5 fence ring, one post at a time —
    the D2 fixture for completion reaching 1.0 and enclosure detection."""
    fence = "minecraft:oak_fence"
    ring = [(i, 0) for i in range(5)] + [(i, 4) for i in range(5)] \
         + [(0, j) for j in range(1, 4)] + [(4, j) for j in range(1, 4)]
    placements = tuple(
        ScriptedPlacement(tick=3 * (n + 1), pos=BlockPos(x, _ROW_Y, z), block_type=fence)
        for n, (x, z) in enumerate(sorted(ring))
    )
    return ScriptedBuild(session_id=session_id, held_item=fence, placements=placements)


def bridge_deck_build(axis: str = "x", session_id: str = "synthetic-bridge") -> ScriptedBuild:
    """Builds the railed-bridge template exactly — 7x3 deck plus both rail rows — with
    its long side along the given axis ("x" or "z"): the D2 fixture proving registration
    finds the build's rotation."""
    plank = "minecraft:oak_planks"
    cells = [(a, 0, b) for a in range(7) for b in range(3)] \
          + [(a, 1, b) for a in range(7) for b in (0, 2)]
    placements = tuple(
        ScriptedPlacement(
            tick=3 * (n + 1),
            pos=BlockPos(a, _ROW_Y + dy, b) if axis == "x" else BlockPos(b, _ROW_Y + dy, a),
            block_type=plank,
        )
        for n, (a, dy, b) in enumerate(cells)
    )
    return ScriptedBuild(session_id=session_id, held_item=plank, placements=placements)
