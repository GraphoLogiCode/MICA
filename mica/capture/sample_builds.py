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
