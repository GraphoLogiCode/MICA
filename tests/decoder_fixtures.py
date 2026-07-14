"""Hand-built fixtures for the decoder-layer tests — in their own file, per the
project's fixtures-live-in-their-own-file convention.

The toy placement script below is small enough to check by eye: six placements on a
line at y=64, with ONE deliberate mistake at (2, 64, 1) that gets broken two ops
later. Every expected helper trace in the tests is derived from this by hand.
"""
from __future__ import annotations

from mica.capture.synthetic import ScriptedPlacement
from mica.contracts.b0 import BlockOp, BlockPos
from mica.contracts.b1 import GOALS, Focus, MacroAction, StateFeats
from mica.contracts.b2 import GlobalStructure, PerGoalStructure, Pose
from mica.contracts.b3 import FusedEvidence

_BLOCK = "minecraft:oak_planks"

_SUBTYPE_OF = {"habitation": "cabin", "infrastructure": "flat span",
               "production": "fence pen", "defense": "square tower",
               "decorative": "fountain"}


def toy_placements() -> tuple[ScriptedPlacement, ...]:
    """Ticks 10..70: places (0,64,0), (1,64,0), a MISTAKE at (2,64,1), then
    (2,64,0), the break of the mistake, then (3,64,0) and (4,64,0)."""
    return (
        ScriptedPlacement(tick=10, pos=BlockPos(0, 64, 0), block_type=_BLOCK),
        ScriptedPlacement(tick=20, pos=BlockPos(1, 64, 0), block_type=_BLOCK),
        ScriptedPlacement(tick=30, pos=BlockPos(2, 64, 1), block_type=_BLOCK),  # mistake
        ScriptedPlacement(tick=40, pos=BlockPos(2, 64, 0), block_type=_BLOCK),
        ScriptedPlacement(tick=50, pos=BlockPos(2, 64, 1), block_type=_BLOCK,
                          op=BlockOp.BREAK),                                    # its fix
        ScriptedPlacement(tick=60, pos=BlockPos(3, 64, 0), block_type=_BLOCK),
        ScriptedPlacement(tick=70, pos=BlockPos(4, 64, 0), block_type=_BLOCK),
    )


def fused_record(tick: int = 100, built_count: int = 8,
                 centroid=(2.0, 64.0, 2.0), event_ids=(1,)) -> FusedEvidence:
    """One well-formed fused record, symbolic channels only (like scripted data).
    event_ids=() makes it a context-style correction (the heads-v2 event gate)."""
    per_goal = {
        g: PerGoalStructure(comp=0.5, edit_distance=5, fit=0.5, pose=Pose(0, 0, 0),
                            subtype=_SUBTYPE_OF[g], delta_comp=0.0)
        for g in GOALS
    }
    return FusedEvidence(
        tick=tick, event_ids=tuple(event_ids), a_hat=MacroAction.PLACE, idle=False,
        state_feats=StateFeats(held_item=_BLOCK, hotbar=(_BLOCK,),
                               pos_delta=(0.0, 0.0, 0.0), yaw_delta=0.0,
                               pitch_delta=0.0, recent_actions=("place", "idle")),
        focus=Focus(block=None, dwell_ticks=0),
        global_feats=GlobalStructure(built_count=built_count,
                                     bbox=(0, 64, 0, 4, 64, 4) if built_count else None,
                                     centroid=centroid if built_count else None,
                                     planar_runs=2, has_enclosure=False, symmetry=0.5,
                                     symmetry_support=built_count, edit_locality=1.0),
        s_goal=None, per_goal=per_goal,
    )
