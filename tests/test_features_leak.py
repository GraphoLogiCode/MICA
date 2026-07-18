"""The featurizer leak rule (review 13 F1, 2026-07-16): no goal-free feature the
trained heads read may be a deterministic function of the action they predict.

The snapshot-rule assertions guard WINDOW timing; nothing guarded FEATURE content —
which is how the idle flag (idle == a_hat is IDLE, by contract) sat inside
shared_dense from Phase E to 2026-07-16 and let both heads read one bit of the
answer. This test pins the rule at the featurizer itself: change ONLY the record's
scored action (a_hat, and the idle flag that is defined from it), and every feature
vector the heads consume must not move.
"""
import dataclasses

from mica.contracts.b1 import Evidence2D, Focus, MacroAction, StateFeats
from mica.contracts.b2 import Evidence3D, GlobalStructure, PerGoalStructure, Pose
from mica.contracts.b3 import fuse
from mica.contracts.goals import GOALS
from mica.intent import features


def _fused(a_hat: MacroAction):
    b1 = Evidence2D(
        tick_range=(0, 19),
        a_hat=a_hat,
        a_hat_conf=1.0,
        idle=a_hat is MacroAction.IDLE,          # the contract invariant, exactly
        state_feats=StateFeats(
            held_item="minecraft:oak_planks",
            hotbar=("minecraft:oak_planks", "minecraft:torch"),
            pos_delta=(1.5, 0.0, -2.0),
            yaw_delta=35.0,
            pitch_delta=-10.0,
            recent_actions=("navigate", "place"),
            inventory=(("minecraft:oak_planks", 64),),
            inventory_tick=15,
        ),
        focus=Focus(block=None, dwell_ticks=0),
        scored=True,
        event_ids=(7,) if a_hat in (MacroAction.PLACE, MacroAction.BREAK) else (),
        h2d=tuple(0.1 for _ in range(features.H2D_DIM)),
        s_goal=(0.1, 0.2, 0.3, 0.2, 0.1),
    )
    b2 = Evidence3D(
        tick=20,
        event_ids=b1.event_ids,
        scored=True,
        per_goal={g: PerGoalStructure(comp=0.5, edit_distance=5, fit=0.5,
                                      pose=Pose(1, 2, 90, 64), subtype="house",
                                      delta_comp=0.1) for g in GOALS},
        global_feats=GlobalStructure(built_count=12, bbox=(0, 64, 0, 4, 66, 4),
                                     centroid=(2.0, 65.0, 2.0), planar_runs=3,
                                     has_enclosure=False, symmetry=0.8,
                                     symmetry_support=12, edit_locality=1.5),
        h3d=tuple(0.2 for _ in range(features.H3D_DIM)),
    )
    return fuse(b1, b2)


def test_no_head_feature_is_a_function_of_the_scored_action():
    # Every action class produces the identical feature vectors: the heads must
    # PREDICT a_hat, never read it. (event_ids co-varies with the action by contract;
    # it is target provenance, not a feature — nothing below consumes it.)
    baselines = None
    for a_hat in (MacroAction.PLACE, MacroAction.BREAK, MacroAction.NAVIGATE,
                  MacroAction.INSPECT, MacroAction.IDLE):
        fused = _fused(a_hat)
        vectors = {
            "shared_dense": features.shared_dense(fused),
            "global_dense": features.global_dense(fused),
            "inventory_dense": features.inventory_dense(fused),
            "goal_dense": [features.goal_dense(fused, g) for g in GOALS],
            "h2d_channel": features.channel(fused.h2d, features.H2D_DIM),
            "h3d_channel": features.channel(fused.h3d, features.H3D_DIM),
        }
        if baselines is None:
            baselines = vectors
        else:
            for name, value in vectors.items():
                assert value == baselines[name], (
                    f"{name} changed when only the scored action changed - "
                    f"a target leak (review 13 F1)")


def test_shared_dense_matches_its_declared_dimension():
    fused = _fused(MacroAction.PLACE)
    assert len(features.shared_dense(fused)) == features.SHARED_DIM
    assert len(features.global_dense(fused)) == features.GLOBAL_DIM
    assert len(features.goal_dense(fused, GOALS[0])) == features.GOAL_DIM
