"""One step of 3D structure evidence: how far the build matches each goal's shape.

This is B2 (Evidence3D), the output of D2. Like B1 it is **evidence, never intent** —
the belief tracker scores goal hypotheses against it. Every feature is computed on the
world state strictly BEFORE the scored action's first block change applies (the snapshot
rule), rebuilt by replaying events onto the session's starting snapshot.

The per-goal features are g-indexed, so they may reach the DELIBERATIVE likelihood head
only — feeding them to the heuristic head would quietly break the mode identifiability
the safety gate depends on (the goal-symmetry rule in D0). The global features are
goal-free and shared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .b1 import GOALS  # the one goal set; fixes the per_goal key set

ROTATIONS = (0, 90, 180, 270)   # the yaw poses registration searches (no mirrors in v1)


@dataclass(frozen=True)
class Pose:
    """Where and which way round a template best fits: a footprint offset plus a yaw."""

    dx: int
    dz: int
    rot: int   # one of ROTATIONS


@dataclass(frozen=True)
class PerGoalStructure:
    """How far the build matches ONE category, at its best style instance and pose.

    A category is backed by one template instance per taxonomy subtype; the winning
    instance's name IS the style read — the fine layer of coarse-to-fine inference,
    delivered without widening any per-category vector.

    Field contract:
      comp          float in [0, 1]   fraction of template cells the world already satisfies
                                      (pre-existing terrain counts — building on it is building)
      edit_distance int >= 0          template cells still missing + player-placed surplus
                                      inside the footprint (a hill is never surplus)
      fit           float in [0, 1]   footprint IoU of the template against PLAYER-BUILT cells
                                      at the winning pose; 0.0 when nothing is built yet
      pose          Pose              the winning offset+rotation — the future hybrid goal's
                                      spatial_target
      subtype       str               the winning instance's taxonomy subtype (the style)
      delta_comp    float in [-1, 1]  comp now minus comp W corrections ago (progress signal)
    """

    comp: float
    edit_distance: int
    fit: float
    pose: Pose
    subtype: str
    delta_comp: float


@dataclass(frozen=True)
class GlobalStructure:
    """Goal-free shape facts about what the player has built. Shared by adapter and both heads.

    bbox/centroid are None until the first block is placed. symmetry_support counts the
    built cells behind the symmetry score — a high score over three blocks means nothing,
    and the consumer must be able to see that. edit_locality is the spread of the recent
    edits around their own centroid (small = focused work); None until two edits exist.
    """

    built_count: int
    bbox: tuple[int, int, int, int, int, int] | None
    centroid: tuple[float, float, float] | None
    planar_runs: int          # maximal straight runs (length >= 3) along x or z
    has_enclosure: bool       # the built footprint closes a loop around empty cells
    symmetry: float           # best mirror-fraction over 4 vertical planes through centroid
    symmetry_support: int
    edit_locality: float | None


@dataclass(frozen=True)
class Evidence3D:
    """One B2 record: the 3D structure evidence for a single correction step.

    Field contract (amended D0 B2 box, 2026-07-02):
      tick        int                 the scored action's tick τ(k). Features are computed
                                      on the state before this correction's FIRST event.
      event_ids   tuple[int]          the same ids the correction's B1 record carries — the
                                      D3 join key. One correction, one (B1, B2) pair, once.
      scored      bool                B1's vocabulary. v1 emits scored records only; a
                                      prediction-only step reuses the last record, which is
                                      exact because the region changes only through events.
      per_goal    mapping goal->PerGoalStructure, keys exactly GOALS — deliberative head only.
      global_feats GlobalStructure    goal-symmetric, shared.
      voxel_patch None in v1          recomputable from raw logs by replay (Prop 1); it is
                                      materialized only when D3 trains the voxel encoder.
      h3d         None in v1          zero-filled/absent identically across arms; d3 is D3's.
    """

    tick: int
    event_ids: tuple[int, ...]
    scored: bool
    per_goal: Mapping[str, PerGoalStructure]
    global_feats: GlobalStructure
    voxel_patch: None = None
    h3d: tuple[float, ...] | None = None
