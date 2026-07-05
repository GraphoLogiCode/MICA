"""The h3d channel end to end, no torch required: a stub shape head stands in for
Uni3D. Covers the producer (per-correction embedding with cache semantics), the
transport (serializer + fuse round trips), and the consumer (the deliberative-only,
centered readout term — including the balance-law guarantee that a uniform readout
changes nothing at all)."""
import json

from mica.capture.sample_builds import wall_row_build
from mica.capture.synthetic import generate_session
from mica.contracts.b1 import GOALS, Focus, MacroAction, StateFeats
from mica.contracts.b2 import GlobalStructure, PerGoalStructure, Pose
from mica.contracts.b3 import FusedEvidence, fuse, fuse_dicts
from mica.contracts.serialize import evidence2d_to_dict, evidence3d_to_dict
from mica.intent import h3d_readout
from mica.intent.heads_v0 import likelihood
from mica.perception.evidence2d import evidence_stream
from mica.perception.evidence3d import build_evidence3d
from mica.perception.voxel_replay import ReplayWorld, region_around_events

_SUBTYPE_OF = {"habitation": "cabin", "infrastructure": "flat span",
               "production": "fence pen", "defense": "square tower",
               "decorative": "fountain"}


def _stub_h3d(built):
    """A deterministic stand-in for Uni3DShapeHead.h3d: None on the empty build,
    else a small vector whose first component is the built-cell count."""
    if not built:
        return None
    return (float(len(built)), 1.0, 0.0, 0.0)


def _session_pieces():
    session = generate_session(wall_row_build())
    corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    world = ReplayWorld(region_around_events(session), {})
    return corrections, build_evidence3d(world, corrections, events, h3d_fn=_stub_h3d)


def test_stream_attaches_the_pre_action_embedding_per_correction():
    _, b2 = _session_pieces()
    built_so_far = 0
    for record in b2:
        if built_so_far == 0:
            assert record.h3d is None            # nothing built yet -> no cloud
        else:
            assert record.h3d is not None
            assert record.h3d[0] == float(built_so_far)   # strictly PRE-action state
        built_so_far += len(record.event_ids)    # wall_row: every event is a placement


def test_h3d_round_trips_through_the_jsonl_forms():
    corrections, b2 = _session_pieces()
    carrying = next(i for i, r in enumerate(b2) if r.h3d is not None)
    line = json.loads(json.dumps(evidence3d_to_dict(b2[carrying])))
    assert line["h3d"] == list(b2[carrying].h3d)
    fused = fuse_dicts(json.loads(json.dumps(evidence2d_to_dict(corrections[carrying]))), line)
    assert fused.h3d == b2[carrying].h3d
    # and the empty case stays None on both sides
    empty = json.loads(json.dumps(evidence3d_to_dict(b2[0])))
    assert empty["h3d"] is None
    assert fuse_dicts(evidence2d_to_dict(corrections[0]), empty).h3d is None


def test_typed_fusion_carries_h3d():
    corrections, b2 = _session_pieces()
    carrying = next(i for i, r in enumerate(b2) if r.h3d is not None)
    assert fuse(corrections[carrying], b2[carrying]).h3d == b2[carrying].h3d


def _fused_with(h3d):
    per_goal = {
        g: PerGoalStructure(comp=0.5, edit_distance=5, fit=0.5, pose=Pose(0, 0, 0),
                            subtype=_SUBTYPE_OF[g], delta_comp=0.0)
        for g in GOALS
    }
    return FusedEvidence(
        tick=100, event_ids=(1,), a_hat=MacroAction.PLACE, idle=False,
        state_feats=StateFeats(held_item="minecraft:dirt", hotbar=("minecraft:dirt",),
                               pos_delta=(0.0, 0.0, 0.0), yaw_delta=0.0, pitch_delta=0.0,
                               recent_actions=("place",)),
        focus=Focus(block=None, dwell_ticks=0),
        global_feats=GlobalStructure(built_count=8, bbox=(0, 64, 0, 4, 64, 4),
                                     centroid=(2.0, 64.0, 2.0), planar_runs=2,
                                     has_enclosure=False, symmetry=0.5,
                                     symmetry_support=8, edit_locality=1.0),
        s_goal=None, per_goal=per_goal, h3d=h3d,
    )


def test_readout_term_is_deliberative_only_and_balance_law_clean():
    # a readout that reads the stub's first component as "production"
    production_row = [10.0, 0.0, 0.0, 0.0]
    zero_row = [0.0, 0.0, 0.0, 0.0]
    favoring = h3d_readout.H3dReadout(
        GOALS, [production_row if g == "production" else zero_row for g in GOALS],
        [0.0] * len(GOALS))
    uniform = h3d_readout.H3dReadout(GOALS, [zero_row for _ in GOALS], [0.0] * len(GOALS))
    evidence = _fused_with((1.0, 1.0, 0.0, 0.0))
    try:
        h3d_readout.activate(None)
        baseline = likelihood(evidence, MacroAction.PLACE)

        h3d_readout.activate(favoring)
        favored = likelihood(evidence, MacroAction.PLACE)
        # deliberative side moves: the readout's favored goal gains, a disfavored one loses
        assert favored[("production", 0)] > baseline[("production", 0)]
        assert favored[("decorative", 0)] < baseline[("decorative", 0)]
        # heuristic side is untouchable — goal symmetry survives the new channel
        assert all(favored[(g, 1)] == baseline[(g, 1)] for g in GOALS)
        # no embedding -> no term, even with a readout active
        assert likelihood(_fused_with(None), MacroAction.PLACE) == baseline

        # the balance law, exactly: a uniform readout centers to zero and changes NOTHING
        h3d_readout.activate(uniform)
        assert likelihood(evidence, MacroAction.PLACE) == baseline
    finally:
        h3d_readout.use_default()
