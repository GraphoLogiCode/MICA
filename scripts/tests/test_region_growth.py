"""Region v3: the capture frame is grow-only. Growth must admit new territory with
its base captured pre-build, never re-frame existing evidence, judge escapes against
the frame in force at adjudication, and keep the offline feature pass in step with
what the live runner sees. The failure this exists for: two real sessions were voided
by a stray first block pinning the frame away from the build."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import run_live  # noqa: E402

import json

import pytest

from mica.capture.synthetic import ScriptedBuild, ScriptedPlacement, generate_session
from mica.contracts.b0 import BlockEvent, BlockOp, BlockPos
from mica.contracts.serialize import packet_to_dict
from mica.perception.evidence3d import Evidence3DStream, build_evidence3d
from mica.perception.evidence2d import evidence_stream
from mica.perception.voxel_replay import Region, ReplayWorld, SnapshotMonitor

_SID = "growtest-0001"
_SMALL = Region(-8, 56, -8, 12, 72, 8)
_GROWN = Region(-8, 56, -8, 40, 72, 8)     # east faces pushed out; a strict superset
_PLANKS = "minecraft:oak_planks"


def _place(eid, x, y, z, block=_PLANKS):
    return BlockEvent(event_id=eid, pos=BlockPos(x, y, z), block_type=block,
                      op=BlockOp.PLACE, actor="tester")


def _packet(tick, *events):
    from types import SimpleNamespace
    return SimpleNamespace(tick=tick, server=SimpleNamespace(block_events=events))


def test_region_grows_only():
    world = ReplayWorld(_SMALL, {})
    world.grow(_SMALL)                                  # equal: no-op
    world.grow(Region(-8, 56, -8, 10, 70, 8))           # smaller: old news, no-op
    assert world.region == _SMALL
    world.grow(_GROWN)                                  # superset: adopted
    assert world.region == _GROWN
    with pytest.raises(AssertionError):                 # moved: never legal
        world.grow(Region(0, 56, 0, 48, 72, 16))


def test_growth_snapshot_adopts_base_and_absolves_the_event():
    # an event lands beyond the old frame; the frame grows and its growth snapshot
    # arrives before adjudication — the event is IN frame when judged, the new
    # territory's terrain becomes base, and nothing escapes or diverges
    monitor = SnapshotMonitor(_SMALL, {})
    monitor.apply_packet(_packet(120, _place(0, 30, 64, 0)))
    terrain = {(25, 60, 0): "minecraft:stone"}
    snapshot_cells = {(30, 64, 0): _PLANKS, **terrain}
    found = monitor.on_snapshot(150, dict(snapshot_cells), _GROWN)
    assert found == [] and not monitor.quarantined
    assert len(monitor.world.escaped) == 0
    assert monitor.world.built() == {(30, 64, 0): _PLANKS}   # the block, not the terrain
    assert monitor.world.base[(25, 60, 0)] == "minecraft:stone"


def test_event_still_outside_the_grown_frame_escapes():
    monitor = SnapshotMonitor(_SMALL, {})
    monitor.apply_packet(_packet(120, _place(0, 300, 64, 0)))   # far beyond any growth
    monitor.on_snapshot(150, {}, _GROWN)
    assert len(monitor.world.escaped) == 1 and monitor.quarantined


def test_growth_base_never_swallows_a_margin_race_placement():
    # the player's block lands in new territory BEFORE the growth snapshot's tick;
    # the snapshot therefore CONTAINS their block — the changed-cell guard must keep
    # it out of the adopted base, or built() would lose a real placement
    monitor = SnapshotMonitor(_SMALL, {})
    monitor.apply_packet(_packet(100, _place(0, 30, 64, 0)))
    found = monitor.on_snapshot(150, {(30, 64, 0): _PLANKS}, _GROWN)
    assert found == []
    assert monitor.world.built() == {(30, 64, 0): _PLANKS}
    assert (30, 64, 0) not in monitor.world.base


def test_feature_world_growth_keeps_built_vs_terrain_exact():
    d2 = Evidence3DStream(ReplayWorld(_SMALL, {}))
    d2.notice_region(_GROWN)                            # frame widens (manifest first)
    d2.extend_world(_GROWN, {(25, 60, 0): "minecraft:stone"})
    d2.seed_history([_place(0, 25, 61, 0)])             # placed ON adopted terrain
    assert d2._world.built() == {(25, 61, 0): _PLANKS}
    assert d2.crop_escapes == 0


def test_offline_feature_pass_adopts_growth_like_live():
    # the parity case: the player breaks adopted terrain and puts the SAME block
    # back. With the growth base the cell is not "built" (block == base); without
    # it the offline pass would count it — and disagree with the live world.
    build = ScriptedBuild(session_id=_SID, held_item="minecraft:stone", placements=(
        ScriptedPlacement(tick=60, pos=BlockPos(25, 64, 0), block_type="minecraft:stone",
                          op=BlockOp.BREAK),
        ScriptedPlacement(tick=70, pos=BlockPos(25, 64, 0), block_type="minecraft:stone"),
    ))
    session = generate_session(build)
    corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    terrain = {(25, 64, 0): "minecraft:stone"}

    # the growth arm starts at the ORIGINAL frame, exactly like a real session —
    # the growth triple both widens it and adopts the terrain base
    with_growth = build_evidence3d(ReplayWorld(_SMALL, {}), corrections, dict(events),
                                   growths=((50, _GROWN, dict(terrain)),))
    without = build_evidence3d(ReplayWorld(_GROWN, {}), corrections, dict(events))
    assert with_growth[-1].global_feats.built_count == 0     # base block put back
    assert without[-1].global_feats.built_count == 1         # blind pass miscounts


def _write_snapshot(snap_dir, tick, region, cells):
    x_count = region.x1 - region.x0 + 1
    z_count = region.z1 - region.z0 + 1
    palette = ["minecraft:air"] + sorted(set(cells.values()))
    index_of = {block: i for i, block in enumerate(palette)}
    runs = []
    for y in range(region.y0, region.y1 + 1):
        for z in range(region.z0, region.z1 + 1):
            for x in range(region.x0, region.x1 + 1):
                index = index_of.get(cells.get((x, y, z), "minecraft:air"), 0)
                if runs and runs[-1][1] == index:
                    runs[-1][0] += 1
                else:
                    runs.append([1, index])
    (snap_dir / f"{tick}.json").write_text(json.dumps(
        {"tick": tick, "region": [region.x0, region.y0, region.z0,
                                  region.x1, region.y1, region.z1],
         "palette": palette, "runs": runs}), encoding="utf-8")


def test_late_attach_catch_up_walks_growth(tmp_path):
    # a session whose frame grew mid-play, attached late: the catch-up must adopt
    # the growth so far-away pre-attach blocks read as built, not as escapes
    build = ScriptedBuild(session_id=_SID, held_item=_PLANKS, placements=(
        ScriptedPlacement(tick=5, pos=BlockPos(0, 64, 0), block_type=_PLANKS),
        ScriptedPlacement(tick=200, pos=BlockPos(30, 64, 0), block_type=_PLANKS),
        ScriptedPlacement(tick=210, pos=BlockPos(31, 64, 0), block_type=_PLANKS),
    ))
    session = generate_session(build)
    jsonl = tmp_path / f"{_SID}.jsonl"
    dicts = [packet_to_dict(p) for p in sorted(session.packets, key=lambda p: p.tick)]
    jsonl.write_text("".join(json.dumps(d) + "\n" for d in dicts), encoding="utf-8")
    (tmp_path / f"{_SID}.manifest.json").write_text(json.dumps({
        "session_id": _SID, "session_start_ms": 0, "mod_version": "fabric-b0-0.0.6",
        "event_schema_version": "1", "frame_every": 1, "frame_width_px": 4,
        "snapshot_quiet_ticks": 40,
        "snapshot_region": [_GROWN.x0, _GROWN.y0, _GROWN.z0,
                            _GROWN.x1, _GROWN.y1, _GROWN.z1],   # manifest = CURRENT box
        "declared_event_count": session.declared_event_count}), encoding="utf-8")
    snap_dir = tmp_path / _SID / "snapshots"
    snap_dir.mkdir(parents=True)
    _write_snapshot(snap_dir, 0, _SMALL, {})                       # original frame
    _write_snapshot(snap_dir, 100, _GROWN, {(0, 64, 0): _PLANKS})  # growth snapshot
    _write_snapshot(snap_dir, 300, _GROWN, {(0, 64, 0): _PLANKS, (30, 64, 0): _PLANKS,
                                            (31, 64, 0): _PLANKS})

    d2, monitor, seed_tick = run_live._seed_structure(str(jsonl))
    assert d2._world.region == _SMALL                  # seeded from the base file's frame
    run_live._catch_up_structure(d2, monitor, str(jsonl), _SID, cutoff_tick=10 ** 9)
    assert d2._world.region == _GROWN
    assert set(d2._world.built()) == {(0, 64, 0), (30, 64, 0), (31, 64, 0)}
    assert d2.crop_escapes == 0
    assert monitor.divergence_count == 0 and not monitor.quarantined
    assert len(monitor.world.escaped) == 0
