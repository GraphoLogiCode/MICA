"""Run D2 over a recorded session: replay-vs-snapshot proof, then evidence3d.jsonl.

    python scripts/run_d2.py [session.jsonl]
    flags: --allow-ungated    consume a session even if it fails the B0 gate (debugging only)

The session must PASS the B0 gate AND carry region snapshots (mod 0.0.3+), because the
replay check is what earns the right to trust the voxel state. Order of proof:
  1. voxel_replay_report.json — replay events onto the starting snapshot, compare against
     every later snapshot. Class (a)/(b) divergences (world dynamics, multi-block items)
     are logged; any class (c) divergence or a build escaping the region QUARANTINES the
     session (exit 1) — there is no honest line between "small unexplained divergence"
     and "corrupted evidence".
  2. evidence3d.jsonl — one B2 record per scored B1 correction, joined by event ids.
  3. d2_provenance.json — template set, search space, windows, taxonomy version.
Exit 0 only if the replay is clean (up to the taxonomy) and the B2 contract validates.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture.discovery import newest_capture                    # noqa: E402
from mica.capture.jsonl_ingest import JsonlSource                    # noqa: E402
from mica.contracts.b2 import ROTATIONS                              # noqa: E402
from mica.perception.evidence2d import evidence_stream               # noqa: E402
from mica.perception.evidence3d import DELTA_COMP_WINDOW, build_evidence3d   # noqa: E402
from mica.perception.templates import TEMPLATE_SET_VERSION, TEMPLATES  # noqa: E402
from mica.perception.voxel_replay import (                           # noqa: E402
    TAXONOMY_VERSION, ReplayWorld, compare_to_snapshot, load_snapshot, region_from_manifest,
)
from mica.validation.b0_gate import gate_checks                      # noqa: E402
from mica.validation.evidence3d_check import check_join, check_stream  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW = os.path.join(_ROOT, "capture", "raw")


def _record_to_dict(record) -> dict:
    g = record.global_feats
    return {
        "tick": record.tick,
        "event_ids": list(record.event_ids),
        "scored": record.scored,
        "per_goal": {
            goal: {
                "comp": feats.comp, "edit_distance": feats.edit_distance, "fit": feats.fit,
                "pose": {"dx": feats.pose.dx, "dz": feats.pose.dz, "rot": feats.pose.rot},
                "subtype": feats.subtype,
                "delta_comp": feats.delta_comp,
            }
            for goal, feats in record.per_goal.items()
        },
        "global": {
            "built_count": g.built_count, "bbox": list(g.bbox) if g.bbox else None,
            "centroid": list(g.centroid) if g.centroid else None,
            "planar_runs": g.planar_runs, "has_enclosure": g.has_enclosure,
            "symmetry": g.symmetry, "symmetry_support": g.symmetry_support,
            "edit_locality": g.edit_locality,
        },
        "voxel_patch": None,   # recomputable by replay; materialized when D3 trains the encoder
        "h3d": None,
    }


def _replay_and_verify(session, snapshot_paths, world):
    """Apply events between snapshots and collect every divergence, tagged by class."""
    events = sorted((event for packet in session.packets for event in packet.server.block_events),
                    key=lambda event: event.event_id)
    event_tick = {event.event_id: packet.tick for packet in session.packets
                  for event in packet.server.block_events}
    divergences = []
    applied = 0
    for path in snapshot_paths[1:]:
        snap_tick, _, cells = load_snapshot(path)
        while applied < len(events) and event_tick[events[applied].event_id] <= snap_tick:
            world.apply(events[applied])
            applied += 1
        for divergence in compare_to_snapshot(world, cells):
            divergences.append((snap_tick, divergence))
    while applied < len(events):   # events after the last snapshot still shape the features
        world.apply(events[applied])
        applied += 1
    return divergences


def main() -> int:
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    jsonl = positional[0] if positional else newest_capture(_RAW)
    if not jsonl or not os.path.exists(jsonl):
        print("no capture found")
        return 1
    session = JsonlSource(jsonl, jsonl.replace(".jsonl", ".manifest.json")).load()
    capture_dir = os.path.dirname(os.path.abspath(jsonl))

    failing = [(n, d) for n, ok, d in gate_checks(session, capture_dir) if not ok]
    region = region_from_manifest(session)
    if region is None:
        failing.append(("region snapshots (D2)", "no snapshot_region in manifest - pre-0.0.3 capture"))
    if failing and "--allow-ungated" not in sys.argv:
        print(f"D2  {os.path.basename(jsonl)}")
        print("  REFUSED: this session is not D2-ready:")
        for name, detail in failing:
            print(f"    [FAIL] {name}  {detail}")
        return 1

    snap_dir = os.path.join(capture_dir, session.manifest.session_id, "snapshots")
    snapshot_paths = sorted(glob.glob(os.path.join(snap_dir, "*.json")),
                            key=lambda p: int(os.path.basename(p)[:-5]))
    _, _, base_cells = load_snapshot(snapshot_paths[0])
    world = ReplayWorld(region, base_cells)

    print(f"D2  {os.path.basename(jsonl)}")
    divergences = _replay_and_verify(session, snapshot_paths, world)
    by_class = {"a": [], "b": [], "c": []}
    for snap_tick, div in divergences:
        by_class[div.taxonomy_class].append((snap_tick, div))
    report = {
        "session_id": session.manifest.session_id,
        "divergence_taxonomy_version": TAXONOMY_VERSION,
        "snapshots_compared": len(snapshot_paths) - 1,
        "divergences": [
            {"snapshot_tick": t, "cell": list(d.cell), "replay": d.replay_block,
             "world": d.world_block, "class": d.taxonomy_class}
            for t, d in divergences
        ],
        "crop_escapes": len(world.escaped),
        "quarantined": bool(by_class["c"]) or bool(world.escaped),
    }
    report_path = jsonl.replace(".jsonl", ".voxel_replay_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"  replay: {len(snapshot_paths) - 1} snapshots compared; divergences"
          f" a={len(by_class['a'])} b={len(by_class['b'])} c={len(by_class['c'])};"
          f" crop escapes {len(world.escaped)}  ->  {os.path.basename(report_path)}")
    if report["quarantined"]:
        print("  [FAIL] replay check - session QUARANTINED (class-c divergence or crop escape)")
        return 1
    print("  [PASS] replay check (all divergences in documented classes)")

    # The features must see the same predict-observe-correct order the tracker will, so
    # rebuild the world from the base and interleave corrections with their own events.
    corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
    events_by_id = {event.event_id: event for packet in session.packets
                    for event in packet.server.block_events}
    fresh_world = ReplayWorld(region, base_cells)
    records = build_evidence3d(fresh_world, corrections, events_by_id)

    out = jsonl.replace(".jsonl", ".evidence3d.jsonl")
    with open(out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_record_to_dict(record)) + "\n")
    # Rank fit-first (player-anchored), comp as tiebreak — the same call the structure-only
    # baseline makes. comp alone can be inflated by terrain, which satisfies cells by design.
    final_top = max(records[-1].per_goal.items(),
                    key=lambda kv: (kv[1].fit, kv[1].comp)) if records else None
    print(f"  evidence records: {len(records)} scored  ->  {os.path.basename(out)}")
    if final_top:
        goal, feats = final_top
        print(f"  final structure: top category '{goal}' (style '{feats.subtype}')"
              f" fit={feats.fit:.2f} comp={feats.comp:.2f}"
              f" pose=({feats.pose.dx},{feats.pose.dz},rot{feats.pose.rot})")

    from mica.contracts.goals import TAXONOMY_VERSION as GOAL_TAXONOMY_VERSION

    provenance = {
        "template_set_version": TEMPLATE_SET_VERSION,
        "goal_taxonomy_version": GOAL_TAXONOMY_VERSION,
        "templates": {goal: [t.name for t in ts] for goal, ts in TEMPLATES.items()},
        "rotations": list(ROTATIONS),
        "delta_comp_window": DELTA_COMP_WINDOW,
        "divergence_taxonomy_version": TAXONOMY_VERSION,
    }
    prov_path = jsonl.replace(".jsonl", ".d2_provenance.json")
    with open(prov_path, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    print(f"  provenance: {os.path.basename(prov_path)}")

    issues = check_stream(records) + check_join(list(corrections), records)
    print(f"  [{'PASS' if not issues else 'FAIL'}] B2 contract + join validation"
          + ("" if not issues else f"  ({len(issues)} issues; first: {issues[0]})"))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
