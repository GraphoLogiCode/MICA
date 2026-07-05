"""The per-session agent-scan report — the earn-its-place artifact for h3d_scan.

    python scripts/run_agent_scan.py <session.jsonl> [--scan <scan.jsonl>] [--h3d]

Pairs a finished capture session with the agent's scan file and answers, with
numbers: how much of the build did the agent actually see (coverage curve), and —
with --h3d — does the embedding of the scanned portion converge toward the exact
build's embedding as coverage grows (cosine curve)? If it does not, the scan channel
carries a different signal and stops before any belief wiring (design note:
"Agent-Scan 3D Channel"). Writes <session>.agent_scan_report.json.

The scan file defaults to the newest agent-*.scan*.jsonl next to the session —
including rotated ones from earlier launches, so a report can run after the rig
has restarted.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture.jsonl_ingest import JsonlSource                      # noqa: E402
from mica.contracts.b0 import is_agent_actor                           # noqa: E402
from mica.perception.agent_scan import (                               # noqa: E402
    accumulate, coverage, load_scan, scanned_build_cells,
)
from mica.perception.voxel_replay import (                             # noqa: E402
    Region, ReplayWorld, load_snapshot, region_from_manifest,
)

_COSINE_STEPS = 10   # sample the embedding-convergence curve at ~deciles of the scan


def _flag(name: str, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def _built_set(jsonl: str, session) -> dict:
    """The exact built set: pre-build base + the session's HUMAN events (A7)."""
    region = region_from_manifest(session)
    snap_dir = os.path.join(os.path.dirname(os.path.abspath(jsonl)),
                            session.manifest.session_id, "snapshots")
    snapshots = sorted(glob.glob(os.path.join(snap_dir, "*.json")),
                       key=lambda p: int(os.path.basename(p)[:-5]))
    if region is None or not snapshots:
        raise SystemExit("session has no region/snapshots - not D2-ready, no scan report")
    _, _, base = load_snapshot(snapshots[0])
    world = ReplayWorld(region, dict(base))
    for packet in sorted(session.packets, key=lambda p: p.tick):
        for event in packet.server.block_events:
            if not is_agent_actor(event.actor):
                world.apply(event)
    return world.built()


def _cosine(a, b) -> float | None:
    if a is None or b is None:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return round(dot / (na * nb), 4) if na and nb else None


def main() -> int:
    positional = [a for i, a in enumerate(sys.argv[1:], 1)
                  if not a.startswith("--") and sys.argv[i - 1] != "--scan"]
    if not positional:
        print(__doc__)
        return 1
    jsonl = positional[0]
    scan_arg = _flag("--scan", None)
    if scan_arg is None:
        candidates = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(jsonl)),
                                                   "agent-*.scan*.jsonl")), key=os.path.getmtime)
        if not candidates:
            print("no agent scan file found next to the session (agent-*.scan*.jsonl)")
            return 1
        scan_arg = candidates[-1]

    session = JsonlSource(jsonl, jsonl.replace(".jsonl", ".manifest.json")).load()
    built = _built_set(jsonl, session)
    # The newest scan file can already belong to the NEXT session (the agent rotates
    # its file per launch, and reports often run while a new world is open — a real
    # report once paired a bridge session with the next world's spawn-area scan).
    # Without --scan, pair by wallclock: of the scans that STARTED after this
    # session began, the earliest is this session's own agent.
    if "--scan" in sys.argv:
        sweeps = load_scan(scan_arg)
    else:
        start_s = session.manifest.session_start_ms / 1000.0
        candidates = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(jsonl)),
                                                   "agent-*.scan*.jsonl")), key=os.path.getmtime)
        scan_arg, sweeps = None, []
        for candidate in candidates:
            trial = load_scan(candidate)
            if trial and trial[0].ts >= start_s:
                scan_arg, sweeps = candidate, trial
                break
        if scan_arg is None:
            print("no agent scan overlaps this session's wallclock - no scan report")
            return 1
    scanned_total = accumulate(sweeps)

    # The coverage curve: after each sweep, how much of the build has been seen.
    curve = []
    running: dict = {}
    for sweep in sweeps:
        for x, y, z, block in sweep.cells:
            running.setdefault((x, y, z), block)
        seen_of_built, built_count = coverage(running, built)
        curve.append({"ts": sweep.ts, "tick": sweep.tick,
                      "scanned": len(running), "built_seen": seen_of_built,
                      "built_total": built_count})

    channel = scanned_build_cells(scanned_total, built)
    report = {
        "session_id": session.manifest.session_id,
        "scan_file": os.path.basename(scan_arg),
        "sweeps": len(sweeps),
        "scanned_cells": len(scanned_total),
        "built_total": len(built),
        "built_seen": len(channel),
        "coverage": round(len(channel) / len(built), 4) if built else None,
        "coverage_curve": curve,
    }

    # The convergence question (--h3d): embed the scanned portion at ~deciles of
    # the sweep sequence and ask how close each one already is to the exact
    # build's embedding. Convergence toward 1.0 as coverage grows is the signal
    # that the scan cloud carries the same shape information.
    if "--h3d" in sys.argv:
        from mica.perception.shape3d import Uni3DShapeHead, assets_ready

        ready, missing = assets_ready()
        if not ready:
            print(f"--h3d needs the Uni3D assets ({missing}); skipping the cosine curve")
        else:
            print("loading frozen Uni3D-B on GPU ...")
            head = Uni3DShapeHead()
            exact = head.h3d(built)
            samples = []
            step = max(1, len(sweeps) // _COSINE_STEPS)
            running = {}
            for index, sweep in enumerate(sweeps):
                for x, y, z, block in sweep.cells:
                    running.setdefault((x, y, z), block)
                last = index == len(sweeps) - 1
                if index % step == 0 or last:
                    partial = scanned_build_cells(running, built)
                    samples.append({
                        "sweep": index,
                        "built_seen": len(partial),
                        "cosine_vs_exact": _cosine(head.h3d(partial), exact),
                    })
            report["h3d_scan"] = {"exact_norm": round(math.sqrt(sum(v * v for v in exact)), 3)
                                  if exact else None,
                                  "convergence": samples}

    out = jsonl.replace(".jsonl", ".agent_scan_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"agent scan  {os.path.basename(scan_arg)} vs {session.manifest.session_id}")
    print(f"  sweeps {report['sweeps']}, scanned {report['scanned_cells']} cells,"
          f" build coverage {report['built_seen']}/{report['built_total']}"
          f" ({report['coverage']})")
    if "h3d_scan" in report and report["h3d_scan"]["convergence"]:
        final = report["h3d_scan"]["convergence"][-1]
        print(f"  h3d_scan vs exact h3d at full scan: cosine {final['cosine_vs_exact']}")
    print(f"  report -> {os.path.basename(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
