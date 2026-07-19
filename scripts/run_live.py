"""Run the perception -> belief pipeline LIVE while Minecraft is running — or replay a
recording through the identical live path and prove it matches the offline wrappers.

    python scripts/run_live.py                       live: connect to the mod at 127.0.0.1:25567
    python scripts/run_live.py --host H --port N     live, custom endpoint
    python scripts/run_live.py --pixels              live + the VPT/MineCLIP pixel head (GPU);
                                                     MICA_PIXELS=1 does the same, --sgoal-stride N
                                                     mirrors run_d1's clip-span axis
    python scripts/run_live.py --h3d                 live + the frozen Uni3D shape channel (GPU);
                                                     MICA_H3D=1 does the same — every B2 record
                                                     then carries h3d, so the fused evidence holds
                                                     BOTH model channels (the adapter's inputs)
    python scripts/run_live.py --place               lift the D5 §9 demo pin (amendment
                                                     2026-07-12): the gate may authorize ONE
                                                     reversible placement per read, licensed
                                                     by conf >= theta_place ONLY (declaring
                                                     during a live session is retired by
                                                     design); the agent's body executes it.
                                                     Off by default — the demo posture stands.
    python scripts/run_live.py --gather              allow the GATHER state (D5 §4 amendment):
                                                     fetch whitelisted materials for a declared
                                                     or theta_place-confident target
    python scripts/run_live.py --session <jsonl>     attach to a named session instead of
                                                     discovering the freshest one (rehearsals)
    python scripts/run_live.py <session.jsonl>       replay + golden-equivalence check (exit 0/1)

Live mode expects you to already be IN the world (the capture session must be open, so
its manifest carries the snapshot region). It writes the same artifacts the offline
scripts would — <session>.evidence2d.jsonl / .evidence3d.jsonl — plus the live-only
logs (.fused.jsonl, .belief.jsonl, .live_run.json), each line flushed as it happens:
`Get-Content <session>.belief.jsonl -Wait` is the live belief view. At session end
(quit the game normally, or Ctrl-C here) the full B0 gate runs against the disk
recording the mod wrote in parallel — the disk copy stays the authoritative artifact.

Replay mode feeds a finished recording through the SAME pipeline objects and diffs
every produced log line against the offline wrappers — the golden-equivalence proof
as one command. Nothing is written to disk in replay mode.
"""
from __future__ import annotations

import glob
import io
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture.discovery import newest_capture                     # noqa: E402
from mica.capture.jsonl_ingest import JsonlSource, packet_from_dict   # noqa: E402
from mica.capture.live_ingest import PacketReorderer, ordered_packets  # noqa: E402
from mica.capture.live_stream import stream_moments                   # noqa: E402
from mica.contracts.b2 import ROTATIONS                               # noqa: E402
from mica.contracts.b3 import fuse                                    # noqa: E402
from mica.contracts.serialize import (                                # noqa: E402
    belief_to_dict, evidence2d_to_dict, evidence3d_to_dict, fused_to_dict,
)
from mica.contracts.b1 import GOALS                                   # noqa: E402
from mica.intent.heads_v0 import likelihood                           # noqa: E402
from mica.intent.tracker import TrackerParams, correct, predict, uniform_belief  # noqa: E402
from mica.live_pipeline import LivePipeline, PipelineConfig, RecordSinks, StallMeter  # noqa: E402
from mica.perception.pixel_head import LivePixelHead, TorchEncoders, write_d1_provenance  # noqa: E402
from mica.perception.evidence2d import evidence_stream                # noqa: E402
from mica.perception.evidence3d import DELTA_COMP_WINDOW, Evidence3DStream, build_evidence3d  # noqa: E402
from mica.perception.templates import TEMPLATE_SET_VERSION, TEMPLATES  # noqa: E402
from mica.perception.voxel_replay import (                            # noqa: E402
    TAXONOMY_VERSION, Region, ReplayWorld, SnapshotMonitor, load_snapshot, region_from_manifest,
)
from mica.validation.b0_gate import gate_checks                       # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW = os.path.join(_ROOT, "capture", "raw")
_STATUS_EVERY_MOMENTS = 20      # ~once a second at 20 Hz
_FRESH_WITHIN_S = 5.0           # a recording touched this recently is the active session
_DISCOVER_TIMEOUT_S = 10.0
# The mod drains its frame-writer queue (up to 10 s on a heavy session) BEFORE it
# writes the finalized manifest — a 5 s wait here lost that race on a real session
# and the disk gate read a provisional manifest that finalized moments later.
_FINALIZE_TIMEOUT_S = 15.0


def _flag_value(name: str, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def _snapshot_paths(jsonl: str, session_id: str) -> list[str]:
    snap_dir = os.path.join(os.path.dirname(os.path.abspath(jsonl)), session_id, "snapshots")
    return sorted(glob.glob(os.path.join(snap_dir, "*.json")),
                  key=lambda p: int(os.path.basename(p)[:-5]))


def _sha256(path: str) -> str:
    import hashlib
    if not os.path.exists(path):
        return "missing"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_d2_provenance(jsonl: str, h3d_on: bool = False) -> None:
    """Same sidecar run_d2 writes, pinned immediately — a crash still leaves it."""
    from mica.contracts.goals import TAXONOMY_VERSION as GOAL_TAXONOMY_VERSION

    provenance = {
        "template_set_version": TEMPLATE_SET_VERSION,
        "goal_taxonomy_version": GOAL_TAXONOMY_VERSION,
        "templates": {goal: [t.name for t in ts] for goal, ts in TEMPLATES.items()},
        "rotations": list(ROTATIONS),
        "delta_comp_window": DELTA_COMP_WINDOW,
        "divergence_taxonomy_version": TAXONOMY_VERSION,
        # The learned shape channel, when enabled — pinned exactly as run_d2 pins it.
        "h3d_source": None if not h3d_on else {
            "encoder": "uni3d-b.pt",
            "encoder_sha256": _sha256(os.path.join(_ROOT, "models", "uni3d-b.pt")),
            "cloud_points": 4096,
            "cloud_seed": 0,
        },
    }
    with open(jsonl.replace(".jsonl", ".d2_provenance.json"), "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)


# --------------------------------------------------------------------------- replay


def _expected_lines(session, world: ReplayWorld | None) -> dict[str, list[str]]:
    """What the offline wrappers produce for this session — the equivalence yardstick."""
    params = TrackerParams()
    b1 = list(evidence_stream(session.packets))
    expected = {"evidence2d": [json.dumps(evidence2d_to_dict(r)) for r in b1],
                "evidence3d": [], "fused": [], "belief": []}
    if world is None:
        return expected
    scored = [r for r in b1 if r.scored]
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    b2 = build_evidence3d(world, tuple(scored), events)
    expected["evidence3d"] = [json.dumps(evidence3d_to_dict(r)) for r in b2]
    belief, prev = uniform_belief(), 0
    b2_iter = iter(b2)
    for record in b1:
        if record.scored:
            fused = fuse(record, next(b2_iter))
            dt = max(fused.tick - prev, 1) / 20.0
            prev = fused.tick
            belief = predict(belief, dt, params)
            belief, norm = correct(belief, likelihood(fused, fused.a_hat), params)
            expected["fused"].append(json.dumps(fused_to_dict(fused)))
            expected["belief"].append(json.dumps(belief_to_dict(fused.tick, "correction", belief, norm)))
        else:
            tick = record.tick_range[1]
            drifted = predict(belief, max(tick - prev, 1) / 20.0, params)
            expected["belief"].append(json.dumps(belief_to_dict(tick, "drift", drifted)))
    return expected


def _replay_check(jsonl: str) -> int:
    """Feed the recording through the live pipeline; diff every log line vs offline."""
    session = JsonlSource(jsonl, jsonl.replace(".jsonl", ".manifest.json")).load()
    region = region_from_manifest(session)
    snapshots = _snapshot_paths(jsonl, session.manifest.session_id) if region else []
    base_cells = load_snapshot(snapshots[0])[2] if snapshots else None

    d2 = None
    expected_world = None
    if region is not None and base_cells is not None:
        d2 = Evidence3DStream(ReplayWorld(region, dict(base_cells)))
        expected_world = ReplayWorld(region, dict(base_cells))
    sinks = RecordSinks(evidence2d=io.StringIO(), evidence3d=io.StringIO(),
                        fused=io.StringIO(), belief=io.StringIO())
    pipeline = LivePipeline(PipelineConfig(params=TrackerParams(), sinks=sinks, d2=d2))
    for packet in sorted(session.packets, key=lambda p: p.tick):
        pipeline.on_moment(packet, None)
    pipeline.finish()

    expected = _expected_lines(session, expected_world)
    print(f"replay-equivalence  {os.path.basename(jsonl)}"
          + ("" if d2 is not None else "   (no snapshots: B1 only)"))
    failed = False
    for name in ("evidence2d", "evidence3d", "fused", "belief"):
        got = getattr(sinks, name).getvalue().splitlines()
        want = expected[name]
        same = got == want
        failed = failed or not same
        detail = f"{len(got)} lines" if same else \
            f"MISMATCH at line {next((i + 1 for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)) + 1)} (live {len(got)} vs offline {len(want)} lines)"
        print(f"  [{'PASS' if same else 'FAIL'}] {name}: {detail}")
    print(f"  live path {'==' if not failed else '!='} offline wrappers, record for record")
    return 1 if failed else 0


# ------------------------------------------------------------------------------ live


def _capture_roots() -> list[str]:
    """Where captures may land. The location follows the launcher profile (the
    mod's -Dmica.captureDir JVM arg), so discovery checks every root named in
    MICA_RAW_DIRS (';'-joined), defaulting to the repo's capture/raw."""
    env = os.environ.get("MICA_RAW_DIRS")
    if env:
        return [d.strip() for d in env.split(";") if d.strip()]
    return [_RAW]


def _discover_active_session(forever: bool = False) -> str | None:
    """The recording the mod is appending to right now (mtime keeps moving),
    searched across every capture root. `forever` is the rig's pre-warm mode:
    keep waiting instead of giving up — the player may not have launched yet."""
    deadline = time.monotonic() + _DISCOVER_TIMEOUT_S
    while forever or time.monotonic() < deadline:
        freshest, freshest_mtime = None, -1.0
        for root in _capture_roots():
            jsonl = newest_capture(root)
            if not jsonl or not os.path.exists(jsonl.replace(".jsonl", ".manifest.json")):
                continue
            mtime = os.path.getmtime(jsonl)
            if time.time() - mtime < _FRESH_WITHIN_S and mtime > freshest_mtime:
                freshest, freshest_mtime = jsonl, mtime
        if freshest:
            return freshest
        time.sleep(0.5)
    return None


def _disk_events(jsonl: str) -> list:
    """Every block event the mod has written to the session file so far, paired with
    its tick, in recorded order. The mod writes this file continuously whether or not
    anyone is listening, so at a mid-session attach it already holds the whole story —
    the catch-up source. A torn final line (the mod caught mid-write) ends the read;
    whatever follows arrives over the live socket anyway."""
    events = []
    with open(jsonl, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                packet = packet_from_dict(json.loads(stripped))
            except (ValueError, KeyError):
                break
            for event in packet.server.block_events:
                events.append((packet.tick, event))
    return events


def _seed_structure(jsonl: str, h3d_fn=None):
    """D2's world + the snapshot monitor, seeded from the right base for WHEN the
    attach happens. Returns (d2, monitor, seed_tick) or (None, None, None) when the
    capture declares no snapshots (pre-0.0.3) or none exist yet — D1-only then.
    `h3d_fn` (optional) is the frozen Uni3D shape channel; it rides the stream's
    feature cache, so it computes only when an event changed the region.

    Two cases. Nothing built yet (the normal rig flow): the NEWEST snapshot is the
    base — pre-freeze the mod keeps rewriting its provisional base as the player
    moves. Building already started (a mid-session recovery): the region is frozen
    and the FIRST snapshot is the true pre-build base; _catch_up_structure then
    replays the recorded history on top of it once the first live moment tells us
    where the socket's coverage begins.
    """
    with open(jsonl.replace(".jsonl", ".manifest.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    region_values = meta.get("snapshot_region", ())
    session_id = meta["session_id"]
    deadline = time.monotonic() + _FINALIZE_TIMEOUT_S
    snapshots = _snapshot_paths(jsonl, session_id)
    while not (len(region_values) == 6 and snapshots) and time.monotonic() < deadline:
        time.sleep(0.5)
        with open(jsonl.replace(".jsonl", ".manifest.json"), encoding="utf-8") as handle:
            meta = json.load(handle)
        region_values = meta.get("snapshot_region", ())
        snapshots = _snapshot_paths(jsonl, session_id)
    if len(region_values) != 6 or not snapshots:
        return None, None, None
    # Seed from the base snapshot's OWN frame, not the manifest's: under region v3
    # the manifest always carries the CURRENT (possibly grown) box, while the base
    # file frames the original anchor — growth is adopted snapshot by snapshot.
    base_path = snapshots[0] if _disk_events(jsonl) else snapshots[-1]
    seed_tick, region, cells = load_snapshot(base_path)
    d2 = Evidence3DStream(ReplayWorld(region, dict(cells)), h3d_fn=h3d_fn)
    monitor = SnapshotMonitor(region, dict(cells))
    return d2, monitor, seed_tick


def _catch_up_structure(d2, monitor, jsonl: str, session_id: str, cutoff_tick: int) -> list:
    """A mid-session attach: everything the mod recorded BEFORE the first moment the
    socket delivered is history the live loop will never see (a real session once
    showed 22 of 100 built cells for exactly this reason). Replay that history from
    the on-disk file into both worlds — the feature world takes human events only
    (the A7 rule, inside seed_history), the shadow world takes every event, pausing
    at each pre-attach snapshot to run the replay-vs-snapshot check. Attaching at
    session start makes this a no-op: nothing on disk is older than the first live
    moment, so live output stays identical to the offline wrappers.

    Returns the replayed (tick, event) history so the caller can feed every OTHER
    world consumer too — the D5 gate tracks the human's standing cells and must not
    start blind to pre-attach blocks (D6 review F2)."""
    history = [(tick, event) for tick, event in _disk_events(jsonl) if tick < cutoff_tick]
    later = []
    for path in _snapshot_paths(jsonl, session_id)[1:]:
        snap_tick, snap_region, cells = load_snapshot(path)
        if snap_tick < cutoff_tick:
            later.append((snap_tick, cells, snap_region))
    if not history and not later:
        return history
    # The feature world walks the same interleave as the monitor: widen the frame
    # before a snapshot's events, adopt its base after them (region v3) — so blocks
    # placed in grown territory before the attach read as built, not as escapes.
    applied = 0
    index = 0
    for snap_tick, cells, snap_region in later:
        d2.notice_region(snap_region)
        while index < len(history) and history[index][0] <= snap_tick:
            applied += d2.seed_history([history[index][1]])
            index += 1
        d2.extend_world(snap_region, cells)
    while index < len(history):
        applied += d2.seed_history([history[index][1]])
        index += 1
    monitor.seed_history(history, later)
    print(f"  late attach: {applied} human / {len(history) - applied} agent events"
          f" caught up from disk (recorded before live tick {cutoff_tick})"
          + ("  ** QUARANTINED during catch-up **" if monitor.quarantined else ""))
    return history


def _status_line(pipeline: LivePipeline, reorderer: PacketReorderer,
                 head: LivePixelHead | None) -> str:
    s = pipeline.status()
    line = (f"[tick {s['tick']}] top={s['top_goal']} p={s['p_top_goal']:.2f}"
            f" z1={s['p_z1']:.2f} H={s['entropy']:.2f}"
            f" | rec {s['scored']}s+{s['context']}c corr {s['corrections']}"
            f" | gaps {s['missing_ticks']} drop {reorderer.counts.gap_ticks}")
    if head is not None:
        px = head.status()
        if px["load_error"]:
            line += "  px:ERROR"
        elif not px["loaded"]:
            line += "  px:loading"
        elif px["s_goal_live"]:
            live_top = max(px["s_goal_live"], key=px["s_goal_live"].get)
            line += f"  px:{live_top} ({px['enriched']} enriched)"
    # The model channels' numbers — the moving proof both are producing evidence.
    if s.get("h2d_norm") is not None:
        line += f"  h2d|v|={s['h2d_norm']:.1f}"
    if s.get("s_goal_last"):
        top_index = max(range(len(s["s_goal_last"])), key=lambda i: s["s_goal_last"][i])
        line += f"  s_goal[{GOALS[top_index]}]={s['s_goal_last'][top_index]:.3f}"
    if s.get("h3d_norm") is not None:
        line += f"  h3d|v|={s['h3d_norm']:.1f} ({s['h3d_records']} rec)"
    if s["crop_escapes"]:
        line += f"  !! D2 BLIND: {s['crop_escapes']} events outside region !!"
    if s["quarantined"]:
        line += "  ** QUARANTINED **"
    return line


_MAX_STATUS_CELLS = 512


def _status_cell_sample(built, cap: int = _MAX_STATUS_CELLS) -> list[list[int]]:
    """The built cells that fit one status write. Small builds go whole; a build
    past the cap sends a FRESH RANDOM sample each write instead of the old sorted
    prefix — the prefix made the agent's coverage patrol permanently blind to
    every cell after the 512th, so large builds never got scanned beyond their
    low-coordinate corner. Random per-write sampling lets the agent accumulate
    the whole set over a few ~1 Hz writes, however big the build grows."""
    cells = [list(cell) for cell in sorted(built)]
    if len(cells) <= cap:
        return cells
    return random.sample(cells, cap)


def _write_live_status(path: str, session_id: str, pipeline: LivePipeline,
                       head: LivePixelHead | None, d2=None, gate: dict | None = None) -> None:
    """The agent bridge: one small JSON snapshot, rewritten atomically ~1 Hz.
    The in-game agent (and anything else local) polls this instead of touching
    the single-consumer live socket. A stale timestamp means the pipeline is gone.
    Carries the player-built cells too — the live source of the Uni3D point cloud,
    which FlowViz renders as the agent's structure view."""
    status = {"ts": time.time(), "session_id": session_id, **pipeline.status()}
    if gate is not None:
        status["gate"] = gate                # the D5 gate block the agent renders
    if head is not None:
        status["pixels"] = head.status()
    if d2 is not None:
        built = d2._world.built()
        status["built_count"] = len(built)   # the uncapped truth; the list is size-capped
        status["built_cells"] = _status_cell_sample(built)
        region = d2._world.region
        status["region"] = [region.x0, region.y0, region.z0, region.x1, region.y1, region.z1]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(status, handle)
    # Windows: os.replace fails with WinError 5 when a READER (FlowViz polls this
    # file; the agent reads it too) holds the destination without delete-sharing
    # at the exact swap instant. This is a DISPLAY file — a failed swap must never
    # reach the stream loop's OSError handler, which reads it as "live stream
    # lost" and ends live processing (it did, 2026-07-19: two sessions' point
    # clouds froze mid-play). Retry briefly, then skip this write; the next one
    # is a second away.
    for attempt in range(3):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(0.02)
    try:
        os.remove(tmp)                        # give up on THIS write, not the session
    except OSError:
        pass


def _load_h3d(h3d_on: bool):
    """The frozen Uni3D shape channel (h3d): with it on, every B2 record carries the
    build's shape embedding, so the fused evidence holds BOTH model channels live —
    the exact input shape the Phase-E adapter (and later the action layer) consumes."""
    if not h3d_on:
        return None
    from mica.perception.shape3d import Uni3DShapeHead, assets_ready

    ready, missing = assets_ready()
    if not ready:
        print(f"  !! --h3d needs the Uni3D assets ({missing}) — run scripts/setup_uni3d.py;"
              " continuing WITHOUT the shape channel")
        return None
    print("  h3d: loading frozen Uni3D-B on GPU ...")
    return Uni3DShapeHead().h3d


def _make_head(pixels_on: bool):
    """The stage-2 pixel head (D6 §8): VPT + MineCLIP on a background GPU worker.
    Loading takes a while — records released before it finishes pass unchanged."""
    if not pixels_on:
        return None
    head = LivePixelHead(TorchEncoders(), stride=max(1, int(_flag_value("--sgoal-stride", 1))))
    print(f"  pixel head: loading frozen VPT + MineCLIP in the background"
          f" (s_goal clip stride {head.stride})")
    if "--wait-models" in sys.argv:
        # Rehearsals wait so every scored record can be enriched; REAL live never
        # should (delaying attach drops moments off the mod's bounded queue).
        print("  --wait-models: holding the attach until the pixel head is loaded ...")
        deadline = time.monotonic() + 600
        while (not head.status()["loaded"] and head.load_error is None
               and time.monotonic() < deadline):
            time.sleep(1.0)
        print(f"  pixel head {'loaded' if head.status()['loaded'] else 'NOT loaded'}"
              f" — attaching")
    return head


# Everything one live run writes next to the session. A second attach to the same
# session must start these files fresh — and must not destroy the first run's.
_RUN_OUTPUT_SUFFIXES = (".evidence2d.jsonl", ".evidence3d.jsonl", ".fused.jsonl",
                        ".belief.jsonl", ".gate_trace.jsonl", ".live_run.json",
                        ".materials_report.json", ".d1_provenance.json",
                        ".d2_provenance.json")


def _bank_previous_run(base: str) -> None:
    """Move a previous run's outputs aside before this run opens its own.

    Without this, a re-attach to the same session mixed two runs in one set of
    files — worse, inconsistently: the belief log was truncated (mode "w") while
    the gate trace appended, so the first run's gate rows pointed at belief
    snapshots that no longer existed and the trace could not explain its own
    decisions (found by scripts/audit_gate_trace.py on two real sessions).
    Now every earlier run keeps its complete, matching set of files under
    previous_runs/run-NN/, and the new run starts clean."""
    present = [base + suffix for suffix in _RUN_OUTPUT_SUFFIXES
               if os.path.exists(base + suffix)]
    if not present:
        return
    session_dir = os.path.dirname(os.path.abspath(base))
    number = 1
    while os.path.isdir(os.path.join(session_dir, "previous_runs", f"run-{number:02d}")):
        number += 1
    dest = os.path.join(session_dir, "previous_runs", f"run-{number:02d}")
    os.makedirs(dest)
    for path in present:
        shutil.move(path, os.path.join(dest, os.path.basename(path)))
    print(f"  banked a previous run's {len(present)} output file(s) -> "
          f"previous_runs/run-{number:02d}/ (runs never share files)")


def _live(host: str, port: int, session_arg: str | None = None) -> int:
    pixels_on = "--pixels" in sys.argv or os.environ.get("MICA_PIXELS") == "1"
    h3d_on = "--h3d" in sys.argv or os.environ.get("MICA_H3D") == "1"
    wait_session = "--wait-session" in sys.argv
    h3d_fn = None
    head = None
    if wait_session:
        # Pre-warm (the rig starts this runner before the player is in a world):
        # both GPU models load NOW, during the launcher/menu screen, so the very
        # first scored records of the session get enriched. Without this, loading
        # began at attach and a real session spent its opening ~40 s "px:loading"
        # with every early record passing unenriched.
        h3d_fn = _load_h3d(h3d_on)
        head = _make_head(pixels_on)
        print("  models pre-warmed; waiting for a session (enter a world to begin) ...")
    jsonl = session_arg or _discover_active_session(forever=wait_session)
    if jsonl is None:
        print("no active capture found: enter the world first (the mod must be recording),"
              " or pass a session.jsonl for replay mode")
        return 1
    base = jsonl[: -len(".jsonl")]
    with open(jsonl.replace(".jsonl", ".manifest.json"), encoding="utf-8") as handle:
        session_id = json.load(handle)["session_id"]
    print(f"live session: {session_id}")
    _bank_previous_run(base)
    if wait_session:
        # The session file exists from game launch, but the region only exists once
        # the player is actually IN a world — hold the attach until then, or the
        # runner would lock itself into D1-only mode against an empty session.
        while True:
            with open(jsonl.replace(".jsonl", ".manifest.json"), encoding="utf-8") as handle:
                if len(json.load(handle).get("snapshot_region", ())) == 6:
                    break
            time.sleep(0.5)

    if not wait_session:
        h3d_fn = _load_h3d(h3d_on)
    d2, monitor, seed_tick = _seed_structure(jsonl, h3d_fn=h3d_fn)
    if d2 is None:
        print("  no snapshot region/files: D1-only mode (behavior log, no structure/belief)"
              + ("  (h3d head loaded but idle — no structure stream)" if h3d_fn else ""))
    else:
        _write_d2_provenance(jsonl, h3d_on=h3d_fn is not None)
        print(f"  structure seeded from snapshot @{seed_tick}; belief live"
              + ("; h3d shape channel ON" if h3d_fn else ""))

    sink_paths = {name: f"{base}.{name}.jsonl" for name in
                  ("evidence2d", "evidence3d", "fused", "belief")}
    handles = {name: open(path, "w", encoding="utf-8") for name, path in sink_paths.items()
               if d2 is not None or name == "evidence2d"}
    print("  belief tail:  Get-Content \"" + sink_paths["belief"] + "\" -Wait")

    if not wait_session:
        head = _make_head(pixels_on)

    reorderer = PacketReorderer()
    # OPT-IN trained heads for the live belief (the demo config; v0 stays the
    # default so the golden-equivalence proof is untouched). Same rebind pattern
    # as run_tracker: swap the module-global likelihood the correct step resolves.
    params = TrackerParams()
    if _flag_value("--heads", None) == "v1":
        from mica.intent import heads_v1
        import mica.live_pipeline as _lp
        _lp.likelihood = heads_v1.likelihood
        params = heads_v1.tracker_params()
        print("  belief heads: TRAINED v1 (opt-in) with their jointly-fitted knobs")
    pipeline = LivePipeline(PipelineConfig(
        params=params,
        sinks=RecordSinks(**{name: handles.get(name) for name in sink_paths}),
        d2=d2, monitor=monitor, pixel_head=head))
    gate_runner, gate_block = None, None
    if d2 is not None and "--no-gate" not in sys.argv:
        from mica.gate.live_loop import AsyncGateRunner, LiveGateRunner, gate_ready
        if gate_ready():
            # preload: the decoder loads NOW (before the socket attaches), never on
            # the pipeline thread mid-session — a lazy first-read load would stall
            # past the mod's drop-oldest buffer and open the session with a gap burst.
            # --gather opts into the D5 §4 amendment (2026-07-10): the GATHER state
            # can fire when materials are missing for the declared (or confidently
            # inferred) target. Off by default so banked behavior is bit-unchanged.
            gather = "--gather" in sys.argv
            # --place lifts the §9 demo pin (amendment 2026-07-12, user decision):
            # the gate may authorize ONE reversible placement per read, licensed by
            # conf >= theta_place ONLY (the declared route is retired for live
            # sessions, 2026-07-13 — live behavior comes from inference alone).
            # Everything else — no flag, replays, counterfactual — stays demo.
            place = "--place" in sys.argv
            # --consent-place arms the CONSENT route only (D5 §10, 2026-07-19):
            # the human's chat "yes" after a voiced suggestion licenses ONE
            # placement of that proposal, whatever the confidence. It does NOT
            # lift the §9 pin — the FSM stays demo, theta_place stays untouched.
            consent_place = "--consent-place" in sys.argv
            gate_runner = LiveGateRunner(f"{base}.gate_trace.jsonl", params,
                                         demo=not place, preload=True,
                                         session_id=session_id,
                                         gather=gather, place=place,
                                         consent_place=consent_place)
            if place:
                print("  D5 gate: LIVE — !! PLACEMENT ENABLED !! (§9 amendment: "
                      "confidence route only, 1 block per read, reversible only; "
                      "say 'stop' in chat to halt the agent)")
            elif consent_place:
                print("  D5 gate: LIVE — consent placement armed (D5 §10: your "
                      "chat 'yes' after a suggestion places that ONE block; "
                      "say 'stop' to halt the agent)")
            else:
                print("  D5 gate: LIVE, decoder pre-warmed (demo config — observe/"
                      "suggest/preview only" + ("; GATHER enabled" if gather else ""))
            print("    trace -> " + os.path.basename(base) + ".gate_trace.jsonl")
        else:
            print("  D5 gate: OFF (no decoder/gate freeze on disk)")
    status_path = os.path.join(os.path.dirname(os.path.abspath(jsonl)), "live_status.json")
    # The runner's own two heavy steps, measured like the pipeline measures its own
    # (D6 review F8): together they say where a gapped session's time actually went.
    gate_stall, snap_stall = StallMeter(), StallMeter()
    if gate_runner is not None:
        # The proof-grade fix (2026-07-13): gate reads cost 200-360 ms mean and
        # used to run INSIDE the socket-draining loop — the measured source of
        # the gap ticks. The worker computes off-thread; the loop below only
        # hands over inputs and picks up the newest finished block. The meter
        # still records every read, now from the worker's own clock.
        gate_runner = AsyncGateRunner(gate_runner, on_timing=gate_stall.add)
    seen_snapshots = set(_snapshot_paths(jsonl, session_id))
    # Snapshots discovered on disk but not yet judged. A file appearing does NOT
    # mean the pipeline has consumed every moment up to its tick — when processing
    # lags the feed by more than the mod's 40-quiet-tick window, comparing at file
    # arrival reads a world missing in-flight events and false-quarantines (D6
    # review F12; reproduced at 10x rehearsal pace). The safe rule: judge snapshot
    # @T only after a moment with tick >= T has been processed — the monitor holds
    # newer events by tick, so judging LATER is always exact.
    pending_snaps: list[tuple[int, Region, dict]] = []
    ended = "eof"
    moments_seen = 0
    escapes_warned_at = 0.0
    pixel_error_shown = False
    caught_up = False
    try:
        print(f"  connecting to {host}:{port} ...")
        for packet, frame in ordered_packets(stream_moments(host, port), reorderer):
            if not caught_up:
                # The first delivered moment defines where live coverage begins;
                # anything recorded before it is caught up from the disk copy.
                caught_up = True
                if d2 is not None:
                    history = _catch_up_structure(d2, monitor, jsonl, session_id,
                                                  packet.tick)
                    if gate_runner is not None:
                        # The gate is the third world consumer of catch-up: its
                        # reversibility read tracks the human's standing cells and
                        # must know about pre-attach blocks too (D6 review F2).
                        gate_runner.ingest_events([event for _, event in history])
            if head is not None and frame is not None and packet.client.pov_frame is not None:
                head.observe(packet.tick, packet.client.pov_frame, frame)
            for problem in pipeline.on_moment(packet, frame):
                print(f"  ! {problem}")
            if gate_runner is not None:
                gate_runner.ingest_events(packet.server.block_events)
            moments_seen += 1
            if moments_seen % _STATUS_EVERY_MOMENTS == 0:
                if head is not None and head.load_error and not pixel_error_shown:
                    pixel_error_shown = True
                    print(f"  !! pixel head failed — running symbolic only: {head.load_error}")
                # The manifest's region can differ from ours two ways. PRE-FREEZE
                # (no events consumed): the provisional anchor MOVED — re-seed while
                # that is still exact (the 110234/193059 false quarantine). POST-
                # FREEZE (region v3): the frame GREW — widen both worlds' escape
                # test now; the adopted base arrives with the growth snapshot file.
                if d2 is not None:
                    with open(jsonl.replace(".jsonl", ".manifest.json"), encoding="utf-8") as handle:
                        region_now = tuple(json.load(handle).get("snapshot_region", ()))
                    if len(region_now) == 6:
                        new_region = Region(*region_now)
                        current = d2._world.region
                        if new_region != current:
                            if pipeline.events_consumed == 0 and not new_region.covers(current):
                                print(f"  region moved (provisional anchor) — re-seeding structure"
                                      f" at {list(region_now)}")
                                d2, monitor, seed_tick = _seed_structure(jsonl, h3d_fn=h3d_fn)
                                if d2 is not None:
                                    pipeline.reseed_structure(d2, monitor)
                                    seen_snapshots = set(_snapshot_paths(jsonl, session_id))
                                    pending_snaps = []   # the fresh monitor starts over
                            else:
                                print(f"  region grew — frame now {list(region_now)}")
                                pipeline.notice_region(new_region)
                if gate_runner is not None:
                    # Hand the worker the freshest inputs and take the newest
                    # FINISHED block — the first status write of a session may
                    # carry no gate block yet (the read lands ~a second later).
                    gate_runner.request(pipeline.belief, pipeline.last_fused,
                                        pipeline.status())
                    gate_block = gate_runner.latest()
                _write_live_status(status_path, session_id, pipeline, head, d2,
                                   gate=gate_block)
                # The failure that must never be quiet again: the build has left the
                # capture region, so every structure feature is reading zero. Shout
                # once immediately, then every ~30 s while it persists.
                escapes = pipeline.status()["crop_escapes"] if d2 is not None else 0
                if escapes and time.monotonic() - escapes_warned_at > 30.0:
                    escapes_warned_at = time.monotonic()
                    with open(jsonl.replace(".jsonl", ".manifest.json"), encoding="utf-8") as handle:
                        region = json.load(handle).get("snapshot_region")
                    print(f"  !!!! BUILD OUTSIDE CAPTURE REGION — D2 IS BLIND "
                          f"({escapes} events escaped; region {region}). Structure "
                          f"features read zero; the session will quarantine. The region "
                          f"lives per GAME LAUNCH: quit Minecraft fully, relaunch, and "
                          f"place the first block at the build site. !!!!")
                line = _status_line(pipeline, reorderer, head)
                if gate_block is not None:
                    line += f"  gate:{gate_block['state']}"
                worst = max(pipeline.stalls["moment"].worst_ms, gate_stall.worst_ms,
                            snap_stall.worst_ms)
                if worst:
                    line += f"  worst-stall {worst:.0f}ms"
                print(line)
                for path in _snapshot_paths(jsonl, session_id):
                    if path not in seen_snapshots:
                        seen_snapshots.add(path)
                        snap_tick, snap_region, cells = load_snapshot(path)
                        pending_snaps.append((snap_tick, snap_region, cells))
                ready = [snap for snap in pending_snaps
                         if pipeline.last_tick is not None and snap[0] <= pipeline.last_tick]
                pending_snaps = [snap for snap in pending_snaps if snap not in ready]
                for snap_tick, snap_region, cells in sorted(ready, key=lambda snap: snap[0]):
                    snap_started = time.perf_counter()
                    found = pipeline.on_snapshot(snap_tick, cells, snap_region)
                    snap_stall.add((time.perf_counter() - snap_started) * 1000.0)
                    print(f"  snapshot @{snap_tick}: {len(found)} divergence(s)"
                          + ("  ** QUARANTINED **" if monitor and monitor.quarantined else ""))
    except KeyboardInterrupt:
        ended = "interrupt"
        print("\n  stopped by hand; closing out")
    except (ConnectionError, OSError) as error:
        ended = "stream_lost"
        print(f"\n  live stream lost ({error}); the disk recording remains complete")

    # Snapshots still waiting for the stream to reach their tick: the stream is
    # over, every delivered moment has been processed, so they are judgeable now
    # — before finish(), so the monitor still holds its pending events.
    for snap_tick, snap_region, cells in sorted(pending_snaps, key=lambda snap: snap[0]):
        found = pipeline.on_snapshot(snap_tick, cells, snap_region)
        print(f"  snapshot @{snap_tick} (end of stream): {len(found)} divergence(s)")
    summary = pipeline.finish()
    for handle in handles.values():
        handle.close()
    if gate_runner is not None:
        gate_runner.close()
    _write_live_status(status_path, session_id, pipeline, head, d2,
                       gate=gate_block)   # final ts marks the end
    if head is not None:
        head.close()
        if head.enriched:
            # The evidence file carries GPU pixel channels: pin what produced them,
            # exactly as run_d1 --pixels does.
            print(f"  pixel provenance: {os.path.basename(write_d1_provenance(sink_paths['evidence2d'], head.stride))}")
            print(f"  pixels: {head.enriched} records enriched, "
                  f"{head.skipped_not_ready} before load, {head.deadline_missed} past deadline")

    # The mod finalizes the manifest on a clean quit; give it a moment, then run the
    # full gate against the disk copy — the authoritative artifact.
    deadline = time.monotonic() + _FINALIZE_TIMEOUT_S
    while time.monotonic() < deadline:
        with open(jsonl.replace(".jsonl", ".manifest.json"), encoding="utf-8") as handle:
            if int(json.load(handle).get("declared_event_count", -1)) >= 0:
                break
        time.sleep(0.5)
    disk = JsonlSource(jsonl, jsonl.replace(".jsonl", ".manifest.json")).load()
    checks = gate_checks(disk, os.path.dirname(os.path.abspath(jsonl)))
    gate_pass = all(ok for _, ok, _ in checks)
    print(f"  [{'PASS' if gate_pass else 'FAIL'}] B0 gate on the disk recording")
    for name, ok, detail in checks:
        if not ok:
            print(f"    [FAIL] {name}  {detail}")

    run_report = {
        "session_id": session_id,
        "ended": ended,
        "moments": moments_seen,
        "ingest": reorderer.counts.to_dict(),
        **summary,
        "gate_on_disk_pass": gate_pass,
    }
    # The runner's own meters join the pipeline's (summary already carries those).
    run_report["timings"]["gate_read"] = gate_stall.to_dict()
    run_report["timings"]["snapshot_check"] = snap_stall.to_dict()
    # The D6 §9 verdict, persisted where the audit reads it — not just printed.
    proof_grade = (run_report["gate_live"]["clean"] and reorderer.counts.gap_ticks == 0
                   and not run_report["quarantined"] and gate_pass)
    run_report["proof_grade"] = proof_grade
    report_path = f"{base}.live_run.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(run_report, handle, indent=2)
    print(f"  run summary -> {os.path.basename(report_path)}"
          f"  (proof-grade: {'yes' if proof_grade else 'NO — see report'})")
    ok = ended == "eof" and not run_report["quarantined"] and gate_pass
    return 0 if ok else 1


# Flags that CONSUME the next argument — their values are never session paths.
_VALUE_FLAGS = ("--host", "--port", "--sgoal-stride", "--session", "--heads")


def _positionals(argv: list[str]) -> list[str]:
    return [a for i, a in enumerate(argv[1:], 1)
            if not a.startswith("--") and argv[i - 1] not in _VALUE_FLAGS]


def main() -> int:
    positional = _positionals(sys.argv)
    if positional and "--gate" in sys.argv:
        # Gate replay: exercise the LIVE runner over a recorded session's banked
        # evidence — the pre-demo proof that the in-world loop behaves.
        from mica.gate.live_loop import replay_gate
        return replay_gate(positional[0])   # always the calibrated v1 replay
    if positional:
        return _replay_check(positional[0])
    return _live(str(_flag_value("--host", "127.0.0.1")), int(_flag_value("--port", 25567)),
                 session_arg=_flag_value("--session", None))


if __name__ == "__main__":
    raise SystemExit(main())
