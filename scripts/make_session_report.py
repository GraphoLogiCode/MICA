"""One command that banks a finished session's story as graphs — the visual proof
that the system worked, drawn only from the recorded artifacts.

    python scripts/make_session_report.py <session.jsonl> [--scan <scan.jsonl>]

WHAT A FULLY SAVED SESSION IS (the banked artifact set, all next to the capture):
  raw (mod-written, never touched):   <sid>.jsonl, <sid>.manifest.json,
                                      <sid>/frames/*.png, <sid>/snapshots/*.json
  derived evidence (offline reruns):  .evidence2d.jsonl (D1 + pixels),
                                      .evidence3d.jsonl (D2 + h3d), .fused.jsonl,
                                      .belief.jsonl, .live_run.json,
                                      .voxel_replay_report.json, provenance sidecars
  agent-side sensor:                  agent-<NAME>.scan.jsonl (rotated per launch)
  this report:                        capture/reports/<sid>/  — the graphs below,
                                      summary.json, and a copy of the scan file used

GRAPHS (each skipped gracefully when its source artifact is absent):
  belief.png     the five category marginals + entropy over the session
  structure.png  per-goal completion + built-cell growth (the D2 structure read)
  channels.png   the model channels' live numbers: h2d/h3d norms, the five s_goal cosines
  actions.png    what the player did: scored macro-action timeline + counts
  scan.png       agent-scan coverage curve (+ h3d_scan convergence when measured)
  clouds.png     the exact built cloud vs the scanned portion, side by side
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402

from mica.capture.jsonl_ingest import JsonlSource                       # noqa: E402
from mica.contracts.b0 import is_agent_actor                            # noqa: E402
from mica.contracts.b1 import GOALS                                     # noqa: E402
from mica.perception.agent_scan import accumulate, load_scan, scanned_build_cells  # noqa: E402

_GOAL_COLORS = {"habitation": "#e8a33d", "infrastructure": "#4c8dd6",
                "production": "#4fae6e", "defense": "#d65f5f", "decorative": "#a07dd0"}


def _read_jsonl(path: str) -> list[dict]:
    """Every parseable line. Bad lines are SKIPPED, not fatal: a report renders
    whatever is recoverable (a crash mid-write, or an aborted earlier attach,
    may leave torn lines anywhere in a log — the good lines still tell the story)."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip().lstrip("\x00").strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except ValueError:
                continue
    return records


def _norm(vector) -> float | None:
    return round(math.sqrt(sum(v * v for v in vector)), 3) if vector else None


def _built_set(jsonl: str, session):
    """The exact built set (pre-build base + the session's human events, A7)."""
    import glob as _glob

    from mica.perception.voxel_replay import ReplayWorld, load_snapshot, region_from_manifest

    region = region_from_manifest(session)
    snap_dir = os.path.join(os.path.dirname(os.path.abspath(jsonl)),
                            session.manifest.session_id, "snapshots")
    snapshots = sorted(_glob.glob(os.path.join(snap_dir, "*.json")),
                       key=lambda p: int(os.path.basename(p)[:-5]))
    if region is None or not snapshots:
        return {}
    _, _, base = load_snapshot(snapshots[0])
    world = ReplayWorld(region, dict(base))
    for packet in sorted(session.packets, key=lambda p: p.tick):
        for event in packet.server.block_events:
            if not is_agent_actor(event.actor):
                world.apply(event)
    return world.built()


def _save(figure, out_dir: str, name: str, made: list[str]) -> None:
    figure.tight_layout()
    path = os.path.join(out_dir, name)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    made.append(name)


def _belief_graph(belief_lines, out_dir, made) -> None:
    if not belief_lines:
        return
    ticks = [line["tick"] for line in belief_lines]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for goal in GOALS:
        series = [sum(p for key, p in line["belief"].items() if key.startswith(goal + "|"))
                  for line in belief_lines]
        axis.plot(ticks, series, label=goal, color=_GOAL_COLORS.get(goal), linewidth=1.2)
    axis.set_xlabel("tick")
    axis.set_ylabel("P(category)")
    axis.set_title("Belief over the session (category marginals)")
    axis.legend(loc="upper left", fontsize=8)
    twin = axis.twinx()
    twin.plot(ticks, [line["entropy"] for line in belief_lines],
              color="#888888", linewidth=0.8, linestyle="--")
    twin.set_ylabel("entropy (dashed)")
    _save(figure, out_dir, "belief.png", made)


def _structure_graph(b2_lines, out_dir, made) -> None:
    if not b2_lines:
        return
    ticks = [line["tick"] for line in b2_lines]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for goal in GOALS:
        axis.plot(ticks, [line["per_goal"][goal]["comp"] for line in b2_lines],
                  label=f"{goal} comp", color=_GOAL_COLORS.get(goal), linewidth=1.2)
    axis.set_xlabel("tick")
    axis.set_ylabel("completion (comp)")
    axis.set_title("D2 structure read: per-goal completion + build growth")
    axis.legend(loc="upper left", fontsize=8)
    twin = axis.twinx()
    twin.plot(ticks, [line["global"]["built_count"] for line in b2_lines],
              color="#444444", linewidth=1.5, linestyle=":")
    twin.set_ylabel("built cells (dotted)")
    _save(figure, out_dir, "structure.png", made)


def _channels_graph(b1_lines, b2_lines, out_dir, made) -> None:
    scored = [line for line in b1_lines if line.get("scored")]
    if not scored and not b2_lines:
        return
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    h2d = [(line["tick_range"][1], _norm(line["h2d"])) for line in scored if line.get("h2d")]
    if h2d:
        top.plot([t for t, _ in h2d], [n for _, n in h2d], color="#4c8dd6",
                 linewidth=1.0, label="h2d |v| (VPT)")
    h3d = [(line["tick"], line["h3d"]) for line in b2_lines if line.get("h3d")]
    if h3d:
        # Uni3D embeddings are unit-normalized, so their L2 norm is 1.0 by
        # construction — a norm plot reads "flat" no matter what the shape does.
        # What actually moves is the DIRECTION: how far each embedding rotated
        # since the previous correction (drift), and how close it already points
        # to the finished build's shape (convergence).
        def _cos(a, b):
            return sum(x * y for x, y in zip(a, b))

        twin3 = top.twinx()
        if len(h3d) > 1:
            twin3.plot([t for t, _ in h3d[1:]],
                       [_cos(h3d[i][1], h3d[i - 1][1]) for i in range(1, len(h3d))],
                       color="#4fae6e", linewidth=1.0, label="h3d drift cos(k, k−1)")
        final = h3d[-1][1]
        twin3.plot([t for t, _ in h3d], [_cos(v, final) for _, v in h3d],
                   color="#2e7d4f", linewidth=1.0, linestyle="--",
                   label="h3d convergence cos(k, final)")
        twin3.set_ylabel("h3d direction cosine (unit-norm embedding)")
        twin3.legend(loc="lower right", fontsize=8)
    top.set_ylabel("h2d L2 norm")
    top.set_title("Model channels producing evidence")
    top.legend(loc="upper left", fontsize=8)
    sgoal = [(line["tick_range"][1], line["s_goal"]) for line in scored if line.get("s_goal")]
    if sgoal:
        for index, goal in enumerate(GOALS):
            bottom.plot([t for t, _ in sgoal], [s[index] for _, s in sgoal],
                        label=goal, color=_GOAL_COLORS.get(goal), linewidth=1.0)
    bottom.set_xlabel("tick")
    bottom.set_ylabel("s_goal raw cosine")
    bottom.legend(loc="upper left", fontsize=8)
    _save(figure, out_dir, "channels.png", made)


def _actions_graph(b1_lines, out_dir, made) -> None:
    scored = [line for line in b1_lines if line.get("scored")]
    if not scored:
        return
    actions = sorted({line["a_hat"] for line in scored})
    lanes = {action: index for index, action in enumerate(actions)}
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 5),
                                         gridspec_kw={"height_ratios": [2, 1]})
    top.scatter([line["tick_range"][1] for line in scored],
                [lanes[line["a_hat"]] for line in scored], s=4, color="#4c8dd6")
    top.set_yticks(list(lanes.values()), list(lanes.keys()), fontsize=8)
    top.set_title("Scored macro-actions over the session")
    top.set_xlabel("tick")
    counts = {action: sum(1 for line in scored if line["a_hat"] == action) for action in actions}
    bottom.bar(list(counts.keys()), list(counts.values()), color="#888888")
    bottom.set_ylabel("records")
    _save(figure, out_dir, "actions.png", made)


def _scan_graph(scan_report, out_dir, made) -> None:
    if not scan_report or not scan_report.get("coverage_curve"):
        return
    curve = scan_report["coverage_curve"]
    figure, axis = plt.subplots(figsize=(10, 4))
    ticks = [point["tick"] if point["tick"] is not None else index
             for index, point in enumerate(curve)]
    total = max(curve[-1]["built_total"], 1)
    axis.plot(ticks, [point["built_seen"] / total for point in curve],
              color="#4fae6e", linewidth=1.5, label="build coverage")
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("tick")
    axis.set_ylabel("scanned / built")
    axis.set_title("Agent scan: how much of the build the agent has seen")
    convergence = (scan_report.get("h3d_scan") or {}).get("convergence") or []
    # the twin axis SHARES x (ticks), so each sample is placed at the tick of the
    # sweep it was measured after — not at its coverage fraction
    points = [(ticks[min(sample["sweep"], len(ticks) - 1)], sample["cosine_vs_exact"])
              for sample in convergence if sample.get("cosine_vs_exact") is not None]
    if points:
        twin = axis.twinx()
        twin.plot([t for t, _ in points], [v for _, v in points], marker="o",
                  color="#a07dd0", linewidth=1.0, linestyle="--")
        twin.set_ylabel("cosine(h3d_scan, exact h3d) (dashed)")
    axis.legend(loc="lower right", fontsize=8)
    _save(figure, out_dir, "scan.png", made)


def _clouds_graph(built, scanned, out_dir, made) -> None:
    """Both clouds in the ONE fixed display frame (pinned): x and y are the two
    horizontal world axes, z is height above the DETECTED GROUND — the built
    set's lowest layer, the same rule the D2 templates use. Lit voxel cubes with
    edge lines instead of loose dots, identical orientation and extents on both
    subplots so the exact and scanned clouds compare face to face. Geometry is
    untouched: only observed cells are drawn, nothing interpolated."""
    if not built:
        return
    import numpy as np

    ground = min(c[1] for c in built)
    xs_all = [c[0] for c in built]
    zs_all = [c[2] for c in built]
    x0, z0 = min(xs_all), min(zs_all)
    nx = max(xs_all) - x0 + 1
    nz = max(zs_all) - z0 + 1
    ny = max(c[1] for c in built) - ground + 1

    def _grid(cells):
        # cells -> a dense occupancy grid in the display frame; bounds come from
        # the BUILT set so both subplots share extents exactly.
        grid = np.zeros((nx, nz, ny), dtype=bool)
        colors = np.empty((nx, nz, ny), dtype=object)
        for (cx, cy, cz) in cells:
            ix, iy, iz = cx - x0, cz - z0, cy - ground
            if 0 <= ix < nx and 0 <= iy < nz and 0 <= iz < ny:
                grid[ix, iy, iz] = True
                colors[ix, iy, iz] = plt.cm.viridis(iz / max(ny - 1, 1))
        return grid, colors

    figure = plt.figure(figsize=(10, 5))
    for index, (title, cells) in enumerate(
            (("exact built cloud (D2)", built),
             ("scanned portion (agent's eyes)", scanned or {}))):
        axis = figure.add_subplot(1, 2, index + 1, projection="3d")
        if cells:
            grid, colors = _grid(cells)
            axis.voxels(grid, facecolors=colors, edgecolor=(0, 0, 0, 0.3),
                        linewidth=0.3, shade=True)
        axis.set_title(f"{title} — {len(cells)} cells", fontsize=9)
        axis.set_xlabel("x", fontsize=8)
        axis.set_ylabel("y", fontsize=8)
        axis.set_zlabel("z (height above ground)", fontsize=7)
        axis.tick_params(labelsize=6)
        axis.view_init(elev=28, azim=-60)   # one orientation, both subplots
    _save(figure, out_dir, "clouds.png", made)


def main() -> int:
    positional = [a for i, a in enumerate(sys.argv[1:], 1)
                  if not a.startswith("--") and sys.argv[i - 1] != "--scan"]
    if not positional:
        print(__doc__)
        return 1
    jsonl = positional[0]
    if not os.path.exists(jsonl) and not jsonl.endswith(".jsonl"):
        # A bare session id: let the one layout-aware resolver find it (dated
        # dir, or the legacy flat root — session_store owns that knowledge).
        from mica.capture.session_store import session_dir
        sid_arg = jsonl
        jsonl = os.path.join(session_dir(sid_arg), f"{sid_arg}.jsonl")
    base = jsonl[: -len(".jsonl")]
    session = JsonlSource(jsonl, base + ".manifest.json").load()
    sid = session.manifest.session_id
    # Reports live in `reports/` NEXT TO the raw tree the session sits under —
    # found by walking up to the nearest `raw` ancestor, so the flat root, the
    # dated `raw/<date>/<sid>/` layout, and test fixtures all resolve the same
    # way (a fixed two-dirs-up climb landed inside the date folder). A session
    # outside any raw tree banks into the repo's capture/reports.
    ancestor = os.path.dirname(os.path.abspath(jsonl))
    reports_root = os.path.join(_ROOT, "capture", "reports")
    while os.path.dirname(ancestor) != ancestor:
        if os.path.basename(ancestor) == "raw":
            reports_root = os.path.join(os.path.dirname(ancestor), "reports")
            break
        ancestor = os.path.dirname(ancestor)
    out_dir = os.path.join(reports_root, sid)
    os.makedirs(out_dir, exist_ok=True)

    b1_lines = _read_jsonl(base + ".evidence2d.jsonl")
    b2_lines = _read_jsonl(base + ".evidence3d.jsonl")
    belief_lines = _read_jsonl(base + ".belief.jsonl")
    scan_report = None
    if os.path.exists(base + ".agent_scan_report.json"):
        scan_report = json.load(open(base + ".agent_scan_report.json", encoding="utf-8"))

    built = _built_set(jsonl, session)
    scanned = None
    scan_arg = None
    if "--scan" in sys.argv:
        scan_arg = sys.argv[sys.argv.index("--scan") + 1]
    else:
        import glob as _glob
        # Agent scan files are root aggregates (the agent writes next to where the
        # MOD records, the flat raw root) — a relocated session's dir holds none,
        # so search both places. Pair by WALLCLOCK, not by newest: the freshest
        # file can already belong to the NEXT session's world (a report once drew
        # "0 scanned cells" for exactly that reason). Of the scans that started
        # after this session began, the earliest is this session's own agent.
        here = os.path.dirname(os.path.abspath(jsonl))
        root = os.path.join(_ROOT, "capture", "raw")
        candidates = sorted(_glob.glob(os.path.join(here, "agent-*.scan*.jsonl")),
                            key=os.path.getmtime)
        if os.path.abspath(here) != os.path.abspath(root):
            # scans next to the session always win; the repo root is the fallback
            candidates += sorted(_glob.glob(os.path.join(root, "agent-*.scan*.jsonl")),
                                 key=os.path.getmtime)
        start_s = session.manifest.session_start_ms / 1000.0
        scan_arg = None
        for candidate in candidates:
            sweeps = load_scan(candidate)
            if sweeps and sweeps[0].ts >= start_s:
                scan_arg = candidate
                break
    if scan_arg and os.path.exists(scan_arg):
        scanned = scanned_build_cells(accumulate(load_scan(scan_arg)), built)
        shutil.copy2(scan_arg, os.path.join(out_dir, "agent_scan.jsonl"))

    made: list[str] = []
    _belief_graph(belief_lines, out_dir, made)
    _structure_graph(b2_lines, out_dir, made)
    _channels_graph(b1_lines, b2_lines, out_dir, made)
    _actions_graph(b1_lines, out_dir, made)
    _scan_graph(scan_report, out_dir, made)
    _clouds_graph(built, scanned, out_dir, made)

    scored = [line for line in b1_lines if line.get("scored")]
    summary = {
        "session_id": sid,
        "records": {"scored": len(scored), "context": len(b1_lines) - len(scored),
                    "structure": len(b2_lines), "belief": len(belief_lines)},
        "pixels_enriched": sum(1 for line in scored if line.get("h2d")),
        "h3d_records": sum(1 for line in b2_lines if line.get("h3d")),
        "built_cells": len(built),
        "scanned_of_built": len(scanned) if scanned is not None else None,
        "final_top_goal": belief_lines[-1]["top_goal"] if belief_lines else None,
        "graphs": made,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"session report  {sid}")
    print(f"  {len(made)} graphs -> {out_dir}")
    for name in made:
        print(f"    {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
