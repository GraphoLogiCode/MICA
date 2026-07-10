"""Sort stray capture outputs into their session folders (dry-run by default).

    python scripts/organize_raw.py                       # show the plan, move nothing
    python scripts/organize_raw.py --apply               # do it
    python scripts/organize_raw.py --apply --session ID  # one session only (rig chain)

After a game, two kinds of output can be left behind at the raw root instead of
sitting with their session:

  agent scans     agent-<NAME>.scan.*.jsonl — the agent's raycast sweeps. New files
                  name their session (the agent tags every sweep); old files carry
                  only an mtime stamp and are matched by wallclock: a scan belongs
                  to the latest session that started before its first sweep. A scan
                  moves INTO its session's folder, next to the capture jsonl, where
                  run_agent_scan.py already looks first. The agent's ACTIVE scan
                  file (agent-<NAME>.scan.jsonl) is rotated in place first when
                  it has gone quiet — never touched while the agent is writing.

  mixed traces    a <session>.gate_trace.jsonl holding MORE than one run (the read
                  counter restarts mid-file — the old append-mode bug, fixed at the
                  source in run_live/live_loop). Earlier runs' rows move to
                  previous_runs/run-NN/ so the main file is one run, as D5 §7 assumes.

Cross-session artifacts at the raw root (calibration reports, arm figures,
labels.json, cascade_status.json, rig_log.jsonl) are LEFT ALONE on purpose: their
writers pin those paths, and moving them would only make regeneration re-create
them at the root while readers stare at stale copies.

Every move is appended to capture/raw/organize_log.jsonl (ts, action, from, to),
so the whole reorganization is auditable and reversible by hand.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture import session_store                              # noqa: E402

_RAW = session_store.RAW_ROOT
# Leave a scan file alone unless it has been quiet this long — an active agent
# appends every 250 ms, so ten minutes of silence means the launch is over.
_ACTIVE_QUIET_S = 600.0
_SESSION_IN_NAME = re.compile(r"\.scan\.(fabric-\d{8}-\d{6})")


def _log_move(action: str, source: str, dest: str, root: str = _RAW) -> None:
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "action": action,
             "from": source, "to": dest}
    with open(os.path.join(root, "organize_log.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------ agent scans

def _session_starts(root: str = _RAW) -> list[tuple[float, str]]:
    """(start seconds, session id) for every capture with a manifest, sorted."""
    starts = []
    for path in glob.glob(os.path.join(root, "**", "fabric-*.manifest.json"),
                          recursive=True):
        if "previous_runs" in path:
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                meta = json.load(handle)
            starts.append((float(meta["session_start_ms"]) / 1000.0,
                           meta["session_id"]))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(starts)


def _tagged_session(scan_path: str) -> str | None:
    """The session id the scan's own sweep lines name, if any line does."""
    try:
        with open(scan_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("session"):
                    return row["session"]
    except OSError:
        return None
    return None


def _first_sweep_s(scan_path: str) -> float | None:
    try:
        with open(scan_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    return float(json.loads(line).get("ts", 0)) / 1000.0
                except ValueError:
                    return None
    except OSError:
        return None
    return None


def scan_owner(scan_path: str, starts: list[tuple[float, str]]) -> str | None:
    """Which session a rotated scan file belongs to, best evidence first:
    the sweeps' own session tag, then the session id in the filename, then
    wallclock (the latest session that started before the first sweep)."""
    tagged = _tagged_session(scan_path)
    if tagged:
        return tagged
    match = _SESSION_IN_NAME.search(os.path.basename(scan_path))
    if match:
        return match.group(1)
    first = _first_sweep_s(scan_path)
    if first is None:
        return None
    owner = None
    for start, session_id in starts:
        if start <= first:
            owner = session_id
        else:
            break
    return owner


def _rotate_stale_active(root: str, apply: bool, plan: list[str]) -> None:
    """An active scan file whose agent is long gone never got its launch-time
    rotation. Rotate it here (session-tagged name when possible) so it becomes a
    normal rotated file the mover below can place."""
    for path in glob.glob(os.path.join(root, "agent-*.scan.jsonl")):
        try:
            stat = os.stat(path)
        except OSError:
            continue
        if stat.st_size == 0 or time.time() - stat.st_mtime < _ACTIVE_QUIET_S:
            continue
        stamp = _tagged_session(path) or str(int(stat.st_mtime * 1000))
        rotated = path[: -len(".jsonl")] + f".{stamp}.jsonl"
        if os.path.exists(rotated):
            rotated = path[: -len(".jsonl")] + f".{stamp}.{int(stat.st_mtime * 1000)}.jsonl"
        plan.append(f"rotate  {os.path.basename(path)} -> {os.path.basename(rotated)}")
        if apply:
            os.rename(path, rotated)
            _log_move("rotate_stale_scan", path, rotated, root)


def organize_scans(root: str = _RAW, apply: bool = False,
                   only_session: str | None = None) -> list[str]:
    """Move every rotated agent scan into its owning session's folder."""
    plan: list[str] = []
    if only_session is None:
        _rotate_stale_active(root, apply, plan)
    starts = _session_starts(root)
    for path in sorted(glob.glob(os.path.join(root, "agent-*.scan.*.jsonl"))):
        owner = scan_owner(path, starts)
        if owner is None:
            plan.append(f"skip    {os.path.basename(path)} (no session matches it)")
            continue
        if only_session is not None and owner != only_session:
            continue
        dest_dir = session_store.session_dir(owner, root)
        if os.path.abspath(dest_dir) == os.path.abspath(root):
            # The owning session still sits flat at the root (not yet relocated);
            # moving the scan "next to it" would be a no-op — leave it for the
            # relocation pass to pick up later.
            plan.append(f"wait    {os.path.basename(path)} (session {owner} not "
                        "relocated yet)")
            continue
        dest = os.path.join(dest_dir, os.path.basename(path))
        if os.path.exists(dest):
            plan.append(f"skip    {os.path.basename(path)} (already in {owner})")
            continue
        plan.append(f"move    {os.path.basename(path)} -> "
                    f"{os.path.relpath(dest, root)}")
        if apply:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(path, dest)
            _log_move("adopt_scan", path, dest, root)
    return plan


def adopt_agent_scans(session_id: str, root: str = _RAW) -> None:
    """The after_game hook: right after a session is relocated into its dated
    folder, pull its agent scans in too. Quiet when there is nothing to do."""
    plan = organize_scans(root, apply=True, only_session=session_id)
    for line in plan:
        print(f"  scans: {line}")


# ------------------------------------------------------------ mixed gate traces

def split_multirun_trace(trace_path: str, apply: bool = False,
                         root: str = _RAW) -> list[str]:
    """Un-mix a gate trace holding several appended runs: every run but the LAST
    moves to previous_runs/run-NN/<name>, the file keeps only the final run.
    (The final run is the one whose belief log survived — earlier runs' belief
    files were truncated by the old append/overwrite mismatch.)"""
    with open(trace_path, encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    runs: list[list[str]] = []
    for line in lines:
        try:
            k = json.loads(line).get("k")
        except ValueError:
            k = None
        if k == 1 or not runs:
            runs.append([])
        runs[-1].append(line)
    if len(runs) <= 1:
        return []
    plan = []
    session_dir = os.path.dirname(os.path.abspath(trace_path))
    name = os.path.basename(trace_path)
    for number, run in enumerate(runs[:-1], start=1):
        dest_dir = os.path.join(session_dir, "previous_runs", f"run-{number:02d}")
        dest = os.path.join(dest_dir, name)
        plan.append(f"split   {name}: run {number} ({len(run)} rows) -> "
                    f"previous_runs/run-{number:02d}/")
        if apply:
            os.makedirs(dest_dir, exist_ok=True)
            with open(dest, "a", encoding="utf-8") as handle:
                handle.writelines(run)
            _log_move("split_trace_run", trace_path, dest, root)
    plan.append(f"keep    {name}: final run ({len(runs[-1])} rows) stays")
    if apply:
        tmp = trace_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.writelines(runs[-1])
        os.replace(tmp, trace_path)
    return plan


def organize_traces(root: str = _RAW, apply: bool = False,
                    only_session: str | None = None) -> list[str]:
    plan: list[str] = []
    pattern = (os.path.join(root, "**", f"{only_session}.gate_trace.jsonl")
               if only_session else
               os.path.join(root, "**", "fabric-*.gate_trace.jsonl"))
    for path in sorted(glob.glob(pattern, recursive=True)):
        if "previous_runs" in path:
            continue
        plan.extend(split_multirun_trace(path, apply, root))
    return plan


# ------------------------------------------------------------------------ main

def main() -> int:
    apply = "--apply" in sys.argv
    only = None
    if "--session" in sys.argv:
        only = sys.argv[sys.argv.index("--session") + 1]
    plan = organize_scans(_RAW, apply, only) + organize_traces(_RAW, apply, only)
    if not plan:
        print("nothing to organize — every output already sits with its session")
        return 0
    print(("applied:" if apply else "plan (dry run — pass --apply to move):"))
    for line in plan:
        print(f"  {line}")
    if not apply:
        print(f"  {sum(1 for l in plan if not l.startswith(('skip', 'wait', 'keep')))} "
              "move(s) pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
