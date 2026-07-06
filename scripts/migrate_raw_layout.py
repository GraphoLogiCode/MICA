"""Reorganize capture/raw into per-date/per-session folders (dry-run by default).

    python scripts/migrate_raw_layout.py            # show the plan, move nothing
    python scripts/migrate_raw_layout.py --apply    # do it

Each flat session `fabric-<id>.*` at the raw root moves into
`capture/raw/<date>/fabric-<id>/`, together with its `fabric-<id>/` snapshots+frames
subdir (moved wholesale so the jsonl<->snapshots siblinghood the pipeline assumes is
preserved — every writer and snapshot glob keeps working unchanged). Global
cross-session artifacts (reports, figures, labels.json, arm2_cache.jsonl, the
agent-* logs, pre_cascade_a/) stay at the root.

Idempotent: a session whose dated folder already exists is skipped. Writes
`capture/raw/_migration_manifest.json` (from->to) so the move is auditable and
reversible. The session_store resolver reads both layouts, so running the pipeline
before, during, or after this migration all work.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture import session_store                              # noqa: E402

_RAW = session_store.RAW_ROOT


def _flat_sessions() -> list[str]:
    """Session ids with a flat `<root>/fabric-*.jsonl` still at the raw root."""
    ids = []
    for path in sorted(glob.glob(os.path.join(_RAW, "fabric-*.jsonl"))):
        name = os.path.basename(path)
        # a raw capture is <id>.jsonl; skip derived siblings (<id>.evidence2d.jsonl etc.)
        if name.count(".") == 1:
            ids.append(name[:-len(".jsonl")])
    return ids


def _session_moves(session_id: str) -> list[tuple[str, str]]:
    """(source, dest) for every flat file + the snapshots/frames subdir of a session."""
    dest_dir = os.path.join(_RAW, session_store.session_date(session_id), session_id)
    moves = []
    for path in sorted(glob.glob(os.path.join(_RAW, f"{session_id}.*"))):
        moves.append((path, os.path.join(dest_dir, os.path.basename(path))))
    subdir = os.path.join(_RAW, session_id)
    if os.path.isdir(subdir):
        moves.append((subdir, os.path.join(dest_dir, session_id)))
    return moves


def main() -> int:
    apply = "--apply" in sys.argv
    sessions = _flat_sessions()
    if not sessions:
        print("no flat fabric-* sessions at the raw root — nothing to migrate")
        return 0

    manifest = []
    skipped = 0
    planned = 0
    for session_id in sessions:
        dest_dir = os.path.join(_RAW, session_store.session_date(session_id), session_id)
        if os.path.isdir(dest_dir):
            skipped += 1
            continue
        moves = _session_moves(session_id)
        planned += 1
        rel = os.path.relpath(dest_dir, _RAW)
        print(f"{'MOVE' if apply else 'PLAN'} {session_id} -> {rel}/  ({len(moves)} items)")
        if apply:
            os.makedirs(dest_dir, exist_ok=True)
            for source, target in moves:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.move(source, target)
                manifest.append({"from": os.path.relpath(source, _RAW),
                                 "to": os.path.relpath(target, _RAW)})

    print(f"\n{planned} session(s) {'moved' if apply else 'to move'}, {skipped} already nested")
    if apply:
        manifest_path = os.path.join(_RAW, "_migration_manifest.json")
        existing = []
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as handle:
                existing = json.load(handle).get("moves", [])
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({"moves": existing + manifest}, handle, indent=2)
        print(f"  manifest -> {os.path.relpath(manifest_path, _RAW)}")
    else:
        print("  (dry run — pass --apply to move)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
