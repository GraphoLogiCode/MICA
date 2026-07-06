"""Where a capture session's files live on disk — the one place that knows.

The raw capture tree is organized by date and session:

    capture/raw/<YYYY-MM-DD>/<session_id>/<session_id>.jsonl
                                          <session_id>.evidence2d.jsonl
                                          ... (all derived siblings) ...
                                          <session_id>/snapshots|frames/

Everything that needs to READ a session's files by id goes through here, so the
layout is defined once. WRITERS don't need this module — they derive their output
directory from the input jsonl's own location (sibling placement), so once a
session's jsonl sits in its dated dir, its derived outputs land there for free.

Two compatibility rules keep the migration safe and reversible:
  - legacy-flat fallback: if the dated dir doesn't exist yet but a flat
    <root>/<session_id>.jsonl does, resolve to the flat root — so un-migrated
    sessions and mid-migration state keep working;
  - non-dated ids (test fixtures like "livetest-0001") always resolve flat, since
    there is no date to organize them by.
"""
from __future__ import annotations

import os
import re

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_ROOT = os.path.join(_ROOT, "capture", "raw")

# fabric-YYYYMMDD-HHMMSS — the capture mod's session id shape.
_FABRIC_ID = re.compile(r"^fabric-(\d{4})(\d{2})(\d{2})-\d{6}$")


def session_date(session_id: str) -> str:
    """The YYYY-MM-DD folder a session sorts into. Raises for non-dated ids."""
    match = _FABRIC_ID.match(session_id)
    if not match:
        raise ValueError(f"{session_id!r} is not a dated fabric session id")
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def session_dir(session_id: str, root: str = RAW_ROOT) -> str:
    """The directory holding session_id's files.

    Nested `<root>/<date>/<session_id>/` once migrated (or for a brand-new dated
    session), else the flat `<root>` while the session is un-migrated or its id
    carries no date.
    """
    try:
        nested = os.path.join(root, session_date(session_id), session_id)
    except ValueError:
        return root                                   # non-dated id -> flat
    if os.path.isdir(nested):
        return nested
    if os.path.exists(os.path.join(root, f"{session_id}.jsonl")):
        return root                                   # legacy flat, not yet migrated
    return nested                                     # new dated session -> organized


def session_file(session_id: str, suffix: str, root: str = RAW_ROOT) -> str:
    """A session's file for one suffix, e.g. session_file(id, '.evidence2d.jsonl')."""
    return os.path.join(session_dir(session_id, root), f"{session_id}{suffix}")


def session_jsonl(session_id: str, root: str = RAW_ROOT) -> str:
    return session_file(session_id, ".jsonl", root)


def snapshots_dir(session_id: str, root: str = RAW_ROOT) -> str:
    """`<session dir>/<session_id>/snapshots` — the inner subdir kept sibling to the
    jsonl so the existing globs resolve unchanged after migration."""
    return os.path.join(session_dir(session_id, root), session_id, "snapshots")


def frames_dir(session_id: str, root: str = RAW_ROOT) -> str:
    return os.path.join(session_dir(session_id, root), session_id, "frames")


def resolve_frame(baked_path: str, session_id: str, capture_dir: str) -> str | None:
    """A frame file, tolerant of the capture tree having been relocated.

    Frame paths are stored ABSOLUTE in the raw jsonl (the mod writes them), so a
    moved session's baked paths go stale. Try the baked path first; then look for
    the frame by name in the session's own frames dir relative to where the jsonl
    actually sits now. Returns None if neither exists."""
    if baked_path and os.path.exists(baked_path):
        return baked_path
    candidate = os.path.join(capture_dir, session_id, "frames", os.path.basename(baked_path))
    return candidate if os.path.exists(candidate) else None
