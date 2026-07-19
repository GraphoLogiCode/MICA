"""Find recordings on disk. One helper so every tool agrees on what "the newest capture" is.

Without this, each script rolled its own newest-file logic and some picked up D1's
`.evidence2d.jsonl` output as if it were a B0 capture (then crashed). They all call this now.
"""
from __future__ import annotations

import glob
import os


def newest_capture(raw_dir: str) -> str | None:
    """The newest raw B0 recording (`.jsonl`) in `raw_dir`, or None when there is none.

    A capture is recognized by what it IS, not by what it isn't: `fabric-<stamp>.jsonl`
    with no extra suffix dots and a `.manifest.json` sidecar (the mod writes both at
    launch). The old blacklist of derived suffixes missed every OTHER .jsonl living in
    the raw dir — rig_log, agent-* logs, gate_trace, caches — so a no-session
    invocation of after_game once targeted "rig_log" as if it were a game (R-5,
    2026-07-18 rig review).
    """
    jsonls = sorted(glob.glob(os.path.join(raw_dir, "fabric-*.jsonl")),
                    key=os.path.getmtime)
    captures = [path for path in jsonls
                if os.path.basename(path).count(".") == 1
                and os.path.exists(path[:-len(".jsonl")] + ".manifest.json")]
    return captures[-1] if captures else None
