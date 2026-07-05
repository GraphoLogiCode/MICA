"""Find recordings on disk. One helper so every tool agrees on what "the newest capture" is.

Without this, each script rolled its own newest-file logic and some picked up D1's
`.evidence2d.jsonl` output as if it were a B0 capture (then crashed). They all call this now.
"""
from __future__ import annotations

import glob
import os


# Derived artifacts that sit next to captures with a .jsonl suffix — never captures.
_DERIVED_SUFFIXES = (".evidence2d.jsonl", ".evidence3d.jsonl", ".fused.jsonl", ".belief.jsonl")


def newest_capture(raw_dir: str) -> str | None:
    """The newest raw B0 recording (`.jsonl`) in `raw_dir`, ignoring derived artifacts
    (evidence, fused, belief logs — everything a pipeline run writes next to a capture).

    Returns the path, or None if there are no captures. A caller that also needs the
    `.manifest.json` sidecar should check for it separately.
    """
    jsonls = sorted(glob.glob(os.path.join(raw_dir, "*.jsonl")), key=os.path.getmtime)
    captures = [path for path in jsonls
                if not path.endswith(_DERIVED_SUFFIXES)]
    return captures[-1] if captures else None
