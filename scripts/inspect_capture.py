"""Summarize a raw capture: how complete each B0 field is, and the known gaps.

Run from the project root:
    python scripts/inspect_capture.py            # newest recording
    python scripts/inspect_capture.py <file.jsonl>
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture.discovery import newest_capture   # noqa: E402

_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "capture", "raw")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else newest_capture(_RAW_DIR)
    if not path or not os.path.exists(path):
        print("no capture found")
        return

    moments = frames = crosshair = gui_open = block_events = 0
    held_items: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            packet = json.loads(line)
            client, server = packet["client"], packet["server"]
            moments += 1
            frames += client["pov_frame"] is not None
            crosshair += client["crosshair_target"] is not None
            gui_open += bool(client["gui_open"])
            block_events += len(server["block_events"])
            held_items.add(client["held_item"])

    pct = lambda n: f"{100 * n // max(moments, 1)}%"
    print(f"file:    {os.path.basename(path)}")
    print(f"moments: {moments}")
    print(f"  pov_frame present:    {frames}/{moments} ({pct(frames)})   -- null on menu ticks + throttled ticks")
    print(f"  crosshair on a block: {crosshair}/{moments} ({pct(crosshair)})   -- null when aiming at open air")
    print(f"  inventory/menu open:  {gui_open}/{moments} ({pct(gui_open)})")
    print(f"  block events:         {block_events}   -- player place/break captured by the mod")
    print(f"  held items seen:      {sorted(held_items)}")
    print("  known gaps: biome='unknown' (hardcoded), mouse_dx/dy=0.0 (not captured)")


if __name__ == "__main__":
    main()
