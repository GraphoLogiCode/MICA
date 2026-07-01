"""Connect to the mod's live B0 stream and print each moment as it arrives.

A quick way to confirm the live feed works: launch the mod, enter a world, then run
    python scripts/live_stream_probe.py
You should see one line per tick, scrolling at ~20/second while you play, each showing
the live latency (mod capture time vs now), held item, frame size, and block events.
Ctrl-C to stop.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture.live_stream import stream_moments   # noqa: E402


def main() -> None:
    print("connecting to 127.0.0.1:25567 ... (start the mod and enter a world)")
    moments = 0
    for packet, frame_bytes in stream_moments():
        moments += 1
        client = packet["client"]
        lag_ms = time.time() * 1000.0 - client["capture_wallclock_ms"]
        frame = "none"
        if frame_bytes is not None:
            meta = client["pov_frame"]
            frame = f"{meta['width']}x{meta['height']} ({len(frame_bytes)} B)"
        events = len(packet["server"]["block_events"])
        if moments % 10 == 0 or events:        # don't flood: every 10th tick, plus any build action
            print(f"tick {packet['tick']:>6}  lag {lag_ms:6.0f} ms  held {client['held_item']:<28} "
                  f"frame {frame:<22} events {events}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped.")
    except ConnectionRefusedError:
        print("no live stream — is the mod running and in a world?")
