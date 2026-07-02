"""Check the live frame channel order (ISSUES U-5).

The mod sends live frames as raw bytes, and if the byte order were wrong the
picture would arrive with red and blue swapped — every saved color would be a lie.
The disk PNGs can't catch this, so we check with a known color on a live frame:

    1. Launch the mod and enter a world.
    2. Fill the whole screen with red — stand against a wall of red wool or red concrete.
    3. python scripts/live_color_check.py

It reads one frame and says whether red is what actually arrived.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture.live_stream import as_rgba, stream_moments   # noqa: E402

_DOMINANCE = 1.5  # a channel must beat the others by this factor to count as "the color seen"


def main() -> None:
    print("waiting for a live frame ... (mod running, in a world, screen filled with red)")
    for packet, frame_bytes in stream_moments():
        if frame_bytes is None:
            continue
        pixels = as_rgba(packet, frame_bytes)
        red = float(pixels[..., 0].mean())
        green = float(pixels[..., 1].mean())
        blue = float(pixels[..., 2].mean())
        print(f"tick {packet['tick']}: mean R={red:.0f} G={green:.0f} B={blue:.0f}")
        if red > green * _DOMINANCE and red > blue * _DOMINANCE:
            print("RED dominates -> channel order is correct. U-5 PASSES.")
        elif blue > red * _DOMINANCE and blue > green * _DOMINANCE:
            print("BLUE dominates -> channels are swapped. U-5 FAILS: flip the byte order "
                  "in frameMessage (mod) or as_rgba (Python).")
        else:
            print("No channel clearly dominates - the screen wasn't solidly red. "
                  "Face a red wall so it fills the view, then rerun.")
        return
    print("stream ended before a frame arrived")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped.")
    except ConnectionRefusedError:
        print("no live stream - is the mod running and in a world?")
