"""Rehearse the LIVE pipeline against a recorded session — no game needed.

    python scripts/replay_live_feed.py <session.jsonl> [--port 25567] [--speed 1]
                                       [--grow-jsonl] [--late-ok]

Plays the mod's part on a localhost socket: every recorded moment goes out in the
real wire format at game pace (20 ticks/second x --speed), and every recorded POV
frame goes out as the raw RGBA bytes the mod would have sent. `run_live.py` attaches
to it exactly as it would to the game — same socket, same ingest, same pipeline —
so the whole live path (including the GPU model channels: --pixels for h2d/s_goal,
--h3d for the shape embedding) can be watched producing changing numbers on demand.

Two flags make a LATE attach rehearsable (the recovery case, where run_live starts
partway into a session):
  --grow-jsonl   stage the session file EMPTY and append each moment as it plays —
                 the way the real mod writes its disk copy in parallel — so a
                 mid-feed attach catches up from a true prefix, not the whole file
  --late-ok      play on schedule from the start whether or not a consumer is
                 connected; whoever attaches receives the feed from that point on
                 (the real socket's semantics)

SAFETY: live mode writes its logs NEXT TO the session it attaches to, which would
overwrite the banked artifacts of a raw capture. So this script first stages a
complete clone (jsonl + manifest + snapshots + frames) into capture/rehearsal/ and
tells you to attach there:

    python scripts/run_live.py --pixels --h3d --session <staged jsonl>

capture/rehearsal/ is derived and disposable; capture/raw/ is never written.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture.wire_synthetic import frame_message, packet_message  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REHEARSAL = os.path.join(_ROOT, "capture", "rehearsal")
_TICK_S = 0.05                 # 20 Hz game pace
_MAX_STEP_S = 1.0              # a long recorded gap never stalls the rehearsal this long


def _flag(name: str, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def _stage(jsonl: str, grow: bool = False) -> tuple[str, list[tuple[int, str, str]]]:
    """A clone under capture/rehearsal/<id>/ — run_live writes THERE.

    Snapshot timing matters: at real attach time only the BASE snapshot exists;
    the mod writes the later ones mid-session (and run_live seeds from the newest
    on disk — pre-staging them all would make the finished build read as terrain).
    So the base snapshot is staged now, and the later ones are returned as
    (tick, src, dst) for the feeder to drop in when the feed reaches their ticks.

    With grow=True the session FILE starts empty too and the feeder appends each
    moment as it plays — the shape a real mid-session disk copy has, which is what
    run_live's late-attach catch-up reads."""
    source_dir = os.path.dirname(os.path.abspath(jsonl))
    with open(jsonl.replace(".jsonl", ".manifest.json"), encoding="utf-8") as handle:
        session_id = json.load(handle)["session_id"]
    stage_dir = os.path.join(_REHEARSAL, session_id)
    staged_jsonl = os.path.join(stage_dir, f"{session_id}.jsonl")
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)        # derived + disposable: every rehearsal starts clean
    os.makedirs(stage_dir, exist_ok=True)
    if grow:
        open(staged_jsonl, "w", encoding="utf-8").close()
    else:
        shutil.copy2(jsonl, staged_jsonl)
    shutil.copy2(jsonl.replace(".jsonl", ".manifest.json"),
                 os.path.join(stage_dir, f"{session_id}.manifest.json"))
    payload_dir = os.path.join(source_dir, session_id)
    pending: list[tuple[int, str, str]] = []
    if os.path.isdir(payload_dir):
        frames_dir = os.path.join(payload_dir, "frames")
        if os.path.isdir(frames_dir):   # the end-of-session gate checks frames on disk
            shutil.copytree(frames_dir, os.path.join(stage_dir, session_id, "frames"),
                            dirs_exist_ok=True)
        snaps = sorted(glob.glob(os.path.join(payload_dir, "snapshots", "*.json")),
                       key=lambda p: int(os.path.basename(p)[:-5]))
        staged_snaps = os.path.join(stage_dir, session_id, "snapshots")
        os.makedirs(staged_snaps, exist_ok=True)
        if snaps:
            shutil.copy2(snaps[0], staged_snaps)          # the base, present at attach
            for path in snaps[1:]:
                tick = int(os.path.basename(path)[:-5])
                pending.append((tick, path, os.path.join(staged_snaps, os.path.basename(path))))
    print(f"staged: {stage_dir} (base snapshot now; {len(pending)} more drip in mid-feed)")
    return staged_jsonl, pending


def _frame_bytes(raw: dict, source_dir: str) -> bytes | None:
    """The recorded PNG for this moment, decoded back to the raw RGBA the wire carries.
    The frame's recorded width/height are corrected to the actual image, so the
    consumer's reshape always matches."""
    ref = raw.get("client", {}).get("pov_frame")
    if not ref or not ref.get("path"):
        return None
    path = ref["path"]
    if not os.path.isabs(path):
        path = os.path.join(source_dir, path)
    if not os.path.exists(path):
        return None
    from PIL import Image
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        ref["width"], ref["height"] = rgba.size
        return rgba.tobytes()


def main() -> int:
    positional = [a for i, a in enumerate(sys.argv[1:], 1)
                  if not a.startswith("--") and sys.argv[i - 1] not in ("--port", "--speed")]
    if not positional:
        print("usage: replay_live_feed.py <session.jsonl> [--port 25567] [--speed 1]"
              " [--grow-jsonl] [--late-ok]")
        return 1
    jsonl = positional[0]
    port = int(_flag("--port", 25567))
    speed = float(_flag("--speed", 1.0))
    grow = "--grow-jsonl" in sys.argv
    late_ok = "--late-ok" in sys.argv
    source_dir = os.path.dirname(os.path.abspath(jsonl))

    staged, pending_snapshots = _stage(jsonl, grow=grow)
    with open(jsonl, encoding="utf-8") as handle:
        entries = [(line.rstrip("\n"), json.loads(line)) for line in handle if line.strip()]
    entries.sort(key=lambda pair: pair[1]["tick"])
    framed = sum(1 for _, m in entries if m.get("client", {}).get("pov_frame"))
    print(f"serving {len(entries)} moments ({framed} with frames) on 127.0.0.1:{port}"
          f" at {speed}x game pace" + ("  (session file grows as it plays)" if grow else ""))
    print(f"attach with:  python scripts/run_live.py --pixels --h3d --session \"{staged}\"")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(1)
    # One consumer, held in a box so the accept can happen in the background:
    # with --late-ok the feed plays on schedule from the start and whoever
    # attaches receives it from that point on — the real socket's semantics.
    holder: dict = {"conn": None}
    if late_ok:
        def _accept():
            try:
                conn, _ = server.accept()
                holder["conn"] = conn
                print("consumer attached — feeding from here on")
            except OSError:
                pass                        # server closed with no one attaching
        threading.Thread(target=_accept, daemon=True).start()
        print("playing on schedule; run_live may attach at any time (--late-ok)")
    else:
        print("waiting for run_live to connect ...")
        conn, _ = server.accept()
        holder["conn"] = conn
        print("consumer attached — playing")

    grow_handle = open(staged, "a", encoding="utf-8") if grow else None
    sent = frames_sent = played = 0
    started = time.monotonic()
    try:
        previous_tick = entries[0][1]["tick"]
        for line, raw in entries:
            step = min((raw["tick"] - previous_tick) * _TICK_S / speed, _MAX_STEP_S / speed)
            if step > 0:
                time.sleep(step)
            previous_tick = raw["tick"]
            while pending_snapshots and raw["tick"] >= pending_snapshots[0][0]:
                _, src, dst = pending_snapshots.pop(0)   # the mod "writes" its snapshot now
                shutil.copy2(src, dst)
            if grow_handle is not None:
                # the ORIGINAL line, before any of the mutations below — this is
                # the mod's parallel disk write, and the catch-up reads it
                grow_handle.write(line + "\n")
                grow_handle.flush()
            played += 1
            if holder["conn"] is None:
                continue                     # no consumer yet: the world still "happens"
            rgba = _frame_bytes(raw, source_dir)
            if rgba is None and raw.get("client", {}).get("pov_frame"):
                raw["client"]["pov_frame"] = None     # frame file gone: honest frameless moment
            try:
                holder["conn"].sendall(packet_message(raw))
                sent += 1
                if rgba is not None:
                    holder["conn"].sendall(frame_message(raw["tick"], rgba))
                    frames_sent += 1
            except (BrokenPipeError, ConnectionResetError, OSError):
                print("consumer disconnected early")
                try:
                    holder["conn"].close()
                except OSError:
                    pass
                holder["conn"] = None
                if not late_ok:
                    break
            if sent and sent % 200 == 0:
                print(f"  {sent}/{len(entries)} moments, {frames_sent} frames,"
                      f" {time.monotonic() - started:.0f}s elapsed")
    finally:
        for _, src, dst in pending_snapshots:          # whatever the feed never reached
            shutil.copy2(src, dst)                     # ... the disk gate still sees it
        if grow_handle is not None:
            for line, _ in entries[played:]:           # complete the disk copy the same way
                grow_handle.write(line + "\n")
            grow_handle.close()
        if holder["conn"] is not None:
            holder["conn"].close()                     # EOF = clean session end
        server.close()
    print(f"done: {sent} of {played} moments sent, {frames_sent} frames"
          f" in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
