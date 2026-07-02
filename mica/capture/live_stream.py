"""Read the mod's live B0 stream and yield whole moments as they happen.

This is the real-time counterpart to jsonl_ingest: instead of reading a finished
recording from disk, it connects to the running mod over a localhost socket and yields
each tick's packet (and frame) the moment it is produced — no disk in the path. D1/D2
consume this exactly like they consume a replayed recording, so the same perception code
runs live or offline.

Wire format (one message): [1 byte type][4-byte big-endian length][payload]
  type 0 = packet : payload is the B0 JSON (UTF-8), same shape jsonl_ingest reads
  type 1 = frame  : payload is [4-byte big-endian tick][raw RGBA bytes, row-major]
A packet that has a frame is held until its frame arrives, then yielded together, so the
caller always gets a complete moment. Frames may be dropped when the consumer lags (the
mod keeps only the freshest) — those ticks yield without a frame, never disappear.
Moments can arrive slightly out of tick order: a frameless packet yields at once while
an earlier packet is still waiting for its frame, so a live consumer must read the
tick field rather than assume arrival order.
"""
from __future__ import annotations

import json
import socket
import struct
from typing import Iterator

_HEADER = struct.Struct(">BI")   # type byte + payload length
_TICK = struct.Struct(">I")
_MAX_WAITING = 16                 # cap on packets awaiting a (possibly dropped) frame


def _read_exactly(reader, n: int) -> bytes | None:
    """Read exactly n bytes, or None if the stream closes first."""
    parts = []
    remaining = n
    while remaining > 0:
        chunk = reader.read(remaining)
        if not chunk:
            return None
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def stream_moments(host: str = "127.0.0.1", port: int = 25567) -> Iterator[tuple[dict, bytes | None]]:
    """Connect and yield (packet, frame_bytes) per live moment.

    `packet` is the parsed B0 JSON (a dict in the same shape as a recorded line).
    `frame_bytes` is the raw RGBA buffer for that tick, or None on ticks with no frame
    (menu / paused / throttled). Use `as_rgba(packet, frame_bytes)` to get an array.
    """
    sock = socket.create_connection((host, port))
    reader = sock.makefile("rb")
    try:
        yield from read_moments(reader)
    finally:
        reader.close()
        sock.close()


def read_moments(reader) -> Iterator[tuple[dict, bytes | None]]:
    """Yield (packet, frame_bytes) from any binary reader carrying the wire format.

    Split out from the socket so the protocol can be tested without a live mod.
    """
    waiting: dict[int, dict] = {}   # tick -> packet still missing its frame
    while True:
        header = _read_exactly(reader, _HEADER.size)
        if header is None:
            return
        msg_type, length = _HEADER.unpack(header)
        payload = _read_exactly(reader, length)
        if payload is None:
            return
        if msg_type == 0:
            packet = json.loads(payload)
            if packet["client"]["pov_frame"] is None:
                yield packet, None
            else:
                waiting[packet["tick"]] = packet
                if len(waiting) > _MAX_WAITING:
                    # Its frame was dropped by the mod. The symbolic half is still a
                    # real moment, so hand it over frameless instead of losing it.
                    stale = waiting.pop(next(iter(waiting)))
                    yield stale, None
        elif msg_type == 1:
            (tick,) = _TICK.unpack(payload[: _TICK.size])
            packet = waiting.pop(tick, None)
            if packet is not None:
                yield packet, payload[_TICK.size:]


def as_rgba(packet: dict, frame_bytes: bytes):
    """Reshape a frame's raw bytes into an (H, W, 4) uint8 RGBA array (needs numpy)."""
    import numpy as np

    meta = packet["client"]["pov_frame"]
    return np.frombuffer(frame_bytes, dtype=np.uint8).reshape(meta["height"], meta["width"], 4)
