"""Play the mod's part on the wire: encode a CapturedSession as live-stream bytes.

The live protocol tests need traffic that looks exactly like the mod's — including
its awkward moments: frames arriving late, frames never arriving, whole moments
dropped under backpressure. No game produces those on demand, so this module builds
the byte stream from a synthetic session with knobs for each misbehavior. It is a
test fixture (kept out of the engine per project convention), used by the live
protocol/ingest/pipeline tests.

Wire format (mirrors B0CaptureClient.framed): [1 byte type][4-byte big-endian length]
[payload]; type 0 = packet JSON, type 1 = [4-byte tick][raw RGBA].
"""
from __future__ import annotations

import json
import struct
from typing import Iterable

from ..contracts.manifest import CapturedSession
from ..contracts.serialize import packet_to_dict

_HEADER_TYPE_PACKET = 0
_HEADER_TYPE_FRAME = 1


def framed(msg_type: int, payload: bytes) -> bytes:
    """One wire message: type byte + length + payload."""
    return bytes([msg_type]) + struct.pack(">I", len(payload)) + payload


def packet_message(packet_dict: dict) -> bytes:
    return framed(_HEADER_TYPE_PACKET, json.dumps(packet_dict).encode("utf-8"))


def frame_message(tick: int, rgba: bytes) -> bytes:
    return framed(_HEADER_TYPE_FRAME, struct.pack(">I", tick) + rgba)


def session_wire(
    session: CapturedSession,
    *,
    drop_ticks: Iterable[int] = (),
    framed_ticks: Iterable[int] = (),
    lost_frame_ticks: Iterable[int] = (),
    frame_delay: int = 0,
    width: int = 2,
    height: int = 1,
) -> bytes:
    """The session as the mod would send it, with misbehavior knobs.

    drop_ticks        moments the mod never sent (backpressure drop) — a real gap.
    framed_ticks      ticks whose packet declares a frame (synthetic packets carry
                      none by default); each gets a matching frame message.
    lost_frame_ticks  of the framed ticks, those whose frame message never arrives —
                      the socket reader must eventually yield them frameless.
    frame_delay       how many later packets go out before a tick's frame does; with
                      frameless packets in between this makes moments arrive out of
                      tick order, which is the case the reorderer exists for.
    """
    drop = set(drop_ticks)
    with_frame = set(framed_ticks)
    lost = set(lost_frame_ticks)
    rgba = bytes(width * height * 4)

    messages: list[bytes] = []
    pending: list[tuple[int, bytes]] = []   # (packets still to go out first, frame msg)
    for packet in sorted(session.packets, key=lambda p: p.tick):
        if packet.tick in drop:
            continue
        raw = packet_to_dict(packet)
        if packet.tick in with_frame:
            raw["client"]["pov_frame"] = {"path": "", "width": width, "height": height}
        messages.append(packet_message(raw))
        ready = [frame for wait, frame in pending if wait <= 0]
        pending = [(wait - 1, frame) for wait, frame in pending if wait > 0]
        messages.extend(ready)
        if packet.tick in with_frame and packet.tick not in lost:
            if frame_delay <= 0:
                messages.append(frame_message(packet.tick, rgba))
            else:
                pending.append((frame_delay, frame_message(packet.tick, rgba)))
    messages.extend(frame for _, frame in pending)   # whatever is still owed at the end
    return b"".join(messages)
