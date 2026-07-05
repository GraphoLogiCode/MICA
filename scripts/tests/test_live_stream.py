"""Round-trips the live-stream wire format: bytes framed exactly as the mod sends them,
parsed back by the client. Verifies the protocol without needing a running game.
The mod-side framing lives in mica/capture/wire_synthetic.py (shared test fixture)."""
import io

from mica.capture.live_stream import read_moments
from mica.capture.wire_synthetic import frame_message as _frame_msg
from mica.capture.wire_synthetic import packet_message as _packet_msg


def _packet(tick: int, *, with_frame: bool) -> dict:
    frame = {"path": "f.png", "width": 2, "height": 1} if with_frame else None
    return {
        "tick": tick,
        "wallclock_ms": 1000 + tick,
        "client": {"capture_wallclock_ms": 1000 + tick, "pov_frame": frame, "held_item": "minecraft:stone"},
        "server": {"block_events": []},
    }


def test_packet_and_frame_pair_into_one_moment():
    rgba = bytes(range(8))   # 2x1 RGBA = 8 bytes
    stream = io.BytesIO(_packet_msg(_packet(0, with_frame=True)) + _frame_msg(0, rgba))
    moments = list(read_moments(stream))
    assert len(moments) == 1
    packet, frame = moments[0]
    assert packet["tick"] == 0
    assert frame == rgba


def test_packet_without_frame_yields_immediately():
    stream = io.BytesIO(_packet_msg(_packet(5, with_frame=False)))
    assert list(read_moments(stream)) == [(_packet(5, with_frame=False), None)]


def test_frame_whose_packet_was_dropped_is_skipped():
    # under backpressure the mod can drop a packet but still send its frame; that frame
    # has nothing to attach to and must not produce a moment (or crash).
    stream = io.BytesIO(_frame_msg(9, bytes(8)))
    assert list(read_moments(stream)) == []


def test_late_frame_still_matches_its_packet():
    rgba = bytes(8)
    stream = io.BytesIO(
        _packet_msg(_packet(0, with_frame=True))
        + _packet_msg(_packet(1, with_frame=False))   # a frameless tick arrives between
        + _frame_msg(0, rgba)                          # frame 0 lands late
    )
    moments = list(read_moments(stream))
    assert [packet["tick"] for packet, _ in moments] == [1, 0]
    assert moments[1][1] == rgba


def test_packet_whose_frame_never_arrives_yields_frameless():
    # Under backpressure the mod drops frames but not packets. When the wait buffer
    # overflows, the oldest packet must still come out (frameless) — a lagging
    # consumer may lose pixels, never whole moments.
    overflow = b"".join(_packet_msg(_packet(t, with_frame=True)) for t in range(18))
    moments = list(read_moments(io.BytesIO(overflow)))
    assert [packet["tick"] for packet, _ in moments] == [0, 1]   # 16 still waiting at EOF
    assert all(frame is None for _, frame in moments)
