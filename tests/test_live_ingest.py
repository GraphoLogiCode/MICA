"""The live ingest layer: raw socket moments become strictly tick-ordered packets.
Covers the reorderer's contract (order restored within the hold, first arrival wins,
holes counted honestly) and the dict→packet parse parity with the JSONL reader."""
import io

from mica.capture.jsonl_ingest import packet_from_dict
from mica.capture.live_ingest import PacketReorderer, ordered_packets
from mica.capture.live_stream import read_moments
from mica.capture.sample_builds import wall_row_build
from mica.capture.synthetic import generate_session
from mica.capture.wire_synthetic import session_wire
from mica.contracts.serialize import packet_to_dict


def _session():
    return generate_session(wall_row_build())


def _dicts(session):
    return [packet_to_dict(p) for p in sorted(session.packets, key=lambda p: p.tick)]


def test_packet_round_trips_through_dict_form():
    # the wire fixture writes with packet_to_dict; the ingest reads with
    # packet_from_dict — the two must be exact inverses or every live test lies
    for packet in _session().packets:
        assert packet_from_dict(packet_to_dict(packet)) == packet


def test_in_order_stream_passes_straight_through():
    reorderer = PacketReorderer()
    out = list(ordered_packets(((d, None) for d in _dicts(_session())), reorderer))
    assert [p.tick for p, _ in out] == [p.tick for p in _session().packets]
    counts = reorderer.counts
    assert (counts.duplicates, counts.stale, counts.gap_ticks) == (0, 0, 0)
    assert counts.parsed == counts.released == len(out)


def test_out_of_order_within_the_hold_is_restored():
    dicts = _dicts(_session())
    shuffled = dicts[:3] + [dicts[5], dicts[4], dicts[3]] + dicts[6:]   # 3,4,5 -> 5,4,3
    reorderer = PacketReorderer()
    out = list(ordered_packets(((d, None) for d in shuffled), reorderer))
    assert [p.tick for p, _ in out] == sorted(d["tick"] for d in dicts)
    assert reorderer.counts.gap_ticks == 0 and reorderer.counts.stale == 0


def test_duplicate_tick_first_arrival_wins():
    # the second copy of tick 4 arrives while the first is still held (waiting for
    # tick 3) — dropped as a duplicate; a copy arriving after release is stale instead
    dicts = _dicts(_session())
    twice = dicts[:3] + [dicts[4], dict(dicts[4], wallclock_ms=999999), dicts[3]] + dicts[5:]
    reorderer = PacketReorderer()
    out = list(ordered_packets(((d, None) for d in twice), reorderer))
    assert [p.tick for p, _ in out] == [d["tick"] for d in dicts]
    kept = next(p for p, _ in out if p.tick == dicts[4]["tick"])
    assert kept.wallclock_ms == dicts[4]["wallclock_ms"]   # not the imposter
    assert reorderer.counts.duplicates == 1
    assert reorderer.counts.stale == 0


def test_copy_arriving_after_release_counts_as_stale():
    dicts = _dicts(_session())
    twice = dicts[:4] + [dict(dicts[3], wallclock_ms=999999)] + dicts[4:]
    reorderer = PacketReorderer()
    out = list(ordered_packets(((d, None) for d in twice), reorderer))
    assert [p.tick for p, _ in out] == [d["tick"] for d in dicts]
    assert reorderer.counts.stale == 1


def test_hole_that_outlives_the_hold_becomes_a_counted_gap():
    session = _session()
    ticks = sorted(p.tick for p in session.packets)
    missing = ticks[2]
    dicts = [d for d in _dicts(session) if d["tick"] != missing]
    reorderer = PacketReorderer(depth=4)   # small hold so the test session can fill it
    out = list(ordered_packets(((d, None) for d in dicts), reorderer))
    assert [p.tick for p, _ in out] == [t for t in ticks if t != missing]
    assert reorderer.counts.gap_ticks == 1


def test_moment_arriving_after_its_turn_is_dropped_as_stale():
    dicts = _dicts(_session())
    late = dicts[:1] + dicts[2:5] + [dicts[1]] + dicts[5:]   # tick 1 shows up way late
    reorderer = PacketReorderer(depth=2)                     # force release past the hole
    out = list(ordered_packets(((d, None) for d in late), reorderer))
    assert [p.tick for p, _ in out] == sorted(set(d["tick"] for d in dicts) - {dicts[1]["tick"]})
    assert reorderer.counts.stale == 1
    assert reorderer.counts.gap_ticks == 1                   # the hole it left behind


def test_wire_to_ordered_packets_end_to_end_with_late_frames():
    # The full front half of the live driver: mod-shaped bytes -> read_moments ->
    # ordered_packets. Late frames make moments arrive out of tick order on purpose;
    # the reorderer must hand the pipeline a clean ascending stream anyway.
    session = _session()
    ticks = sorted(p.tick for p in session.packets)
    wire = session_wire(session, framed_ticks=ticks[1:4], frame_delay=3)
    reorderer = PacketReorderer()
    out = list(ordered_packets(read_moments(io.BytesIO(wire)), reorderer))
    assert [p.tick for p, _ in out] == ticks
    framed = [p.tick for p, frame in out if frame is not None]
    assert framed == ticks[1:4]                              # frames rode along intact
    assert reorderer.counts.gap_ticks == 0


def test_wire_with_dropped_moments_yields_the_gap():
    session = _session()
    ticks = sorted(p.tick for p in session.packets)
    wire = session_wire(session, drop_ticks={ticks[3]})
    reorderer = PacketReorderer(depth=4)
    out = list(ordered_packets(read_moments(io.BytesIO(wire)), reorderer))
    assert [p.tick for p, _ in out] == [t for t in ticks if t != ticks[3]]
    assert reorderer.counts.gap_ticks == 1
