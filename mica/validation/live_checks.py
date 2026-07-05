"""The streaming half of the B0 gate: per-packet sanity while the game is running.

The full gate (b0_gate.py) needs a finished recording — a finalized manifest, the
declared event count, snapshot files on disk. A live consumer can't wait for any of
that, but most of what the gate checks is visible packet by packet: ticks should be
contiguous, the two clocks should agree, block-event ids should count 0,1,2,... with
none missing, and an action tick should have a tick before it to read evidence from.

So the split is: LiveGate raises problems the moment they are visible in the stream,
and the untouched b0_gate runs at session end against the disk recording the mod
writes in parallel — the disk copy stays the authoritative proof artifact. A LiveGate
problem does not stop the pipeline (the session keeps being useful for assistance);
it makes the run's summary say, honestly, that this session is not proof-grade.
"""
from __future__ import annotations

from ..contracts.b0 import ObservationPacket
from ..contracts.timebase import DEFAULT_ALIGNMENT_BOUND_MS, alignment_residual_ms

# The client fields D1 reads. Some are legitimately null on a given tick (crosshair
# on open air, frame on a menu tick) — so these are counted, never per-tick problems;
# a field that NEVER shows up is reported by summary() as a coverage hole.
_WATCHED_FIELDS = ("input_state", "crosshair_target", "held_item", "hotbar",
                   "pov_frame", "player_pos")


class LiveGate:
    """Feed it every packet (already in strict tick order); it returns the problems
    that packet just made visible and keeps running totals for the end-of-run summary."""

    def __init__(self, alignment_bound_ms: float = DEFAULT_ALIGNMENT_BOUND_MS):
        self._bound_ms = alignment_bound_ms
        self._first_tick: int | None = None
        self._prev_tick: int | None = None
        self._packets = 0
        self._gap_ticks = 0            # missing moments between seen ticks
        self._gap_stretches = 0        # how many separate holes
        self._max_residual_ms = 0.0
        self._residual_violations = 0
        self._next_event_id = 0        # ids must run 0,1,2,... with no holes
        self._event_id_skips = 0       # ids that never appeared (a dropped event!)
        self._events_seen = 0
        self._actions_without_prior = 0
        self._field_counts = {name: 0 for name in _WATCHED_FIELDS}

    def check(self, packet: ObservationPacket) -> tuple[str, ...]:
        """Problems this packet makes visible; empty means nothing new is wrong."""
        problems: list[str] = []
        tick = packet.tick

        if self._first_tick is None:
            self._first_tick = tick
        had_prior = self._prev_tick is not None and self._prev_tick == tick - 1
        if self._prev_tick is not None and tick > self._prev_tick + 1:
            missing = tick - self._prev_tick - 1
            self._gap_ticks += missing
            self._gap_stretches += 1
            problems.append(f"tick gap: {missing} moment(s) missing before tick {tick}")
        self._prev_tick = tick
        self._packets += 1

        residual = abs(alignment_residual_ms(packet.wallclock_ms,
                                             packet.client.capture_wallclock_ms))
        self._max_residual_ms = max(self._max_residual_ms, residual)
        if residual > self._bound_ms:
            self._residual_violations += 1
            problems.append(f"alignment residual {residual:.0f} ms exceeds bound at tick {tick}")

        for event in packet.server.block_events:
            self._events_seen += 1
            if event.event_id > self._next_event_id:
                skipped = event.event_id - self._next_event_id
                self._event_id_skips += skipped
                problems.append(
                    f"event id jumped {self._next_event_id} -> {event.event_id} at tick {tick}:"
                    f" {skipped} block event(s) were lost")
                self._next_event_id = event.event_id + 1
            elif event.event_id < self._next_event_id:
                problems.append(
                    f"event id {event.event_id} repeated or out of order at tick {tick}")
            else:
                self._next_event_id += 1

        if packet.server.block_events and not had_prior:
            # The snapshot rule needs a "before": an action on the first tick, or right
            # after a hole, has damaged (or no) pre-action evidence.
            self._actions_without_prior += 1
            problems.append(f"action at tick {tick} has no prior moment to read evidence from")

        client = packet.client
        for name in _WATCHED_FIELDS:
            value = getattr(client, name, None) if name != "player_pos" \
                else packet.server.player_pos
            if value is not None:
                self._field_counts[name] += 1

        return tuple(problems)

    def summary(self) -> dict:
        """Running totals for live_run.json; `clean` is the streaming-visible verdict."""
        never_seen = [name for name, count in self._field_counts.items()
                      if self._packets and count == 0 and name != "pov_frame"]
        return {
            "packets": self._packets,
            "first_tick": self._first_tick,
            "last_tick": self._prev_tick,
            "tick_gaps": {"missing_ticks": self._gap_ticks, "stretches": self._gap_stretches},
            "alignment": {"max_abs_ms": self._max_residual_ms,
                          "bound_ms": self._bound_ms,
                          "violations": self._residual_violations},
            "block_events": {"seen": self._events_seen,
                             "next_expected_id": self._next_event_id,
                             "ids_skipped": self._event_id_skips},
            "actions_without_prior_moment": self._actions_without_prior,
            "field_coverage": dict(self._field_counts),
            "fields_never_seen": never_seen,   # frames excluded: as-available by contract
            "clean": (self._gap_ticks == 0 and self._residual_violations == 0
                      and self._event_id_skips == 0 and self._actions_without_prior == 0
                      and not never_seen),
        }
