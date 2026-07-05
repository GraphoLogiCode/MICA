"""Turn the live socket's moments into the strict tick order the perception cores need.

The mod's live feed is honest but not tidy: moments can arrive slightly out of tick
order (a frameless packet overtakes one still waiting for its frame), a moment can be
missing outright (the mod drops the oldest queued message when the consumer lags), and
in principle a message could arrive twice. Evidence2DStream requires strictly
increasing ticks, so this sits between the socket and the pipeline: parse each moment
into an ObservationPacket, hold a small buffer, release in tick order, and count
everything that had to be dropped or skipped — the counts go into the live run's
summary, because a session with holes is not proof-grade.

The hold depth (32) is deliberately larger than the socket reader's own out-of-order
window (live_stream._MAX_WAITING = 16), so a legally-late moment is always still
waiting here when its turn comes. Only a moment the mod truly never sent turns into a
counted gap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from ..contracts.b0 import ObservationPacket
from .jsonl_ingest import packet_from_dict

_REORDER_DEPTH = 32

Moment = tuple[ObservationPacket, "bytes | None"]


@dataclass
class IngestCounts:
    """What the reorderer saw and what it had to give up on."""

    parsed: int = 0        # moments taken off the socket
    released: int = 0      # moments handed to the pipeline, in strict tick order
    duplicates: int = 0    # a tick that arrived again while its first copy was held
    stale: int = 0         # a tick that arrived after newer ticks were already released
    gap_ticks: int = 0     # ticks given up on: never arrived before the hold filled up

    def to_dict(self) -> dict:
        return {"parsed": self.parsed, "released": self.released,
                "duplicates": self.duplicates, "stale": self.stale,
                "gap_ticks": self.gap_ticks}


class PacketReorderer:
    """Restore strict tick order with a bounded hold; first arrival of a tick wins."""

    def __init__(self, depth: int = _REORDER_DEPTH):
        self._depth = depth
        self._held: dict[int, Moment] = {}
        self._last_released: int | None = None
        self.counts = IngestCounts()

    def push(self, packet_dict: dict, frame: bytes | None) -> list[Moment]:
        """Take one raw moment off the socket; return whatever is now releasable."""
        packet = packet_from_dict(packet_dict)
        self.counts.parsed += 1
        tick = packet.tick
        if self._last_released is not None and tick <= self._last_released:
            self.counts.stale += 1        # newer ticks already went out; too late
            return []
        if tick in self._held:
            self.counts.duplicates += 1   # first arrival wins
            return []
        self._held[tick] = (packet, frame)
        return self._release()

    def drain(self) -> list[Moment]:
        """The stream ended: release everything still held, in order, counting holes."""
        released = []
        for tick in sorted(self._held):
            if self._last_released is not None and tick > self._last_released + 1:
                self.counts.gap_ticks += tick - self._last_released - 1
            released.append(self._held.pop(tick))
            self._last_released = tick
        self.counts.released += len(released)
        return released

    def _release(self) -> list[Moment]:
        released = []
        while self._held:
            smallest = min(self._held)
            consecutive = (self._last_released is None
                           or smallest == self._last_released + 1)
            if consecutive:
                pass                       # its turn has come
            elif len(self._held) > self._depth:
                # The hole before `smallest` outlived the hold — that moment was
                # really dropped by the mod, not merely late. Give up on it.
                self.counts.gap_ticks += smallest - self._last_released - 1
            else:
                break                      # wait: the missing tick may still arrive
            released.append(self._held.pop(smallest))
            self._last_released = smallest
        self.counts.released += len(released)
        return released


def ordered_packets(
    moments: Iterable[tuple[dict, bytes | None]],
    reorderer: PacketReorderer | None = None,
) -> Iterator[Moment]:
    """The live driver's front step: socket moments in, tick-ordered packets out.

    Pass your own PacketReorderer to read its counts afterward; frames ride along
    untouched (the symbolic pipeline ignores them; the future pixel head reads them).
    """
    reorderer = reorderer if reorderer is not None else PacketReorderer()
    for packet_dict, frame in moments:
        yield from reorderer.push(packet_dict, frame)
    yield from reorderer.drain()
