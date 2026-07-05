"""D1 — the 2D behavior perception stream: B0 packets in, B1 (Evidence2D) records out.

What it does, in one pass over the tick stream:
  1. SEGMENT the per-tick stream into macro-actions — contiguous runs of the same behavior
     (a placing burst, a navigation stretch, an idle pause).
  2. CLASSIFY each segment into one MacroAction by rule. Block events are lossless server
     truth, so PLACE/BREAK are exact; NAVIGATE/INSPECT/IDLE are read from inputs + motion.
  3. For each segment, gather PRE-ACTION evidence — state_feats + focus from a window that
     ends strictly before the action starts (the snapshot rule), plus the recent
     macro-action history — and emit one scored Evidence2D (a correction step).
  4. Inside any run longer than a second, also emit UNSCORED context records about once a
     second (scored=False, no events). The tracker and the gate read these between
     corrections — attention and state drift while the player walks or pauses — but they
     are never scored evidence and never train a likelihood head.

Nothing here trains and nothing loads a model: h2d/s_goal stay None and are filled by the
frozen VPT/MineCLIP heads in Layer 2. The segmenter is Evidence2DStream — a machine fed one
packet per tick that releases each record the moment it is knowable, so the same code runs
against the live socket during play and against recordings. `evidence_stream` is the
recording driver: it sorts a captured session and feeds it through that machine; the golden-
equivalence tests hold the two paths bit-identical.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator

from ..contracts.b0 import BlockOp, ObservationPacket, is_agent_actor
from ..contracts.b1 import Evidence2D, Focus, MacroAction, StateFeats

_WINDOW_TICKS = 20          # 1 second of pre-action evidence (20 ticks/second)
_PERIODIC_TICKS = 20        # one unscored context record per second inside a long run
_HISTORY_LEN = 5           # how many recent macro-actions travel in state_feats
# A camera turn bigger than this (degrees, in one tick) counts as looking around (INSPECT).
# 1.0° is sensitive — over half the real session's records came out INSPECT/IDLE — so this
# is a tuning knob the idle-precision measurement will revisit, not a settled value.
_CAMERA_DEGREES = 1.0
_POSITION_EPS = 0.05       # movement smaller than this is standing still, not navigating
_MOVE_KEYS = ("forward", "back", "left", "right", "jump", "sprint")

# How much to trust each rule. PLACE/BREAK ride on lossless events. IDLE gets 1.0 even
# though it is inferred (from the absence of input and motion) — a v1 choice pending the
# idle-precision measurement (ISSUES B1-F7). SCAFFOLD isn't emitted in v1, so it isn't
# listed; the lookup defaults safely rather than KeyError if a future rule produces it.
_CONFIDENCE = {
    MacroAction.PLACE: 1.0,
    MacroAction.BREAK: 1.0,
    MacroAction.IDLE: 1.0,
    MacroAction.NAVIGATE: 0.9,
    MacroAction.INSPECT: 0.7,
}
_CONFIDENCE_DEFAULT = 0.7


def _pos_changed(now, before) -> bool:
    return (abs(now.x - before.x) > _POSITION_EPS
            or abs(now.y - before.y) > _POSITION_EPS
            or abs(now.z - before.z) > _POSITION_EPS)


def _camera_moved(now, before) -> bool:
    return abs(now.yaw - before.yaw) > _CAMERA_DEGREES or abs(now.pitch - before.pitch) > _CAMERA_DEGREES


def _human_events(packet: ObservationPacket):
    """The tick's block events with the agent's own excluded (A7, the do-operator rule
    in contracts/b0.py): what MICA places must never read as human behavior, or the
    belief would confirm itself. Human-only sessions pass through unchanged."""
    return tuple(e for e in packet.server.block_events if not is_agent_actor(e.actor))


def _tick_behavior(packet: ObservationPacket, prev: ObservationPacket | None) -> MacroAction:
    """What the player did on this one tick — the instantaneous label runs are built from.

    Order matters: block events outrank motion (you can walk while placing), and a tick
    carrying both a place and a break labels PLACE. The position fallback catches motion
    without keys — which includes falling and being pushed, not only walking on purpose.
    Only the HUMAN's block events count: a tick where only the agent placed something
    classifies from the human's inputs and motion, exactly as if no event had occurred.
    """
    events = _human_events(packet)
    if any(e.op is BlockOp.PLACE for e in events):
        return MacroAction.PLACE
    if any(e.op is BlockOp.BREAK for e in events):
        return MacroAction.BREAK
    keys = packet.client.input_state.keys if packet.client.input_state else ()
    moving = any(key in _MOVE_KEYS for key in keys)
    if not moving and prev is not None and packet.server.player_pos and prev.server.player_pos:
        moving = _pos_changed(packet.server.player_pos, prev.server.player_pos)
    if moving:
        return MacroAction.NAVIGATE
    if prev is not None and _camera_moved(packet.client, prev.client):
        return MacroAction.INSPECT
    return MacroAction.IDLE


def _state_feats(window: list[ObservationPacket], recent: tuple[str, ...]) -> StateFeats:
    """Symbolic state over the pre-action window: held item + hotbar at the last moment,
    net movement/look across the window, and the macro-actions that came just before."""
    first, last = window[0], window[-1]
    pos_now, pos_then = last.server.player_pos, first.server.player_pos
    pos_delta = (
        (pos_now.x - pos_then.x, pos_now.y - pos_then.y, pos_now.z - pos_then.z)
        if pos_now and pos_then else (0.0, 0.0, 0.0)
    )
    return StateFeats(
        held_item=last.client.held_item or "minecraft:air",
        hotbar=last.client.hotbar or (),
        pos_delta=pos_delta,
        yaw_delta=last.client.yaw - first.client.yaw,
        pitch_delta=last.client.pitch - first.client.pitch,
        recent_actions=recent,
    )


def _focus(window: list[ObservationPacket]) -> Focus:
    """The block the crosshair rested on at the end of the window, and how long it lingered."""
    target = window[-1].client.crosshair_target
    block = target.block_pos if target is not None else None
    dwell = 0
    if block is not None:
        for packet in reversed(window):
            here = packet.client.crosshair_target
            if here is not None and here.block_pos == block:
                dwell += 1
            else:
                break
    return Focus(block=block, dwell_ticks=dwell)


@dataclass
class _OpenRun:
    """The macro-action currently in progress — what the stream knows but hasn't closed.

    A run of PLACE/BREAK ticks can't emit its scored record until the run ends, because
    block-event ids keep arriving while the burst continues. `deferred` holds that
    almost-finished record; `held` keeps any ~1 Hz context records that came due during
    the burst, so everything still leaves in the order the batch driver used.
    """

    action: MacroAction
    t0: int                          # first tick of the run (the action starts here)
    t1: int                          # last tick seen so far
    events: list[int]                # block-event ids the run has consumed so far
    recent: tuple[str, ...]          # macro-action history from before this run
    deferred: Evidence2D | None = None
    held: list[Evidence2D] = field(default_factory=list)


class Evidence2DStream:
    """D1 as a tick-fed machine: give it one packet per tick, get records as they
    become knowable. This is THE segmenter — the offline `evidence_stream` below is
    just a loop that feeds a recording through it.

    Feed order must be strictly increasing ticks. Ticks may be missing (a live socket
    is allowed to drop moments); windows then simply hold fewer packets. On gap-free
    input the records are bit-identical to the old batch driver's, in the same order —
    the golden-equivalence tests hold the two together.
    """

    def __init__(self, window: int = _WINDOW_TICKS):
        self._window = window
        self._buffer: dict[int, ObservationPacket] = {}   # the last window+1 ticks, by tick
        self._prev: ObservationPacket | None = None       # previous packet fed (any run)
        self._history: deque[str] = deque(maxlen=_HISTORY_LEN)
        self._run: _OpenRun | None = None
        self._last_tick: int | None = None

    def feed(self, packet: ObservationPacket) -> tuple[Evidence2D, ...]:
        """Take the next tick's packet; return every record this tick lets us release."""
        if self._last_tick is not None and packet.tick <= self._last_tick:
            raise ValueError(f"ticks must strictly increase: got {packet.tick} after {self._last_tick}")
        self._last_tick = packet.tick

        # Keep only the ticks a window starting at this tick could ever reach back to.
        self._buffer[packet.tick] = packet
        cutoff = packet.tick - self._window
        while self._buffer:
            oldest = next(iter(self._buffer))
            if oldest >= cutoff:
                break
            del self._buffer[oldest]

        here = _tick_behavior(packet, self._prev)
        self._prev = packet

        released: list[Evidence2D] = []
        if self._run is not None and here is self._run.action:
            run = self._run
            run.t1 = packet.tick
            run.events.extend(e.event_id for e in _human_events(packet))
            # One unscored context record per second inside a long run, exactly at the
            # ticks the batch driver used (t0+20, t0+40, ...).
            if packet.tick > run.t0 and (packet.tick - run.t0) % _PERIODIC_TICKS == 0:
                context = self._context_record(packet.tick)
                if context is not None:
                    if run.deferred is not None:
                        run.held.append(context)   # keep batch order: scored leaves first
                    else:
                        released.append(context)
        else:
            released.extend(self._close_run())
            released.extend(self._open_run(packet, here))
        return tuple(released)

    def finish(self) -> tuple[Evidence2D, ...]:
        """The stream ended (recording exhausted, or the live session closed):
        release whatever the still-open run was holding."""
        return tuple(self._close_run())

    @property
    def current_behavior(self) -> MacroAction | None:
        """The macro-action in progress right now (None before the first packet) —
        the live readout a safety gate needs for its idle/active decision."""
        return self._run.action if self._run is not None else None

    def _open_run(self, packet: ObservationPacket, here: MacroAction) -> list[Evidence2D]:
        """Start a new run at this tick and build its scored record from the window
        of ticks strictly before it (the snapshot rule)."""
        recent = tuple(self._history)          # history BEFORE this run, like the batch driver
        self._history.append(here.value)
        self._run = _OpenRun(
            action=here,
            t0=packet.tick,
            t1=packet.tick,
            events=[e.event_id for e in _human_events(packet)],
            recent=recent,
        )
        win = [self._buffer[t] for t in range(packet.tick - self._window, packet.tick)
               if t in self._buffer]
        if not win:
            # Nothing exists before the action (the very first run) — no scored record,
            # same as batch; any events this run carries will show up as unconsumed.
            return []
        record = Evidence2D(
            tick_range=(win[0].tick, win[-1].tick),
            a_hat=here,
            a_hat_conf=_CONFIDENCE.get(here, _CONFIDENCE_DEFAULT),
            idle=here is MacroAction.IDLE,
            state_feats=_state_feats(win, recent),
            focus=_focus(win),
            scored=True,
            event_ids=(),
        )
        if here is MacroAction.PLACE or here is MacroAction.BREAK:
            # A build run's event ids aren't final until the run closes — hold the
            # record. Everything else about it is already final.
            self._run.deferred = record
            return []
        # Any other behavior can't carry block events (an event tick always classifies
        # PLACE/BREAK), so event_ids=() is final and the record can leave right now.
        return [record]

    def _close_run(self) -> list[Evidence2D]:
        """End the open run: attach the final event ids to a deferred build record and
        release it, followed by any context records that were waiting behind it."""
        run, self._run = self._run, None
        if run is None:
            return []
        released: list[Evidence2D] = []
        if run.deferred is not None:
            released.append(replace(run.deferred, event_ids=tuple(run.events)))
        released.extend(run.held)
        return released

    def _context_record(self, tick: int) -> Evidence2D | None:
        """An unscored look at the ongoing run — the window here may include the run's
        own ticks (nothing scored, nothing trained on, so nothing can leak)."""
        run = self._run
        win = [self._buffer[t] for t in range(tick - self._window + 1, tick + 1)
               if t in self._buffer]
        if not win:
            return None
        return Evidence2D(
            tick_range=(win[0].tick, win[-1].tick),
            a_hat=run.action,
            a_hat_conf=_CONFIDENCE.get(run.action, _CONFIDENCE_DEFAULT),
            idle=run.action is MacroAction.IDLE,
            state_feats=_state_feats(win, run.recent),
            focus=_focus(win),
            scored=False,
            event_ids=(),
        )


def evidence_stream(packets: Iterable[ObservationPacket], window: int = _WINDOW_TICKS) -> Iterator[Evidence2D]:
    """Yield one scored Evidence2D per macro-action, each from a window ending strictly
    before it, plus one unscored context record per second inside any long run.

    A macro-action at the very first tick has no prior window to read, so it is skipped for
    emission (it still counts as history for the next action) — the snapshot rule needs a
    'before' to exist.
    """
    # The recording driver: order the session, then feed the same machine the live path
    # uses, one packet at a time. Recordings fit in memory, so sorting here is fine; an
    # unbounded live source goes through Evidence2DStream directly.
    stream = Evidence2DStream(window)
    for packet in sorted(packets, key=lambda packet: packet.tick):
        yield from stream.feed(packet)
    yield from stream.finish()
