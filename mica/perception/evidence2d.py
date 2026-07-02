"""D1 — the 2D behavior perception stream: B0 packets in, B1 (Evidence2D) records out.

What it does, in one pass over a recorded session:
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
frozen VPT/MineCLIP heads in Layer 2. The input is a tick-ordered iterable of B0 packets — a
recorded CapturedSession. This driver is offline-batch (it sorts the whole session), so a
live source would need a rolling-window variant; the B1 records would be the same either way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from ..contracts.b0 import BlockOp, ObservationPacket
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


@dataclass(frozen=True)
class _Segment:
    a_hat: MacroAction
    t0: int                     # onset tick (the action starts here)
    t1: int                     # last tick of the run
    event_ids: tuple[int, ...]  # block events the run consumed


def _pos_changed(now, before) -> bool:
    return (abs(now.x - before.x) > _POSITION_EPS
            or abs(now.y - before.y) > _POSITION_EPS
            or abs(now.z - before.z) > _POSITION_EPS)


def _camera_moved(now, before) -> bool:
    return abs(now.yaw - before.yaw) > _CAMERA_DEGREES or abs(now.pitch - before.pitch) > _CAMERA_DEGREES


def _tick_behavior(packet: ObservationPacket, prev: ObservationPacket | None) -> MacroAction:
    """What the player did on this one tick — the instantaneous label runs are built from.

    Order matters: block events outrank motion (you can walk while placing), and a tick
    carrying both a place and a break labels PLACE. The position fallback catches motion
    without keys — which includes falling and being pushed, not only walking on purpose.
    """
    events = packet.server.block_events
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


def _segment(packets: tuple[ObservationPacket, ...]) -> list[_Segment]:
    """Group consecutive same-behavior ticks into macro-actions."""
    segments: list[_Segment] = []
    action: MacroAction | None = None
    t0 = t1 = 0
    events: list[int] = []
    prev: ObservationPacket | None = None
    for packet in packets:
        here = _tick_behavior(packet, prev)
        if here is not action:
            if action is not None:
                segments.append(_Segment(action, t0, t1, tuple(events)))
            action, t0, events = here, packet.tick, []
        t1 = packet.tick
        events.extend(e.event_id for e in packet.server.block_events)
        prev = packet
    if action is not None:
        segments.append(_Segment(action, t0, t1, tuple(events)))
    return segments


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


def evidence_stream(packets: Iterable[ObservationPacket], window: int = _WINDOW_TICKS) -> Iterator[Evidence2D]:
    """Yield one scored Evidence2D per macro-action, each from a window ending strictly
    before it, plus one unscored context record per second inside any long run.

    A macro-action at the very first tick has no prior window to read, so it is skipped for
    emission (it still counts as history for the next action) — the snapshot rule needs a
    'before' to exist.
    """
    # Offline-batch: materialize and tick-sort the whole session. A live source would need a
    # rolling-window variant instead — sorted() cannot run on an unbounded stream (ISSUES.md U-2).
    ordered = sorted(packets, key=lambda packet: packet.tick)
    by_tick = {packet.tick: packet for packet in ordered}
    history: list[str] = []
    for segment in _segment(tuple(ordered)):
        recent = tuple(history[-_HISTORY_LEN:])
        history.append(segment.a_hat.value)
        win = [by_tick[t] for t in range(segment.t0 - window, segment.t0) if t in by_tick]
        if win:
            yield Evidence2D(
                tick_range=(win[0].tick, win[-1].tick),
                a_hat=segment.a_hat,
                a_hat_conf=_CONFIDENCE.get(segment.a_hat, _CONFIDENCE_DEFAULT),
                idle=segment.a_hat is MacroAction.IDLE,
                state_feats=_state_feats(win, recent),
                focus=_focus(win),
                scored=True,
                event_ids=segment.event_ids,
            )
        # If win is empty, nothing exists before the action — skip the correction. If this
        # (first) segment carried block events, those ids then go unconsumed — ISSUES.md U-1.
        yield from _context_records(segment, by_tick, window, recent)


def _context_records(
    segment: _Segment, by_tick: dict[int, ObservationPacket], window: int, recent: tuple[str, ...]
) -> Iterator[Evidence2D]:
    """Unscored ~1 Hz context inside a long run — what the tracker and gate read between
    corrections. The window may include the run's own ticks (even rendered effects of
    already-scored events): nothing here is scored or trained on, so nothing can leak."""
    for p in range(segment.t0 + _PERIODIC_TICKS, segment.t1 + 1, _PERIODIC_TICKS):
        win = [by_tick[t] for t in range(p - window + 1, p + 1) if t in by_tick]
        if not win:
            continue
        yield Evidence2D(
            tick_range=(win[0].tick, win[-1].tick),
            a_hat=segment.a_hat,
            a_hat_conf=_CONFIDENCE.get(segment.a_hat, _CONFIDENCE_DEFAULT),
            idle=segment.a_hat is MacroAction.IDLE,
            state_feats=_state_feats(win, recent),
            focus=_focus(win),
            scored=False,
            event_ids=(),
        )
