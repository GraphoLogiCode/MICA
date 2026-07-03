"""Pretend recording: turns a planned set of block changes into captured moments.

This is the engine. It lets us test everything before a real game is hooked up.
Ready-made sample recordings live in sample_builds.py, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..contracts.b0 import (
    BlockEvent,
    BlockOp,
    BlockPos,
    ClientObservation,
    CrosshairTarget,
    InputState,
    ObservationPacket,
    PlayerPos,
    ServerObservation,
)
from ..contracts.manifest import CapturedSession, SessionManifest
from ..contracts.timebase import MS_PER_TICK

_TRAILING_TICKS = 2  # quiet steps after the last change — real recordings keep running past the
                     # last block, and the trailing idle is what closes the final action segment
_GROUND_Y = 64  # the height the pretend player stands at, in a flat world
_PLACEMENT_PITCH = 45.0  # the player looks down to place or break a block
_IDLE_INPUT = InputState(keys=(), mouse_buttons=(), mouse_dx=0.0, mouse_dy=0.0)
_PLACE_INPUT = InputState(keys=(), mouse_buttons=("right",), mouse_dx=0.0, mouse_dy=0.0)
_BREAK_INPUT = InputState(keys=(), mouse_buttons=("left",), mouse_dx=0.0, mouse_dy=0.0)


@dataclass(frozen=True)
class ScriptedPlacement:
    """One planned block change: which block, where, when, and whether it's placed or broken."""

    tick: int
    pos: BlockPos
    block_type: str
    op: BlockOp = BlockOp.PLACE


@dataclass(frozen=True)
class ScriptedBuild:
    """A full plan for a pretend recording: the block changes plus a few settings."""

    session_id: str
    held_item: str
    placements: tuple[ScriptedPlacement, ...]
    session_start_ms: int = 0
    alignment_jitter_ms: float = 0.0  # how far the player's clock sits off the server's, on purpose
    server_lag_ms: float = 0.0  # how late the server runs behind the ideal 20-steps-per-second pace
    dimension: str = "overworld"
    biome: str = "plains"


def _server_wallclock_ms(build: ScriptedBuild, tick: int) -> float:
    # The server's real clock: the ideal time for this step, plus however late the server is running.
    return build.session_start_ms + tick * MS_PER_TICK + build.server_lag_ms


def _jitter(build: ScriptedBuild, tick: int) -> float:
    # Flip the offset +/- each step so it isn't always in the same direction.
    sign = 1.0 if tick % 2 == 0 else -1.0
    return build.alignment_jitter_ms * sign


def _changes_at(build: ScriptedBuild, tick: int) -> tuple[ScriptedPlacement, ...]:
    """Every block change planned for this step (a step may have several, or none)."""
    return tuple(placement for placement in build.placements if placement.tick == tick)


def _held_at(build: ScriptedBuild, tick: int) -> str:
    """A builder equips what they are about to place — the held item switches to the
    next change's block as it comes up, so pre-action windows see the real hand."""
    upcoming = [p for p in build.placements
                if p.op is BlockOp.PLACE and 0 <= p.tick - tick <= 20]
    if upcoming:
        return min(upcoming, key=lambda p: p.tick).block_type
    return build.held_item


def _client_at(build: ScriptedBuild, tick: int) -> ClientObservation:
    """What the player's game shows this step: idle, unless a block change is due."""
    # The player's clock tracks the server's clock, off by a small offset.
    capture_ms = _server_wallclock_ms(build, tick) + _jitter(build, tick)
    held = _held_at(build, tick)
    idle = ClientObservation(
        capture_wallclock_ms=capture_ms,
        pov_frame=None,
        input_state=_IDLE_INPUT,
        yaw=0.0,
        pitch=0.0,
        crosshair_target=CrosshairTarget(block_pos=None, face=None, entity=None),
        held_item=held,
        hotbar=(held,),
        gui_open=False,
    )
    changes = _changes_at(build, tick)
    if not changes:
        return idle
    first = changes[0]
    action_input = _BREAK_INPUT if first.op is BlockOp.BREAK else _PLACE_INPUT
    # A change moves these things; the held item becomes what is being placed, because
    # a real builder holds the block they use — the D1 held-item channel carries that.
    return replace(
        idle,
        input_state=action_input,
        pitch=_PLACEMENT_PITCH,
        held_item=first.block_type,
        crosshair_target=CrosshairTarget(block_pos=first.pos, face="top", entity=None),
    )


def _server_at(
    build: ScriptedBuild, tick: int, next_event_id: int
) -> tuple[ServerObservation, int]:
    """What the server reports this step — one numbered event per block change.

    Hands back the next id number so every block change across the whole recording
    gets its own, in order.
    """
    player_pos = PlayerPos(x=0.0, y=float(_GROUND_Y), z=0.0)
    events: list[BlockEvent] = []
    deltas: list[tuple[str, int]] = []
    event_id = next_event_id
    for change in _changes_at(build, tick):
        events.append(
            BlockEvent(
                event_id=event_id,
                pos=change.pos,
                block_type=change.block_type,
                op=change.op,
                actor=build.session_id,
            )
        )
        # Placing uses up one of that block; breaking gives one back.
        deltas.append((change.block_type, -1 if change.op is BlockOp.PLACE else 1))
        event_id += 1
    server = ServerObservation(
        player_pos=player_pos,
        block_events=tuple(events),
        inventory_delta=tuple(deltas),
        dimension=build.dimension,
        biome=build.biome,
    )
    return server, event_id


def generate_session(build: ScriptedBuild) -> CapturedSession:
    """Turn a build plan into a full pretend recording: the setup plus one captured moment per step."""
    # default=0 lets a walk-only plan (no changes) still make a few idle moments.
    last_tick = max((placement.tick for placement in build.placements), default=0)
    manifest = SessionManifest(
        session_id=build.session_id,
        session_start_ms=build.session_start_ms,
    )
    packets: list[ObservationPacket] = []
    next_event_id = 0
    for tick in range(last_tick + _TRAILING_TICKS + 1):
        server, next_event_id = _server_at(build, tick, next_event_id)
        packet = ObservationPacket(
            tick=tick,
            wallclock_ms=int(_server_wallclock_ms(build, tick)),
            client=_client_at(build, tick),
            server=server,
        )
        packets.append(packet)
    return CapturedSession(
        manifest=manifest,
        packets=tuple(packets),
        declared_event_count=next_event_id,
    )
