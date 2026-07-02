"""One moment of recorded play: what the player saw and did, and what changed in the world.

Everything carries the same timestamp, and every later step is built from this.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BlockOp(Enum):
    """The two kinds of change the capture records: the player placing or breaking a block.

    The world changes in other ways too (water flows, fire spreads) — the mod records
    only the player's own hand, because that is the behavior intent is read from.
    """

    PLACE = "place"
    BREAK = "break"


@dataclass(frozen=True)
class BlockPos:
    """Where a block is. Whole numbers, because blocks sit on a grid."""

    x: int
    y: int
    z: int


@dataclass(frozen=True)
class PlayerPos:
    """Where the player is. Decimals, because the player moves smoothly, not block by block."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class CrosshairTarget:
    """What the player is aiming at. All blank means they're aiming at empty space."""

    block_pos: BlockPos | None
    face: str | None
    entity: str | None


@dataclass(frozen=True)
class InputState:
    """The keys and mouse the player is using right now — the first sign of what they mean to do."""

    keys: tuple[str, ...]
    mouse_buttons: tuple[str, ...]
    # The mod writes these as 0.0 (a documented stub) — look changes live in yaw/pitch instead.
    mouse_dx: float
    mouse_dy: float


@dataclass(frozen=True)
class FrameRef:
    """Points to a saved picture of the player's view, kept in a separate file because pictures are large."""

    path: str
    width: int
    height: int


@dataclass(frozen=True)
class BlockEvent:
    # Numbered 0, 1, 2, ... within a session, in the order the recorder saw them. Two
    # checks depend on that exact numbering: the missing-change check compares the ids
    # against 0..declared-1, and single consumption means each id lands in one evidence
    # record. A gap or reuse breaks both.
    event_id: int
    pos: BlockPos
    block_type: str
    op: BlockOp
    actor: str


@dataclass(frozen=True)
class ClientObservation:
    """What the player's own game shows and does this moment.

    It keeps its own clock time, because the player's clock and the game server's
    clock can disagree slightly. Several fields can be None: a recording that only
    watches the server (not the player's own game) can't see the keys, crosshair,
    hotbar, or menu — and the coverage check reports those gaps.
    """

    capture_wallclock_ms: float
    pov_frame: FrameRef | None
    input_state: InputState | None
    yaw: float
    pitch: float
    crosshair_target: CrosshairTarget | None
    held_item: str | None
    hotbar: tuple[str, ...] | None
    gui_open: bool | None


@dataclass(frozen=True)
class ServerObservation:
    """What the game server reports this moment.

    Break events are exact (read from the game after the block is gone). Place events
    are currently INFERRED from the right-click and can rarely be wrong — a click on a
    chest or at build height records a placement that never happened. Structure
    rebuilding (D2) must wait for the authoritative place capture (ISSUES D-1).
    """

    player_pos: PlayerPos | None
    block_events: tuple[BlockEvent, ...]
    # (item, amount) pairs — the mod writes an empty list today (a documented stub).
    inventory_delta: tuple[tuple[str, int], ...]
    dimension: str
    biome: str


@dataclass(frozen=True)
class ObservationPacket:
    """Everything captured for one moment of play, from both the player's game and the server."""

    tick: int
    # The server's clock time, used only by the drift check. In singleplayer the game
    # and the recorder share one clock, so this equals the client stamp and drift reads 0.
    wallclock_ms: int
    client: ClientObservation
    server: ServerObservation
