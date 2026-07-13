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


# Who caused a block change. The actor is the causing player's USERNAME (world
# processes appear under names like "fire"), and the rig pins the two usernames
# exactly (user decision 2026-07-04): the human plays as HumanBuilder, the agent
# joins the server as MICA_AI — replicas as MICA_AI_1, MICA_AI_2, ... The agent
# being a real named player is the point: the capture attributes events by
# username, so everything MICA ever places is tagged correctly with no special
# writer path. (Collision with a stranger named MICA_AI is a non-risk on the
# controlled offline rig — HumanBuilder is the pinned human account.)
#
# The rule this enables (A7, the do-operator): EVIDENCE consumers (D1's
# macro-actions, D2's feature world) read human events only, so the agent's own
# blocks can never become intent evidence; VERIFICATION consumers (the snapshot
# monitor's shadow world, the B0 gate, event-id continuity) read ALL events,
# because the real world contains the agent's blocks and the proofs must match
# reality. Filtering happens at consumption, never at capture — the raw log
# always records everything.
HUMAN_ACTOR = "HumanBuilder"   # the rig's canonical human username (not enforced on
                               # old captures — anything non-agent reads as human)
AGENT_ACTOR = "MICA_AI"        # the agent's username; replicas append _1, _2, ...


def is_agent_actor(actor: str) -> bool:
    """True when a block event was caused by MICA's own agent rather than the human:
    exactly MICA_AI, or MICA_AI_<suffix> (so MICA_AI_2 matches and a hypothetical
    'MICA_AIX' does not)."""
    return actor == AGENT_ACTOR or actor.startswith(AGENT_ACTOR + "_")


@dataclass(frozen=True)
class BlockEvent:
    # Numbered 0, 1, 2, ... within a session, in the order the recorder saw them. Two
    # checks depend on that exact numbering: the missing-change check compares the ids
    # against 0..declared-1, and single consumption means each id lands in one evidence
    # record. A gap or reuse breaks both. The numbering covers EVERY actor's events —
    # agent events keep their ids (continuity checks span them); they are excluded
    # later, at the evidence boundary (see the actor rule above).
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
    # The player's whole inventory as (item, count) pairs, aggregated over main +
    # hotbar + offhand slots. Sampled sparsely (the mod writes it ~once a second and
    # when it changes), so most ticks carry None — and every capture from before
    # 2026-07-10 carries None on all of them. Consumers treat it exactly like the
    # pixel channels: optional, absent-tolerant.
    inventory: tuple[tuple[str, int], ...] | None = None


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
