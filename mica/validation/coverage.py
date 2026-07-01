"""Checks that a recording actually contains everything the later steps will need to read from it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..contracts.b0 import ObservationPacket

SessionProbe = Callable[[tuple[ObservationPacket, ...]], bool]


@dataclass(frozen=True)
class FieldProbe:
    """One field a later step needs, with a test for whether the recording has it.

    Listed as data, not as if/else code, so adding a new step needs no code change.
    """

    label: str
    present: SessionProbe


@dataclass(frozen=True)
class ReaderRequirement:
    """A later step, and the list of fields it reads from the recording."""

    name: str
    probes: tuple[FieldProbe, ...]


@dataclass(frozen=True)
class CoverageResult:
    """For one later step: was everything it needs present, and if not, what's missing."""

    reader: str
    satisfied: bool
    missing: tuple[str, ...]


def _every_moment(packets, predicate) -> bool:
    # A field that belongs in every moment: every captured moment must actually have it.
    return bool(packets) and all(predicate(packet) for packet in packets)


def _some_moment(packets, predicate) -> bool:
    # An "as available" field — present only when relevant (the crosshair points at
    # something only some of the time). The recording supplies it if any moment has
    # it; a capture where it never appears (a server-only capture) is what we flag.
    return any(predicate(packet) for packet in packets)


def _every_block_change(packets, predicate) -> bool:
    # A field that belongs on each block change: every block change must have it.
    # True when there are no block changes — a recording with no building is still valid.
    return all(predicate(event) for packet in packets for event in packet.server.block_events)


# The step that rebuilds the 3D structure reads each block change. We don't require a
# recording to contain any block changes, but the ones it has must be complete.
_STRUCTURE_READER = ReaderRequirement(
    name="D2 3D structure stream",
    probes=(
        FieldProbe("block change position", lambda ps: _every_block_change(ps, lambda e: e.pos is not None)),
        FieldProbe("block change type", lambda ps: _every_block_change(ps, lambda e: bool(e.block_type))),
        FieldProbe("block change place-or-break", lambda ps: _every_block_change(ps, lambda e: e.op is not None)),
        FieldProbe("block change id", lambda ps: _every_block_change(ps, lambda e: e.event_id is not None)),
    ),
)

# The step that reads the player's actions reads their inputs in every moment.
_ACTION_READER = ReaderRequirement(
    name="D1 2D symbolic half",
    probes=(
        FieldProbe("inputs", lambda ps: _every_moment(ps, lambda p: p.client.input_state is not None)),
        FieldProbe("crosshair", lambda ps: _some_moment(ps, lambda p: p.client.crosshair_target is not None)),
        FieldProbe("held item", lambda ps: _every_moment(ps, lambda p: p.client.held_item is not None)),
        FieldProbe("hotbar", lambda ps: _every_moment(ps, lambda p: p.client.hotbar is not None)),
        FieldProbe("look direction", lambda ps: _every_moment(ps, lambda p: p.client.yaw is not None and p.client.pitch is not None)),
        FieldProbe("menu-open flag", lambda ps: _every_moment(ps, lambda p: p.client.gui_open is not None)),
        FieldProbe("player position", lambda ps: _every_moment(ps, lambda p: p.server.player_pos is not None)),
    ),
)

# The step that looks at the screen image needs saved frames. They're as-available:
# the mod throttles them and skips menus, so we require them present at least sometimes.
_IMAGE_READER = ReaderRequirement(
    name="D1 2D pixel half (VPT/MineCLIP)",
    probes=(
        FieldProbe("pov_frame", lambda ps: _some_moment(ps, lambda p: p.client.pov_frame is not None)),
    ),
)

ALL_READERS = (_STRUCTURE_READER, _ACTION_READER, _IMAGE_READER)

# Steps the recording must satisfy on its own. The screen-image step waits on the
# real game capture, so we report it but don't require it.
GATING_READERS = (_STRUCTURE_READER, _ACTION_READER)


def evaluate_reader(
    packets: tuple[ObservationPacket, ...], requirement: ReaderRequirement
) -> CoverageResult:
    """Which of a step's needed fields the recording supplies."""
    missing = tuple(probe.label for probe in requirement.probes if not probe.present(packets))
    return CoverageResult(reader=requirement.name, satisfied=not missing, missing=missing)


def evaluate_coverage(
    packets: tuple[ObservationPacket, ...],
) -> tuple[CoverageResult, ...]:
    """Run the check for every later step."""
    return tuple(evaluate_reader(packets, reader) for reader in ALL_READERS)


def gating_gaps(packets: tuple[ObservationPacket, ...]) -> tuple[str, ...]:
    """List the missing fields among the steps the recording must satisfy on its own."""
    gaps: list[str] = []
    for reader in GATING_READERS:
        result = evaluate_reader(packets, reader)
        for label in result.missing:
            gaps.append(f"{result.reader}: {label}")
    return tuple(gaps)
